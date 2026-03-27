import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.app.indexer.bm25_builder import tokenize, BM25Okapi
from backend.app.indexer.store import load_bm25, load_chunks
from backend.app.models.search import CodeChunk

REPO_ID = "jdereg--java-util"
REPO_NAME = "jdereg/java-util"
BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
LORA_PATH = str(
    ROOT / "benchmark" / "lora_training" / "output" / "rewriter_lora" / "final"
)
SAMPLES_PATH = ROOT / "benchmark" / "results" / "benchmark_samples.json"
RESULTS_DIR = ROOT / "benchmark" / "results"
K_VALUES = [1, 5, 10, 20]

REWRITE_PROMPT = (
    "Rewrite this search query into structured retrieval hints "
    "for a Java codebase.\n\nQuery: {query}\n\nOutput JSON:"
)


def generate(
    model: torch.nn.Module,
    tokenizer: torch.nn.Module,
    prompt: str,
    device: torch.device,
    max_new_tokens=256,
):
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def parse_rewrite(raw: str, original_query: str):
    json_str = raw.strip()
    if "```" in json_str:
        for part in json_str.split("```")[1:]:
            c = part.strip()
            if c.startswith("json"):
                c = c[4:].strip()
            if c.startswith("{"):
                json_str = c
                break
    start = json_str.find("{")
    end = json_str.rfind("}")
    if start != -1 and end > start:
        json_str = json_str[start : end + 1]
    try:
        data = json.loads(json_str)
        parts = []
        for key in ("search_queries", "keywords", "method_hints", "project_terms"):
            for v in data.get(key, []):
                if isinstance(v, str):
                    parts.append(v)
        if parts:
            return " ".join(parts), True
    except json.JSONDecodeError, TypeError:
        pass
    return f"{original_query} {raw.strip()[:300]}", False


def bm25_search(query_str: str, bm25: BM25Okapi, chunks: list[CodeChunk], top_k=20):
    tokens = tokenize(query_str)
    if not tokens:
        return [], []
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][: top_k * 3]
    files, methods, seen = [], [], set()
    for idx in top_indices:
        if scores[idx] <= 0:
            break
        chunk = chunks[int(idx)]
        if chunk.file_path not in seen:
            files.append(chunk.file_path)
            seen.add(chunk.file_path)
        if chunk.method_name and chunk.method_name not in methods:
            methods.append(chunk.method_name)
        if len(files) >= top_k:
            break
    return files, methods


def compute_metrics(samples, all_results):
    ground_truth = {s["event_id"]: set(s["changed_files"]) for s in samples}
    by_retriever = {}
    for r in all_results:
        by_retriever.setdefault(r["retriever"], []).append(r)

    metrics = {}
    for ret_name, results in by_retriever.items():
        m = {}
        for k in K_VALUES:
            hits, rr_sum, total = 0, 0.0, 0
            for r in results:
                gt = ground_truth.get(r["sample_id"], set())
                if not gt:
                    continue
                total += 1
                top_k_files = r["retrieved_files"][:k]
                if any(f in gt for f in top_k_files):
                    hits += 1
                for rank, f in enumerate(top_k_files, 1):
                    if f in gt:
                        rr_sum += 1.0 / rank
                        break
            m[f"recall@{k}"] = round(hits / total, 4) if total else 0
            m[f"mrr@{k}"] = round(rr_sum / total, 4) if total else 0
        m["total_samples"] = len(results)
        metrics[ret_name] = m
    return metrics


def run_mode_rewrite(samples, bm25, chunks, model, tokenizer, device, ret_name):
    results = []
    raw_map = {}
    json_valid = 0
    max_k = max(K_VALUES)
    files_map = {}

    for i, s in enumerate(samples):
        if (i + 1) % 10 == 1:
            print(f"  [{i + 1}/{len(samples)}] {s['query'][:60]}...", flush=True)
        prompt = REWRITE_PROMPT.format(query=s["query"])
        raw = generate(model, tokenizer, prompt, device)
        rewritten, valid = parse_rewrite(raw, s["query"])
        if valid:
            json_valid += 1
        files, methods = bm25_search(rewritten, bm25, chunks, top_k=max_k)
        results.append(
            {
                "sample_id": s["event_id"],
                "retriever": ret_name,
                "retrieved_files": files,
                "retrieved_methods": methods,
                "scores": [],
                "top_k": max_k,
            }
        )
        files_map[s["event_id"]] = files
        raw_map[s["event_id"]] = raw

    return results, files_map, raw_map, json_valid


def main():
    start = time.time()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    max_k = max(K_VALUES)

    print("Loading benchmark samples...")
    with open(SAMPLES_PATH) as f:
        data = json.load(f)
    samples = data["repos"].get(REPO_NAME, [])
    print(f"  {len(samples)} samples for {REPO_NAME}")

    print("Loading BM25 index...")
    chunks = load_chunks(REPO_ID)
    bm25, _ = load_bm25(REPO_ID)
    print(f"  {len(chunks)} chunks indexed")

    all_results = []
    raw_results_map = {}
    json_stats = {}

    print("\n" + "=" * 60)
    print("MODE 1: BM25 (raw query)")
    print("=" * 60)
    for s in samples:
        files, methods = bm25_search(s["query"], bm25, chunks, top_k=max_k)
        all_results.append(
            {
                "sample_id": s["event_id"],
                "retriever": "BM25_raw",
                "retrieved_files": files,
                "retrieved_methods": methods,
                "scores": [],
                "top_k": max_k,
            }
        )
        raw_results_map[s["event_id"]] = files
    print(f"  Done: {len(samples)} queries")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

    print("\n" + "=" * 60)
    print("MODE 2: Base Qwen rewrite (no LoRA)")
    print("=" * 60)
    print(f"  Loading model on {device}...")
    base_model = (
        AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            dtype=torch.float16,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )

    base_results, base_files_map, base_raw_map, base_valid = run_mode_rewrite(
        samples, bm25, chunks, base_model, tokenizer, device, "BM25_base_rewrite"
    )
    all_results.extend(base_results)
    json_stats["base_model"] = {"valid": base_valid, "total": len(samples)}
    print(f"  JSON valid: {base_valid}/{len(samples)}")

    del base_model
    gc.collect()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    print("  Memory freed")

    print("\n" + "=" * 60)
    print("MODE 3: LoRA Qwen rewrite")
    print("=" * 60)
    lora_files_map = {}
    lora_raw_map = {}

    if not Path(LORA_PATH).exists():
        print(f"  ERROR: LoRA adapter not found at {LORA_PATH}")
    else:
        from peft import PeftModel

        print(f"  Loading base + LoRA on {device}...")
        lora_base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            dtype=torch.float16,
            trust_remote_code=True,
        )
        lora_model = PeftModel.from_pretrained(lora_base, LORA_PATH)
        lora_model = lora_model.to(device).eval()

        lora_results, lora_files_map, lora_raw_map, lora_valid = run_mode_rewrite(
            samples, bm25, chunks, lora_model, tokenizer, device, "BM25_lora_rewrite"
        )
        all_results.extend(lora_results)
        json_stats["lora_model"] = {"valid": lora_valid, "total": len(samples)}
        print(f"  JSON valid: {lora_valid}/{len(samples)}")

        del lora_model, lora_base
        gc.collect()
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    print("\n" + "=" * 60)
    print("MODE 4: BM25 (raw query + LoRA rewrite combined)")
    print("=" * 60)
    combined_files_map = {}
    if lora_raw_map:
        for s in samples:
            eid = s["event_id"]
            raw_output = lora_raw_map.get(eid, "")
            # Parse the JSON from LoRA output and extract search terms
            rewritten, _ = parse_rewrite(raw_output, "")
            # Combine original query with rewritten terms
            combined_query = f"{s['query']} {rewritten}"
            files, methods = bm25_search(combined_query, bm25, chunks, top_k=max_k)
            all_results.append(
                {
                    "sample_id": eid,
                    "retriever": "BM25_combined",
                    "retrieved_files": files,
                    "retrieved_methods": methods,
                    "scores": [],
                    "top_k": max_k,
                }
            )
            combined_files_map[eid] = files
        print(f"  Done: {len(samples)} queries")
    else:
        print("  Skipped (no LoRA results)")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    metrics = compute_metrics(samples, all_results)

    header = f"{'Retriever':<25}"
    for k in K_VALUES:
        header += f" {'R@' + str(k):>7} {'MRR@' + str(k):>7}"
    print("\n" + header)
    print("=" * len(header))
    for ret in metrics:
        m = metrics[ret]
        row = f"{ret:<25}"
        for k in K_VALUES:
            row += f" {m.get(f'recall@{k}', 0):>7.3f} {m.get(f'mrr@{k}', 0):>7.3f}"
        print(row)

    ground_truth = {s["event_id"]: set(s["changed_files"]) for s in samples}
    qualitative = []
    print("\nQUALITATIVE EXAMPLES")
    print("=" * 60)
    for s in samples[:10]:
        eid = s["event_id"]
        gt = ground_truth[eid]
        q = {
            "query": s["query"],
            "gt_files": list(gt),
            "base_raw": base_raw_map.get(eid, ""),
            "lora_raw": lora_raw_map.get(eid, ""),
            "raw_hit": any(f in gt for f in raw_results_map.get(eid, [])[:5]),
            "base_hit": any(f in gt for f in base_files_map.get(eid, [])[:5]),
            "lora_hit": any(f in gt for f in lora_files_map.get(eid, [])[:5]),
            "combined_hit": any(f in gt for f in combined_files_map.get(eid, [])[:5]),
            "raw_top5": raw_results_map.get(eid, [])[:5],
            "base_top5": base_files_map.get(eid, [])[:5],
            "lora_top5": lora_files_map.get(eid, [])[:5],
            "combined_top5": combined_files_map.get(eid, [])[:5],
        }
        qualitative.append(q)
        print(f"\nQuery: {s['query'][:80]}")
        print(f"  GT: {list(gt)[:2]}")
        print(f"  BM25 hit@5: {q['raw_hit']}")
        print(f"  Base hit@5: {q['base_hit']}")
        print(f"  LoRA hit@5: {q['lora_hit']}")
        print(f"  Combined hit@5: {q['combined_hit']}")
        if q["lora_raw"]:
            print(f"  LoRA output: {q['lora_raw'][:150]}")

    elapsed = time.time() - start
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "rewriter_experiment_results.json", "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "qualitative_examples": qualitative,
                "json_parse_stats": json_stats,
                "elapsed_seconds": round(elapsed, 1),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(RESULTS_DIR / "rewriter_experiment_report.md", "w") as f:
        f.write("# Query Rewriter LoRA Experiment Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Repository:** {REPO_NAME}\n")
        f.write(f"**Total time:** {elapsed:.1f}s\n\n")
        f.write("## Metrics Comparison\n\n| Retriever |")
        for k in K_VALUES:
            f.write(f" R@{k} | MRR@{k} |")
        f.write("\n|---|")
        for _ in K_VALUES:
            f.write("---|---|")
        f.write("\n")
        for ret in metrics:
            m = metrics[ret]
            f.write(f"| {ret} |")
            for k in K_VALUES:
                f.write(
                    f" {m.get(f'recall@{k}', 0):.3f} | {m.get(f'mrr@{k}', 0):.3f} |"
                )
            f.write("\n")
        f.write(f"\n## JSON Parse Success\n\n")
        for mode, stats in json_stats.items():
            rate = stats["valid"] / stats["total"] * 100 if stats["total"] else 0
            f.write(f"- **{mode}:** {stats['valid']}/{stats['total']} ({rate:.0f}%)\n")
        f.write("\n## Qualitative Examples\n\n")
        for i, q in enumerate(qualitative):
            f.write(f"### Example {i + 1}\n\n")
            f.write(f"**Query:** {q['query']}\n\n")
            f.write(f"**GT:** {', '.join(q['gt_files'][:3])}\n\n")
            if q.get("lora_raw"):
                f.write(f"**LoRA output:**\n```\n{q['lora_raw'][:500]}\n```\n\n")
            if q.get("base_raw"):
                f.write(f"**Base output:**\n```\n{q['base_raw'][:500]}\n```\n\n")
            f.write(
                f"BM25 hit@5: {q['raw_hit']} | Base hit@5: {q['base_hit']} | LoRA hit@5: {q['lora_hit']}\n\n---\n\n"
            )

    print(f"\nReports saved to {RESULTS_DIR}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
