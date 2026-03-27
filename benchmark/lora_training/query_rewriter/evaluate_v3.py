import gc
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
import torch
from llm_client import generate
from llm_client import unload_model as unload_generator
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.app.indexer.bm25_builder import tokenize
from backend.app.indexer.store import load_bm25, load_chunks

REPO_ID = "jdereg--java-util"
REPO_NAME = "jdereg/java-util"
BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
LORA_V3_PATH = str(
    ROOT / "benchmark" / "lora_training" / "output" / "rewriter_lora_v3" / "final"
)
SAMPLES_PATH = ROOT / "benchmark" / "results" / "benchmark_samples.json"
RESULTS_DIR = ROOT / "benchmark" / "results"
DATA_DIR = Path(__file__).parent / "data"
K_VALUES = [1, 5, 10, 20]

REWRITE_PROMPT = (
    "Rewrite this search query into structured retrieval hints "
    "for a Java codebase.\n\nQuery: {query}\n\nOutput JSON:"
)

DECOMPOSE_SYSTEM = (
    "Extract 1-3 short code search intents from this commit description. "
    "Each intent should be a concise search query (3-10 words) targeting "
    "one specific code concern. Output ONLY a JSON array of strings."
)

ABLATION_FIELDS = {
    "full": [
        "search_queries",
        "keywords",
        "method_hints",
        "project_terms",
        "api_hints",
    ],
    "queries_only": ["search_queries"],
    "hints_only": ["method_hints", "project_terms", "api_hints"],
    "no_keywords": ["search_queries", "method_hints", "project_terms", "api_hints"],
    "no_methods": ["search_queries", "keywords", "project_terms", "api_hints"],
    "no_project": ["search_queries", "keywords", "method_hints", "api_hints"],
    "no_api": ["search_queries", "keywords", "method_hints", "project_terms"],
}


def generate_rewrite(model, tokenizer, prompt, device, max_new_tokens=256):
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


def parse_rewrite(raw, original_query, fields=None):
    if fields is None:
        fields = ABLATION_FIELDS["full"]

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
        for key in fields:
            for v in data.get(key, []):
                if isinstance(v, str):
                    parts.append(v)
        if parts:
            return " ".join(parts), True
    except json.JSONDecodeError, TypeError:
        pass
    return f"{original_query} {raw.strip()[:300]}", False


def bm25_search(query_str, bm25, chunks, top_k=20):
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


def multi_intent_search(intents, bm25, chunks, top_k=20):
    file_scores = defaultdict(float)
    for intent in intents:
        files, _ = bm25_search(intent, bm25, chunks, top_k=top_k)
        for rank, f in enumerate(files):
            file_scores[f] += 1.0 / (rank + 60)

    ranked = sorted(file_scores.items(), key=lambda x: -x[1])
    return [f for f, _ in ranked[:top_k]], []


def decompose_query_llm(query, llm_generate_fn):
    raw = llm_generate_fn(query)
    text = raw.strip()
    if "```" in text:
        for part in text.split("```")[1:]:
            c = part.strip()
            if c.startswith("json"):
                c = c[4:].strip()
            if c.startswith("["):
                text = c
                break
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
        if isinstance(data, list) and all(isinstance(s, str) for s in data):
            filtered = [s.strip() for s in data if 3 <= len(s.strip().split()) <= 15]
            if filtered:
                return filtered[:3]
    except json.JSONDecodeError, TypeError:
        pass
    return decompose_query_rules(query)


def decompose_query_rules(query):
    parts = re.split(r"(?:\. |\n| - |\n- )", query)

    intents = []
    for part in parts:
        part = part.strip()
        if not part or len(part) < 10:
            continue
        part = re.sub(
            r"^(Fix|Add|Update|Improve|Enhance|Optimize|Refactor|Security:)\s+",
            "",
            part,
        )
        part = re.sub(r"^(All \d+.* tests pass.*|Maintains backward.*)", "", part)
        part = part.strip()
        if not part or len(part) < 10:
            continue
        if len(part) > 80:
            part = part[:80].rsplit(" ", 1)[0]
        intents.append(part)
        if len(intents) >= 3:
            break

    if not intents:
        return [query[:80].rsplit(" ", 1)[0]]

    return intents


def run_decomposition(samples, decompose_fn, cache_path):
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached decompositions")

    decomposed = {}
    new_count = 0
    for s in samples:
        eid = s["event_id"]
        if eid in cache:
            decomposed[eid] = cache[eid]
        else:
            intents = decompose_fn(s["query"])
            decomposed[eid] = intents
            cache[eid] = intents
            new_count += 1

    if new_count > 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"Decomposed {new_count} new queries, saved to {cache_path}")

    n_intents = [len(v) for v in decomposed.values()]
    print(f"Avg intents per query: {np.mean(n_intents):.1f}")
    print(f"Total intents: {sum(n_intents)}")

    return decomposed


def compute_collapse_diagnostics(raw_outputs):
    total = len(raw_outputs)
    if total == 0:
        return {}

    unique = set(raw_outputs.values())

    outputs = list(raw_outputs.values())
    sims = []
    limit = min(len(outputs), 50)
    for i in range(limit):
        for j in range(i + 1, limit):
            t_i = set(outputs[i].lower().split())
            t_j = set(outputs[j].lower().split())
            union = t_i | t_j
            if union:
                sims.append(len(t_i & t_j) / len(union))

    counter = Counter(raw_outputs.values())
    most_common_count = counter.most_common(1)[0][1] if counter else 0

    return {
        "total_outputs": total,
        "unique_outputs": len(unique),
        "uniqueness_ratio": round(len(unique) / total, 3),
        "most_common_count": most_common_count,
        "most_common_ratio": round(most_common_count / total, 3),
        "mean_pairwise_jaccard": round(float(np.mean(sims)), 3) if sims else 0,
        "median_pairwise_jaccard": round(float(np.median(sims)), 3) if sims else 0,
        "collapsed": len(unique) / total < 0.5,
    }


def compute_metrics(samples, all_results):
    ground_truth = {s["event_id"]: set(s["changed_files"]) for s in samples}
    by_retriever = defaultdict(list)
    for r in all_results:
        by_retriever[r["retriever"]].append(r)

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


def run_mode_bm25_raw(samples, bm25, chunks, max_k):
    results = []
    for s in samples:
        files, methods = bm25_search(s["query"], bm25, chunks, top_k=max_k)
        results.append(
            {
                "sample_id": s["event_id"],
                "retriever": "A_BM25_raw",
                "retrieved_files": files,
                "retrieved_methods": methods,
                "scores": [],
                "top_k": max_k,
            }
        )
    return results


def run_mode_decomposed(samples, decomposed, bm25, chunks, max_k):
    results = []
    for s in samples:
        intents = decomposed.get(s["event_id"], [s["query"]])
        files, methods = multi_intent_search(intents, bm25, chunks, top_k=max_k)
        results.append(
            {
                "sample_id": s["event_id"],
                "retriever": "B_BM25_decomposed",
                "retrieved_files": files,
                "retrieved_methods": methods,
                "scores": [],
                "top_k": max_k,
            }
        )
    return results


def run_mode_rewrite(
    samples,
    decomposed,
    bm25,
    chunks,
    model,
    tokenizer,
    device,
    ret_name,
    combined=False,
    use_decompose=True,
    fields=None,
    max_k=20,
):
    results = []
    raw_outputs = {}
    json_valid = 0

    for i, s in enumerate(samples):
        if (i + 1) % 10 == 1:
            print(f"[{i + 1}/{len(samples)}] {s['query'][:60]}...", flush=True)

        if use_decompose:
            intents = decomposed.get(s["event_id"], [s["query"]])
        else:
            intents = [s["query"]]

        all_files_scores = defaultdict(float)

        for intent in intents:
            prompt = REWRITE_PROMPT.format(query=intent)
            raw = generate_rewrite(model, tokenizer, prompt, device)
            rewritten, valid = parse_rewrite(raw, intent, fields=fields)

            if valid:
                json_valid += 1

            if combined:
                search_query = f"{intent} {rewritten}"
            else:
                search_query = rewritten

            files, _ = bm25_search(search_query, bm25, chunks, top_k=max_k)
            for rank, f in enumerate(files):
                all_files_scores[f] += 1.0 / (rank + 60)

            if intent == intents[0]:
                raw_outputs[s["event_id"]] = raw

        ranked = sorted(all_files_scores.items(), key=lambda x: -x[1])
        final_files = [f for f, _ in ranked[:max_k]]

        results.append(
            {
                "sample_id": s["event_id"],
                "retriever": ret_name,
                "retrieved_files": final_files,
                "retrieved_methods": [],
                "scores": [],
                "top_k": max_k,
            }
        )

    return results, raw_outputs, json_valid


def main():
    start = time.time()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    max_k = max(K_VALUES)

    print("Loading benchmark samples...")
    with open(SAMPLES_PATH) as f:
        data = json.load(f)
    samples = data["repos"].get(REPO_NAME, [])
    print(f"{len(samples)} samples for {REPO_NAME}")

    print("Loading BM25 index...")
    chunks = load_chunks(REPO_ID)
    bm25, _ = load_bm25(REPO_ID)
    print(f"{len(chunks)} chunks indexed")

    all_results = []
    diagnostics = {}
    json_stats = {}

    print("---QUERY DECOMPOSITION---")

    decompose_cache = DATA_DIR / "decomposed_eval_queries_v3.json"

    try:

        def decompose_fn(query):
            return decompose_query_llm(
                query,
                lambda q: generate(
                    q, system=DECOMPOSE_SYSTEM, max_new_tokens=200, temperature=0.3
                ),
            )

        decomposed = run_decomposition(samples, decompose_fn, decompose_cache)
        unload_generator()
    except Exception as e:
        print(f"LLM decomposition failed ({e}), using rule-based fallback")
        decomposed = run_decomposition(samples, decompose_query_rules, decompose_cache)

    print("\n  Decomposition examples:")
    for s in samples[:3]:
        eid = s["event_id"]
        print(f"Query: {s['query'][:80]}...")
        print(f"Intents: {decomposed.get(eid, [])}")
        print()

    print("=" * 60)
    print("MODE A: BM25 (raw query)")
    print("=" * 60)
    results_a = run_mode_bm25_raw(samples, bm25, chunks, max_k)
    all_results.extend(results_a)
    print(f"Done: {len(samples)} queries")

    print("\n" + "=" * 60)
    print("MODE B: BM25 (decomposed intents)")
    print("=" * 60)
    results_b = run_mode_decomposed(samples, decomposed, bm25, chunks, max_k)
    all_results.extend(results_b)
    print(f"Done: {len(samples)} queries")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

    print("\n" + "=" * 60)
    print("MODE E: Base Qwen + decomposed (combined)")
    print("=" * 60)
    print(f"Loading model on {device}...")
    base_model = (
        AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            dtype=torch.float16,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )

    results_e, raw_e, valid_e = run_mode_rewrite(
        samples,
        decomposed,
        bm25,
        chunks,
        base_model,
        tokenizer,
        device,
        ret_name="E_base_combined",
        combined=True,
        use_decompose=True,
        max_k=max_k,
    )
    all_results.extend(results_e)
    json_stats["base_model"] = {"valid": valid_e, "total": len(samples)}
    diagnostics["E_base"] = compute_collapse_diagnostics(raw_e)
    print(f"JSON valid: {valid_e}/{len(samples)}")

    del base_model
    gc.collect()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    if not Path(LORA_V3_PATH).exists():
        print(f"\n  ERROR: LoRA v3 adapter not found at {LORA_V3_PATH}")
        print("Skipping modes C, D, F")
    else:
        print(f"\n  Loading base + LoRA v3 on {device}...")
        lora_base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            dtype=torch.float16,
            trust_remote_code=True,
        )
        lora_model = PeftModel.from_pretrained(lora_base, LORA_V3_PATH)
        lora_model = lora_model.to(device).eval()

        print("\n" + "=" * 60)
        print("MODE C: LoRA standalone (decomposed)")
        print("=" * 60)
        results_c, raw_c, valid_c = run_mode_rewrite(
            samples,
            decomposed,
            bm25,
            chunks,
            lora_model,
            tokenizer,
            device,
            ret_name="C_lora_standalone",
            combined=False,
            use_decompose=True,
            max_k=max_k,
        )
        all_results.extend(results_c)
        json_stats["lora_standalone"] = {"valid": valid_c, "total": len(samples)}
        diagnostics["C_lora_standalone"] = compute_collapse_diagnostics(raw_c)
        print(f"JSON valid: {valid_c}/{len(samples)}")

        print("\n" + "=" * 60)
        print("MODE D: LoRA combined (decomposed)")
        print("=" * 60)
        results_d, raw_d, valid_d = run_mode_rewrite(
            samples,
            decomposed,
            bm25,
            chunks,
            lora_model,
            tokenizer,
            device,
            ret_name="D_lora_combined",
            combined=True,
            use_decompose=True,
            max_k=max_k,
        )
        all_results.extend(results_d)
        json_stats["lora_combined"] = {"valid": valid_d, "total": len(samples)}
        diagnostics["D_lora_combined"] = compute_collapse_diagnostics(raw_d)
        print(f"JSON valid: {valid_d}/{len(samples)}")

        print("\n" + "=" * 60)
        print("MODE F: LoRA combined (raw, no decompose)")
        print("=" * 60)
        results_f, raw_f, valid_f = run_mode_rewrite(
            samples,
            decomposed,
            bm25,
            chunks,
            lora_model,
            tokenizer,
            device,
            ret_name="F_combined_raw",
            combined=True,
            use_decompose=False,
            max_k=max_k,
        )
        all_results.extend(results_f)
        json_stats["lora_combined_raw"] = {"valid": valid_f, "total": len(samples)}
        diagnostics["F_combined_raw"] = compute_collapse_diagnostics(raw_f)
        print(f"JSON valid: {valid_f}/{len(samples)}")

        print("\n" + "=" * 60)
        print("FIELD ABLATION (on Mode D pattern)")
        print("=" * 60)
        ablation_results = {}
        for abl_name, abl_fields in ABLATION_FIELDS.items():
            if abl_name == "full":
                continue
            print(f"\n  Ablation: {abl_name} (fields: {abl_fields})")
            abl_r, _, abl_valid = run_mode_rewrite(
                samples,
                decomposed,
                bm25,
                chunks,
                lora_model,
                tokenizer,
                device,
                ret_name=f"D_abl_{abl_name}",
                combined=True,
                use_decompose=True,
                fields=abl_fields,
                max_k=max_k,
            )
            all_results.extend(abl_r)
            ablation_results[abl_name] = {"valid": abl_valid, "total": len(samples)}

        del lora_model, lora_base
        gc.collect()
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    metrics = compute_metrics(samples, all_results)

    header = f"{'Retriever':<30}"
    for k in K_VALUES:
        header += f"{'R@' + str(k):>7} {'MRR@' + str(k):>7}"
    print("\n" + header)
    print("=" * len(header))

    main_modes = [k for k in metrics if not k.startswith("D_abl_")]
    abl_modes = [k for k in metrics if k.startswith("D_abl_")]

    for ret in main_modes:
        m = metrics[ret]
        row = f"{ret:<30}"
        for k in K_VALUES:
            row += f"{m.get(f'recall@{k}', 0):>7.3f} {m.get(f'mrr@{k}', 0):>7.3f}"
        print(row)

    if abl_modes:
        print("\n--- Field Ablation ---")
        for ret in sorted(abl_modes):
            m = metrics[ret]
            row = f"{ret:<30}"
            for k in K_VALUES:
                row += f"{m.get(f'recall@{k}', 0):>7.3f} {m.get(f'mrr@{k}', 0):>7.3f}"
            print(row)

    print("\n" + "=" * 60)
    print("COLLAPSE DIAGNOSTICS")
    print("=" * 60)
    for mode, diag in diagnostics.items():
        print(f"\n  {mode}:")
        for key, val in diag.items():
            print(f"{key}: {val}")

    elapsed = time.time() - start
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report_data = {
        "metrics": metrics,
        "json_parse_stats": json_stats,
        "collapse_diagnostics": diagnostics,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lora_path": LORA_V3_PATH,
        "base_model": BASE_MODEL_NAME,
        "repo": REPO_NAME,
        "num_samples": len(samples),
    }

    with open(RESULTS_DIR / "rewriter_v3_experiment_results.json", "w") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    with open(RESULTS_DIR / "rewriter_v3_experiment_report.md", "w") as f:
        f.write("# V3 Query Rewriter LoRA Experiment Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Repository:** {REPO_NAME}\n")
        f.write(f"**Total time:** {elapsed:.1f}s\n\n")

        f.write("## Main Modes\n\n")
        f.write("| Mode |")
        for k in K_VALUES:
            f.write(f"R@{k} | MRR@{k} |")
        f.write("\n|---|")
        for _ in K_VALUES:
            f.write("---|---|")
        f.write("\n")
        for ret in main_modes:
            m = metrics[ret]
            f.write(f"| {ret} |")
            for k in K_VALUES:
                f.write(f"{m.get(f'recall@{k}', 0):.3f} | {m.get(f'mrr@{k}', 0):.3f} |")
            f.write("\n")

        if abl_modes:
            f.write("\n## Field Ablation (Mode D pattern)\n\n")
            f.write("| Ablation |")
            for k in K_VALUES:
                f.write(f"R@{k} | MRR@{k} |")
            f.write("\n|---|")
            for _ in K_VALUES:
                f.write("---|---|")
            f.write("\n")
            for ret in sorted(abl_modes):
                m = metrics[ret]
                label = ret.replace("D_abl_", "")
                f.write(f"| {label} |")
                for k in K_VALUES:
                    f.write(
                        f"{m.get(f'recall@{k}', 0):.3f} | {m.get(f'mrr@{k}', 0):.3f} |"
                    )
                f.write("\n")

        f.write("\n## JSON Parse Success\n\n")
        for mode, stats in json_stats.items():
            rate = stats["valid"] / stats["total"] * 100 if stats["total"] else 0
            f.write(f"- **{mode}:** {stats['valid']}/{stats['total']} ({rate:.0f}%)\n")

        f.write("\n## Collapse Diagnostics\n\n")
        for mode, diag in diagnostics.items():
            f.write(f"### {mode}\n")
            for key, val in diag.items():
                f.write(f"- {key}: {val}\n")
            f.write("\n")

    print(f"Reports saved to {RESULTS_DIR}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
