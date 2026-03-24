#!/usr/bin/env python3
"""
Run benchmark comparing BM25, BM25+FileAgg, and LoRA/Qwen reranking.
Pre-initializes the LoRA model before running to avoid MPS segfaults.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

LORA_PATH = str(Path(__file__).parent / "lora_training" / "output" / "scorer_lora" / "final")

# Repos to benchmark (small only, excluding hmis/dotCMS/goodmem)
SMALL_REPOS = [
    "jdereg/java-util",
    "panghy/javaflow",
    "etonai/OpenFields2",
]

# Retrievers that DON'T need LLM (fast)
FAST_RETRIEVERS = ["bm25", "bm25_fileagg"]
# Retrievers that need LLM (base Qwen, no LoRA — LoRA causes OOM)
LLM_RETRIEVERS = ["bm25_qwen_rerank"]


def load_dataset():
    samples_path = Path(__file__).parent / "results" / "benchmark_samples.json"
    with open(samples_path) as f:
        data = json.load(f)
    # Filter to small repos only
    filtered_repos = {k: v for k, v in data["repos"].items() if k in SMALL_REPOS}
    data["repos"] = filtered_repos
    data["total_samples"] = sum(len(s) for s in filtered_repos.values())
    data["total_repos"] = len(filtered_repos)
    return data


def repo_id_from_name(name: str) -> str:
    return name.replace("/", "--")


def run_retriever(retriever_name, dataset, max_k=20):
    """Run a single retriever on all samples, return results."""
    from benchmark.retrievers import get_retriever
    from benchmark.clone_and_index import validate_samples
    from benchmark.config import BenchmarkDataset, BenchmarkSample, RetrievalResult

    results = []
    total = sum(len(s) for s in dataset["repos"].values())
    query_num = 0

    for repo_name, samples_data in dataset["repos"].items():
        repo_id = repo_id_from_name(repo_name)
        retriever = get_retriever(retriever_name, repo_id)

        for sample_data in samples_data:
            query_num += 1
            query = sample_data["query"]
            event_id = sample_data["event_id"]

            if query_num % 10 == 1:
                print(f"  [{query_num}/{total}] {query[:60]}...", flush=True)

            try:
                t0 = time.time()
                hits, elapsed = retriever.retrieve_timed(query, top_k=max_k)

                retrieved_files = []
                retrieved_methods = []
                scores = []
                seen_files = set()
                for chunk, score in hits:
                    if chunk.file_path not in seen_files:
                        retrieved_files.append(chunk.file_path)
                        seen_files.add(chunk.file_path)
                    if chunk.method_name and chunk.method_name not in retrieved_methods:
                        retrieved_methods.append(chunk.method_name)
                    scores.append(score)

                results.append({
                    "sample_id": event_id,
                    "retriever": retriever_name,
                    "retrieved_files": retrieved_files,
                    "retrieved_methods": retrieved_methods,
                    "scores": scores,
                    "top_k": max_k,
                })
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                results.append({
                    "sample_id": event_id,
                    "retriever": retriever_name,
                    "retrieved_files": [],
                    "retrieved_methods": [],
                    "scores": [],
                    "top_k": max_k,
                })

    return results


def compute_metrics(dataset, all_results, k_values=[1, 3, 5, 10, 20]):
    """Compute Recall@K and MRR for each retriever."""
    # Build ground truth: event_id -> set of changed files
    ground_truth = {}
    for repo_name, samples in dataset["repos"].items():
        for s in samples:
            ground_truth[s["event_id"]] = set(s["changed_files"])

    # Group results by retriever
    by_retriever = {}
    for r in all_results:
        by_retriever.setdefault(r["retriever"], []).append(r)

    metrics = {}
    for ret_name, results in by_retriever.items():
        ret_metrics = {}
        for k in k_values:
            hits = 0
            rr_sum = 0.0
            total = 0
            for r in results:
                gt = ground_truth.get(r["sample_id"], set())
                if not gt:
                    continue
                total += 1
                top_k_files = r["retrieved_files"][:k]

                # Recall@K: did we find ANY ground truth file?
                if any(f in gt for f in top_k_files):
                    hits += 1

                # MRR: rank of first correct file
                for rank, f in enumerate(top_k_files, 1):
                    if f in gt:
                        rr_sum += 1.0 / rank
                        break

            recall = hits / total if total > 0 else 0.0
            mrr = rr_sum / total if total > 0 else 0.0
            ret_metrics[f"recall@{k}"] = round(recall, 4)
            ret_metrics[f"mrr@{k}"] = round(mrr, 4)

        ret_metrics["total_samples"] = len(results)
        metrics[ret_name] = ret_metrics

    return metrics


def print_table(metrics, k_values=[1, 3, 5, 10, 20]):
    """Pretty-print metrics table."""
    retrievers = sorted(metrics.keys())

    # Header
    header = f"{'Retriever':<25}"
    for k in k_values:
        header += f" {'R@'+str(k):>7} {'MRR@'+str(k):>7}"
    header += f" {'Samples':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for ret in retrievers:
        m = metrics[ret]
        row = f"{ret:<25}"
        for k in k_values:
            row += f" {m.get(f'recall@{k}', 0):.3f}   {m.get(f'mrr@{k}', 0):.3f}"
            row += " "
        row += f" {m['total_samples']:>7}"
        print(row)

    print("=" * len(header))


def save_results(metrics, all_results, elapsed):
    """Save results to JSON and markdown."""
    results_dir = Path(__file__).parent / "results"

    # JSON
    json_path = results_dir / "lora_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "elapsed_seconds": elapsed,
            "repos": list(SMALL_REPOS),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Markdown
    md_path = results_dir / "lora_benchmark_results.md"
    k_values = [1, 3, 5, 10, 20]
    with open(md_path, "w") as f:
        f.write("# LoRA Benchmark Results\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Repos:** {', '.join(SMALL_REPOS)}\n")
        f.write(f"**Total time:** {elapsed:.1f}s\n\n")

        f.write("## Recall@K (File-Level)\n\n")
        f.write("| Retriever |")
        for k in k_values:
            f.write(f" R@{k} | MRR@{k} |")
        f.write("\n")
        f.write("|---|")
        for _ in k_values:
            f.write("---|---|")
        f.write("\n")

        for ret in sorted(metrics.keys()):
            m = metrics[ret]
            f.write(f"| {ret} |")
            for k in k_values:
                f.write(f" {m.get(f'recall@{k}', 0):.3f} | {m.get(f'mrr@{k}', 0):.3f} |")
            f.write("\n")

    print(f"Saved Markdown: {md_path}")


def main():
    start = time.time()
    dataset = load_dataset()
    total = dataset["total_samples"]
    print(f"Dataset: {total} samples across {dataset['total_repos']} repos")
    print(f"Repos: {list(dataset['repos'].keys())}")

    all_results = []
    cached_fast_path = Path(__file__).parent / "results" / "lora_benchmark_fast_results.json"

    # Phase 1: Fast retrievers (no LLM)
    if cached_fast_path.exists():
        print(f"\nLoading cached fast results from {cached_fast_path}", flush=True)
        with open(cached_fast_path) as f:
            all_results = json.load(f)
        print(f"  Loaded {len(all_results)} cached results")
    else:
        print("\n" + "=" * 60)
        print("PHASE 1: Fast retrievers (BM25, FileAgg)")
        print("=" * 60)
        for ret_name in FAST_RETRIEVERS:
            print(f"\n--- {ret_name} ---", flush=True)
            t0 = time.time()
            results = run_retriever(ret_name, dataset)
            print(f"  Done in {time.time()-t0:.1f}s ({len(results)} results)")
            all_results.extend(results)

        # Cache fast results
        with open(cached_fast_path, "w") as f:
            json.dump(all_results, f)
        print(f"\nCached fast results to {cached_fast_path}")

    # Phase 2: LLM retrievers — init LoRA model
    if LLM_RETRIEVERS:
        print("\n" + "=" * 60)
        print("PHASE 2: LLM retrievers (LoRA reranking)")
        print("=" * 60)
        print("\nPre-loading Qwen model (base, no LoRA)...", flush=True)
        from app.ml.model_manager import get_model_manager
        manager = get_model_manager()  # No LoRA — base model, float16
        _ = manager.qwen  # Force load
        print(f"Qwen model loaded on {manager.device}", flush=True)

        for ret_name in LLM_RETRIEVERS:
            print(f"\n--- {ret_name} ---", flush=True)
            t0 = time.time()
            results = run_retriever(ret_name, dataset)
            print(f"  Done in {time.time()-t0:.1f}s ({len(results)} results)")
            all_results.extend(results)

    # Compute and display
    elapsed = time.time() - start
    metrics = compute_metrics(dataset, all_results)
    print_table(metrics)
    save_results(metrics, all_results, elapsed)
    print(f"\nTotal benchmark time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
