import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from benchmark.config import (
    DEFAULT_MAX_REPOS,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_TOP_K_VALUES,
    EVAL_RESULTS_PATH,
    RAW_RESULTS_PATH,
    SAMPLES_PATH,
    BenchmarkDataset,
    BenchmarkSample,
    EvalResults,
    RetrievalResult,
)
from benchmark.clone_and_index import (
    clone_and_index_all,
    repo_id_from_name,
    validate_samples,
)
from benchmark.evaluator import evaluate
from benchmark.extract_samples import extract_samples
from benchmark.retrievers import RETRIEVER_REGISTRY, get_retriever
from benchmark.visualize import plot_results, print_summary_table, save_results_markdown


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieval Benchmark for Repo-Searcher"
    )
    parser.add_argument(
        "--jsonl",
        default="/Users/sckwoky/Downloads/claude-20250813.jsonl",
        help="Path to the JSONL dataset",
    )
    parser.add_argument("--max-repos", type=int, default=DEFAULT_MAX_REPOS)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument(
        "--top-k", default="1,3,5,10,20", help="Comma-separated K values"
    )
    parser.add_argument(
        "--retrievers",
        default="bm25,vector,rrf",
        help=f"Comma-separated retrievers. Available: {','.join(RETRIEVER_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Reuse cached benchmark_samples.json",
    )
    parser.add_argument(
        "--skip-index", action="store_true", help="Reuse cached indexes"
    )
    parser.add_argument(
        "--skip-retrieve", action="store_true", help="Reuse cached raw_results.json"
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=20,
        help="Max top-K to retrieve (should be >= max of --top-k)",
    )
    parser.add_argument(
        "--exclude-repos",
        default="",
        help="Comma-separated repo names to exclude (e.g. 'hmislk/hmis,dotCMS/core')",
    )
    return parser.parse_args()


def stage_extract(args) -> BenchmarkDataset:
    """Stage 1: Extract benchmark samples from JSONL."""
    print("\n" + "=" * 60)
    print("STAGE 1: Extracting benchmark samples")
    print("=" * 60)

    if args.skip_extract and SAMPLES_PATH.exists():
        print(f"Loading cached samples from {SAMPLES_PATH}")
        with open(SAMPLES_PATH) as f:
            return BenchmarkDataset(**json.load(f))

    return extract_samples(
        jsonl_path=args.jsonl,
        max_repos=args.max_repos,
        min_total_samples=args.min_samples,
    )


def stage_index(dataset: BenchmarkDataset, skip: bool) -> BenchmarkDataset:
    """Stage 2: Clone repos and build indexes."""
    print("\n" + "=" * 60)
    print("STAGE 2: Cloning and indexing repositories")
    print("=" * 60)

    if not skip:
        clone_and_index_all(dataset)

    print("\nValidating samples against indexes...")
    filtered, stats = validate_samples(dataset)
    print(
        f"\nAfter validation: {filtered.total_samples} samples across {filtered.total_repos} repos"
    )
    return filtered


def stage_retrieve(
    dataset: BenchmarkDataset,
    retriever_names: list[str],
    max_k: int,
    skip: bool,
) -> list[RetrievalResult]:
    """Stage 3: Run all retrievers on all samples."""
    print("\n" + "=" * 60)
    print("STAGE 3: Running retrievers")
    print("=" * 60)

    if skip and RAW_RESULTS_PATH.exists():
        print(f"Loading cached results from {RAW_RESULTS_PATH}")
        with open(RAW_RESULTS_PATH) as f:
            data = json.load(f)
        return [RetrievalResult(**r) for r in data]

    all_results: list[RetrievalResult] = []
    total_queries = sum(len(samples) for samples in dataset.repos.values())

    for ret_name in retriever_names:
        print(f"\n--- Retriever: {ret_name} ---")
        query_num = 0

        for repo_name, samples in dataset.repos.items():
            repo_id = repo_id_from_name(repo_name)
            retriever = get_retriever(ret_name, repo_id)

            for sample in samples:
                query_num += 1
                if query_num % 10 == 1 or query_num == total_queries:
                    print(f"  [{query_num}/{total_queries}] {sample.query[:60]}...")

                try:
                    results, elapsed = retriever.retrieve_timed(
                        sample.query, top_k=max_k
                    )

                    retrieved_files = []
                    retrieved_methods = []
                    scores = []
                    seen_files = set()

                    for chunk, score in results:
                        if chunk.file_path not in seen_files:
                            retrieved_files.append(chunk.file_path)
                            seen_files.add(chunk.file_path)
                        if (
                            chunk.method_name
                            and chunk.method_name not in retrieved_methods
                        ):
                            retrieved_methods.append(chunk.method_name)
                        scores.append(score)

                    all_results.append(
                        RetrievalResult(
                            sample_id=sample.event_id,
                            retriever=ret_name,
                            retrieved_files=retrieved_files,
                            retrieved_methods=retrieved_methods,
                            scores=scores,
                            top_k=max_k,
                        )
                    )
                except Exception as e:
                    print(f"    ERROR: {e}")
                    all_results.append(
                        RetrievalResult(
                            sample_id=sample.event_id,
                            retriever=ret_name,
                            retrieved_files=[],
                            retrieved_methods=[],
                            scores=[],
                            top_k=max_k,
                        )
                    )

    RAW_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_RESULTS_PATH, "w") as f:
        json.dump([r.model_dump() for r in all_results], f, indent=2)
    print(f"\nSaved raw results to {RAW_RESULTS_PATH}")

    return all_results


def stage_evaluate(
    dataset: BenchmarkDataset,
    results: list[RetrievalResult],
    k_values: list[int],
) -> EvalResults:
    """Stage 4: Compute metrics."""
    print("\n" + "=" * 60)
    print("STAGE 4: Computing metrics")
    print("=" * 60)

    all_samples = []
    for samples in dataset.repos.values():
        all_samples.extend(samples)

    eval_results = evaluate(all_samples, results, k_values)

    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(eval_results.model_dump(), f, indent=2)
    print(f"Saved eval results to {EVAL_RESULTS_PATH}")

    return eval_results


def stage_visualize(eval_results: EvalResults, k_values: list[int]):
    print("\n" + "=" * 60)
    print("STAGE 5: Visualization")
    print("=" * 60)

    print_summary_table(eval_results, k_values)
    plot_results(eval_results)
    save_results_markdown(eval_results, k_values)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    k_values = [int(k) for k in args.top_k.split(",")]
    retriever_names = [r.strip() for r in args.retrievers.split(",")]
    max_k = max(max(k_values), args.max_k)

    for name in retriever_names:
        if name not in RETRIEVER_REGISTRY:
            print(
                f"ERROR: Unknown retriever '{name}'. Available: {list(RETRIEVER_REGISTRY.keys())}"
            )
            sys.exit(1)

    start_time = time.time()
    dataset = stage_extract(args)

    if args.exclude_repos:
        exclude = {r.strip() for r in args.exclude_repos.split(",")}
        dataset.repos = {k: v for k, v in dataset.repos.items() if k not in exclude}
        dataset.total_samples = sum(len(s) for s in dataset.repos.values())
        dataset.total_repos = len(dataset.repos)
        print(
            f"After excluding {exclude}: {dataset.total_samples} samples across {dataset.total_repos} repos"
        )

    dataset = stage_index(dataset, args.skip_index)
    results = stage_retrieve(dataset, retriever_names, max_k, args.skip_retrieve)
    eval_results = stage_evaluate(dataset, results, k_values)
    stage_visualize(eval_results, k_values)

    elapsed = time.time() - start_time
    print(f"\nTotal benchmark time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
