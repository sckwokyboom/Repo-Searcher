import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from benchmark.clone_and_index import repo_id_from_name
from benchmark.config import (
    BenchmarkDataset,
    BenchmarkSample,
    RetrievalResult,
    SAMPLES_PATH,
)
from benchmark.evaluator import compute_sample_metrics, aggregate_metrics
from benchmark.retrievers import (
    BM25FileAgg,
    BM25GraphExpansion,
    BM25PrioritizedExpansion,
    _build_file_chunk_index,
    _build_chunk_id_index,
    _get_call_graph,
)
from app.config import settings
from app.indexer.store import load_chunks

EXPERIMENT_DIR = Path(__file__).parent / "results" / "graph_expansion"
N_SAMPLES = 75
SEED = 42
TOP_K = 10
K_VALUES = [1, 3, 5, 10]


def load_dataset() -> BenchmarkDataset:
    with open(SAMPLES_PATH) as f:
        return BenchmarkDataset(**json.load(f))


def _indexed_repos(dataset: BenchmarkDataset) -> dict[str, list[BenchmarkSample]]:
    """Filter dataset to repos that have indexes on disk."""
    filtered = {}
    for repo, samples in dataset.repos.items():
        repo_id = repo_id_from_name(repo)
        index_dir = settings.indexes_dir / repo_id
        if (index_dir / "chunks.json").exists():
            filtered[repo] = samples
        else:
            print(f"Skipping {repo} (no index)")
    return filtered


def sample_queries(
    dataset: BenchmarkDataset,
    n: int = N_SAMPLES,
    seed: int = SEED,
) -> list[tuple[str, BenchmarkSample]]:
    """Stratified sampling proportional to repo size, only from indexed repos."""
    rng = random.Random(seed)
    indexed = _indexed_repos(dataset)
    all_pairs = []
    for repo, samples in indexed.items():
        for s in samples:
            all_pairs.append((repo, s))

    total = len(all_pairs)
    selected: list[tuple[str, BenchmarkSample]] = []
    for repo, samples in indexed.items():
        repo_n = max(1, round(n * len(samples) / total))
        repo_pairs = [(repo, s) for s in samples]
        selected.extend(rng.sample(repo_pairs, min(repo_n, len(repo_pairs))))

    if len(selected) > n:
        selected = rng.sample(selected, n)
    rng.shuffle(selected)
    return selected


def run_experiment():
    print("=" * 60)
    print("Graph Expansion Experiment")
    print("=" * 60)

    # Load data
    print("\nLoading dataset...")
    dataset = load_dataset()
    print(f"Total: {dataset.total_samples} samples across {dataset.total_repos} repos")

    queries = sample_queries(dataset)
    print(
        f"Sampled {len(queries)} queries across {len(set(r for r, _ in queries))} repos"
    )

    # Run retrievers
    modes = {
        "A_baseline": lambda rid: BM25FileAgg(rid),
        "B_graph_expand": lambda rid: BM25GraphExpansion(rid),
        "C_prior_expand": lambda rid: BM25PrioritizedExpansion(rid),
    }

    all_results: dict[str, list[RetrievalResult]] = {m: [] for m in modes}
    detailed_logs: list[dict] = []

    # Cache retrievers per repo to share loaded indexes
    retriever_cache: dict[str, dict[str, object]] = {}

    print(f"\nRunning {len(queries)} queries x {len(modes)} modes...")
    start = time.time()

    for qi, (repo_name, sample) in enumerate(queries):
        repo_id = repo_id_from_name(repo_name)

        if qi % 10 == 0:
            print(f"[{qi + 1}/{len(queries)}] {repo_name}: {sample.query[:60]}...")

        # Init retrievers for this repo (share indexes)
        if repo_id not in retriever_cache:
            base = BM25FileAgg(repo_id)
            # Force-load chunks and bm25 once
            _ = base.chunks
            _ = base.bm25
            retriever_cache[repo_id] = {
                "_chunks": base._chunks,
                "_bm25": base._bm25,
                "_corpus": base._corpus,
            }

        cached = retriever_cache[repo_id]
        gt_files = set(sample.changed_files)

        log_entry = {
            "sample_id": sample.event_id,
            "repo": repo_name,
            "query": sample.query,
            "ground_truth_files": sample.changed_files,
        }

        for mode_name, factory in modes.items():
            ret = factory(repo_id)
            ret._chunks = cached["_chunks"]
            ret._bm25 = cached["_bm25"]
            ret._corpus = cached["_corpus"]

            try:
                results = ret.retrieve(sample.query, top_k=TOP_K)
            except Exception as e:
                print(f"ERROR [{mode_name}]: {e}")
                results = []

            # Extract file-level results
            retrieved_files = []
            scores = []
            seen = set()
            for chunk, score in results:
                if chunk.file_path not in seen:
                    retrieved_files.append(chunk.file_path)
                    scores.append(score)
                    seen.add(chunk.file_path)

            all_results[mode_name].append(
                RetrievalResult(
                    sample_id=sample.event_id,
                    retriever=mode_name,
                    retrieved_files=retrieved_files,
                    retrieved_methods=[
                        c.method_name for c, _ in results if c.method_name
                    ],
                    scores=scores,
                    top_k=TOP_K,
                )
            )

            log_entry[f"{mode_name}_files"] = retrieved_files
            log_entry[f"{mode_name}_scores"] = scores
            log_entry[f"{mode_name}_hit@10"] = bool(
                gt_files & set(retrieved_files[:10])
            )
            log_entry[f"{mode_name}_hit@5"] = bool(gt_files & set(retrieved_files[:5]))

        # Compute graph expansion details for the log
        baseline_files_set = set(log_entry.get("A_baseline_files", []))
        b_files_set = set(log_entry.get("B_graph_expand_files", []))
        c_files_set = set(log_entry.get("C_prior_expand_files", []))

        log_entry["B_new_from_graph"] = sorted(b_files_set - baseline_files_set)
        log_entry["C_new_from_graph"] = sorted(c_files_set - baseline_files_set)
        log_entry["B_gained_gt"] = sorted(
            (gt_files & b_files_set) - (gt_files & baseline_files_set)
        )
        log_entry["B_lost_gt"] = sorted(
            (gt_files & baseline_files_set) - (gt_files & b_files_set)
        )
        log_entry["C_gained_gt"] = sorted(
            (gt_files & c_files_set) - (gt_files & baseline_files_set)
        )
        log_entry["C_lost_gt"] = sorted(
            (gt_files & baseline_files_set) - (gt_files & c_files_set)
        )

        # Graph stats
        try:
            graph = _get_call_graph(repo_id)
            chunks = cached["_chunks"]
            fci = _build_file_chunk_index(chunks)
            seed_fps = log_entry.get("A_baseline_files", [])[:5]
            total_neighbors = 0
            for sfp in seed_fps:
                for ci in fci.get(sfp, []):
                    cid = chunks[ci].chunk_id
                    if cid in graph:
                        total_neighbors += graph.degree(cid)
            log_entry["seed_files"] = seed_fps
            log_entry["total_seed_graph_degree"] = total_neighbors
        except Exception:
            log_entry["seed_files"] = []
            log_entry["total_seed_graph_degree"] = 0

        detailed_logs.append(log_entry)

    elapsed = time.time() - start
    print(f"\nRetrieval done in {elapsed:.1f}s")

    # Evaluate
    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)

    all_sample_metrics = []
    samples_list = [s for _, s in queries]

    for mode_name, results in all_results.items():
        for sample, result in zip(samples_list, results):
            m = compute_sample_metrics(sample, result, K_VALUES)
            all_sample_metrics.append(m)

    aggregated = aggregate_metrics(all_sample_metrics, K_VALUES)

    # Print summary table
    print(f"\n{'Mode':<25} {'Hit@5':>8} {'Hit@10':>8} {'Recall@10':>10} {'MRR':>8}")
    print("-" * 65)
    for agg in aggregated:
        print(
            f"{agg.retriever:<25} "
            f"{agg.hit_at_k.get(5, 0):.4f}  "
            f"{agg.hit_at_k.get(10, 0):.4f}  "
            f"{agg.recall_at_k.get(10, 0):.4f}    "
            f"{agg.mrr:.4f}"
        )

    # Detailed per-k table
    print(f"\n{'Mode':<25}", end="")
    for k in K_VALUES:
        print(f"{'Hit@' + str(k):>8} {'Rec@' + str(k):>8}", end="")
    print(f"{'MRR':>8}")
    print("-" * (25 + len(K_VALUES) * 18 + 8))
    for agg in aggregated:
        print(f"{agg.retriever:<25}", end="")
        for k in K_VALUES:
            print(
                f"{agg.hit_at_k.get(k, 0):>8.4f} {agg.recall_at_k.get(k, 0):>8.4f}",
                end="",
            )
        print(f"{agg.mrr:>8.4f}")

    # Select showcase cases
    helped_b = [l for l in detailed_logs if l["B_gained_gt"]]
    helped_c = [l for l in detailed_logs if l["C_gained_gt"]]
    hurt_b = [l for l in detailed_logs if l["B_lost_gt"]]
    hurt_c = [l for l in detailed_logs if l["C_lost_gt"]]
    no_change = [
        l
        for l in detailed_logs
        if not l["B_gained_gt"]
        and not l["B_lost_gt"]
        and not l["C_gained_gt"]
        and not l["C_lost_gt"]
    ]
    large_expand = sorted(
        detailed_logs, key=lambda l: len(l.get("B_new_from_graph", [])), reverse=True
    )

    showcase = []
    for l in helped_b[:4]:
        l["_showcase_reason"] = "B_helped"
        showcase.append(l)
    for l in helped_c[:3]:
        if l not in showcase:
            l["_showcase_reason"] = "C_helped"
            showcase.append(l)
    for l in hurt_b[:2]:
        if l not in showcase:
            l["_showcase_reason"] = "B_hurt"
            showcase.append(l)
    for l in no_change[:2]:
        if l not in showcase:
            l["_showcase_reason"] = "no_change"
            showcase.append(l)
    for l in large_expand[:2]:
        if l not in showcase:
            l["_showcase_reason"] = "large_expansion"
            showcase.append(l)
    showcase = showcase[:12]

    # Print showcase summary
    print(f"\n{'=' * 60}")
    print(f"SHOWCASE CASES ({len(showcase)} selected)")
    print(f"{'=' * 60}")
    print(f"B helped: {len(helped_b)} queries")
    print(f"C helped: {len(helped_c)} queries")
    print(f"B hurt:   {len(hurt_b)} queries")
    print(f"C hurt:   {len(hurt_c)} queries")
    print(f"No change: {len(no_change)} queries")

    for sc in showcase:
        print(f"\n  [{sc['_showcase_reason']}] {sc['repo']}")
        print(f"Query: {sc['query'][:80]}")
        print(f"GT: {sc['ground_truth_files']}")
        if sc.get("B_gained_gt"):
            print(f"B gained: {sc['B_gained_gt']}")
        if sc.get("B_lost_gt"):
            print(f"B lost:   {sc['B_lost_gt']}")
        if sc.get("B_new_from_graph"):
            print(f"B added {len(sc['B_new_from_graph'])} new files from graph")

    # Save artifacts
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EXPERIMENT_DIR / "detailed_logs.json", "w") as f:
        json.dump(detailed_logs, f, indent=2, default=str)

    with open(EXPERIMENT_DIR / "showcase_cases.json", "w") as f:
        json.dump(showcase, f, indent=2, default=str)

    # Generate report
    report_lines = [
        "# Graph Expansion Experiment Report\n",
        "## 1. Setup\n",
        f"- **Samples**: {len(queries)} queries across {len(set(r for r, _ in queries))} repos",
        f"- **Baseline**: BM25+FileAgg (BM25 with file-level score aggregation)",
        f"- **Graph relation**: method invocations (call graph, 1-hop)",
        f"- **Hub filter**: skip nodes with degree > 50",
        f"- **Mode B**: BM25+FileAgg normalized + graph boost (edge_weight=1.0, boost=0.2) from top-5 seeds",
        f"- **Mode C**: BM25+FileAgg normalized + graph bonus (0.3 * seed_norm), budget max 15 new files",
        f"- **Elapsed**: {elapsed:.1f}s\n",
        "## 2. Results\n",
        f"| Mode | Hit@5 | Hit@10 | Recall@10 | MRR |",
        f"|------|-------|--------|-----------|-----|",
    ]
    for agg in aggregated:
        report_lines.append(
            f"| {agg.retriever} | {agg.hit_at_k.get(5, 0):.4f} | "
            f"{agg.hit_at_k.get(10, 0):.4f} | {agg.recall_at_k.get(10, 0):.4f} | "
            f"{agg.mrr:.4f} |"
        )

    # Deltas
    baseline_agg = next((a for a in aggregated if "baseline" in a.retriever), None)
    if baseline_agg:
        report_lines.append("\n### Deltas vs baseline\n")
        report_lines.append("| Mode | dHit@5 | dHit@10 | dRecall@10 | dMRR |")
        report_lines.append("|------|--------|---------|------------|------|")
        for agg in aggregated:
            if agg.retriever == baseline_agg.retriever:
                continue
            dh5 = agg.hit_at_k.get(5, 0) - baseline_agg.hit_at_k.get(5, 0)
            dh10 = agg.hit_at_k.get(10, 0) - baseline_agg.hit_at_k.get(10, 0)
            dr10 = agg.recall_at_k.get(10, 0) - baseline_agg.recall_at_k.get(10, 0)
            dmrr = agg.mrr - baseline_agg.mrr
            report_lines.append(
                f"| {agg.retriever} | {dh5:+.4f} | {dh10:+.4f} | "
                f"{dr10:+.4f} | {dmrr:+.4f} |"
            )

    # Showcase
    report_lines.append(f"\n## 3. Showcase Cases\n")
    report_lines.append(f"- Queries where B helped: {len(helped_b)}")
    report_lines.append(f"- Queries where C helped: {len(helped_c)}")
    report_lines.append(f"- Queries where B hurt: {len(hurt_b)}")
    report_lines.append(f"- Queries where C hurt: {len(hurt_c)}")
    report_lines.append(f"- No change: {len(no_change)}\n")

    for sc in showcase:
        report_lines.append(f"### [{sc['_showcase_reason']}] {sc['repo']}")
        report_lines.append(f"**Query**: {sc['query'][:120]}")
        report_lines.append(f"**GT files**: {', '.join(sc['ground_truth_files'])}")
        report_lines.append(f"**Baseline hit@10**: {sc['A_baseline_hit@10']}")
        report_lines.append(f"**B hit@10**: {sc['B_graph_expand_hit@10']}")
        report_lines.append(f"**C hit@10**: {sc['C_prior_expand_hit@10']}")
        if sc.get("B_gained_gt"):
            report_lines.append(f"**B gained GT**: {', '.join(sc['B_gained_gt'])}")
        if sc.get("B_lost_gt"):
            report_lines.append(f"**B lost GT**: {', '.join(sc['B_lost_gt'])}")
        report_lines.append(
            f"**New files from graph (B)**: {len(sc.get('B_new_from_graph', []))}"
        )
        report_lines.append("")

    # Conclusion
    report_lines.append("\n## 4. Conclusion\n")
    if baseline_agg:
        b_agg = next((a for a in aggregated if "graph_expand" in a.retriever), None)
        c_agg = next((a for a in aggregated if "prior_expand" in a.retriever), None)
        best_delta_h10 = 0
        if b_agg:
            best_delta_h10 = max(
                best_delta_h10,
                b_agg.hit_at_k.get(10, 0) - baseline_agg.hit_at_k.get(10, 0),
            )
        if c_agg:
            best_delta_h10 = max(
                best_delta_h10,
                c_agg.hit_at_k.get(10, 0) - baseline_agg.hit_at_k.get(10, 0),
            )

        if best_delta_h10 >= 0.03:
            report_lines.append(
                "**Decision: CONTINUE** - local graph expansion gives measurable signal "
                f"(best dHit@10 = {best_delta_h10:+.4f}). Worth building stage-2 graph retrieval."
            )
        elif best_delta_h10 > 0 or len(helped_b) > 5:
            report_lines.append(
                "**Decision: LIMITED CONTINUE** - graph expansion helps on a subset of queries "
                f"(helped {len(helped_b)} queries, best dHit@10 = {best_delta_h10:+.4f}). "
                "May be worth conditional application."
            )
        else:
            report_lines.append(
                "**Decision: STOP** - graph expansion does not provide measurable improvement "
                f"on this benchmark (best dHit@10 = {best_delta_h10:+.4f})."
            )

    report_text = "\n".join(report_lines)
    with open(EXPERIMENT_DIR / "EXPERIMENT_REPORT.md", "w") as f:
        f.write(report_text)

    print(f"\nArtifacts saved to {EXPERIMENT_DIR}/")
    print(f"- detailed_logs.json ({len(detailed_logs)} entries)")
    print(f"- showcase_cases.json ({len(showcase)} cases)")
    print(f"- EXPERIMENT_REPORT.md")


if __name__ == "__main__":
    run_experiment()
