import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl.trainer.sft_trainer import SFTTrainer

from backend.app.config import settings
from backend.app.indexer.store import load_chunks
from benchmark.clone_and_index import repo_id_from_name
from benchmark.config import (
    SAMPLES_PATH,
    BenchmarkDataset,
    BenchmarkSample,
    RetrievalResult,
)
from benchmark.evaluator import aggregate_metrics, compute_sample_metrics
from benchmark.retrievers import (
    BM25FileAgg,
    SafeGraphExpansionV2,
    _build_chunk_id_index,
    _build_file_chunk_index,
    _get_call_graph,
)

EXPERIMENT_DIR = Path(__file__).parent / "results" / "graph_frontier_v2"
N_SAMPLES = 75
SEED = 42
TOP_K = 10
K_VALUES = [1, 3, 5, 10]


def load_dataset() -> BenchmarkDataset:
    with open(SAMPLES_PATH) as f:
        return BenchmarkDataset(**json.load(f))


def _indexed_repos(dataset: BenchmarkDataset) -> dict[str, list[BenchmarkSample]]:
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


def get_cached(repo_id: str, cache: dict) -> dict:
    if repo_id not in cache:
        base = BM25FileAgg(repo_id)
        _ = base.chunks
        _ = base.bm25
        cache[repo_id] = {
            "_chunks": base._chunks,
            "_bm25": base._bm25,
            "_corpus": base._corpus,
        }
    return cache[repo_id]


def make_retriever(cls, repo_id: str, cache: dict, **kwargs):
    cached = get_cached(repo_id, cache)
    ret = cls(repo_id, **kwargs) if kwargs else cls(repo_id)
    ret._chunks = cached["_chunks"]
    ret._bm25 = cached["_bm25"]
    ret._corpus = cached["_corpus"]
    return ret


def save_json(data, filename: str):
    path = EXPERIMENT_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved {path}")


def run_e0_oracle_coverage(
    queries: list[tuple[str, BenchmarkSample]],
    cache: dict,
) -> dict:
    print("\n" + "=" * 60)
    print("E0: Oracle Coverage for Frontier")
    print("=" * 60)

    coverage_baseline_10 = 0
    coverage_pool_20 = 0
    per_query = []

    for qi, (repo_name, sample) in enumerate(queries):
        repo_id = repo_id_from_name(repo_name)
        gt_files = set(sample.changed_files)

        baseline_ret = make_retriever(BM25FileAgg, repo_id, cache)
        baseline_results = baseline_ret.retrieve(sample.query, top_k=20)
        baseline_files = [c.file_path for c, _ in baseline_results]
        baseline_top10 = set(baseline_files[:10])
        cached_data = get_cached(repo_id, cache)
        chunks = cached_data["_chunks"]
        file_chunk_idx = _build_file_chunk_index(chunks)
        chunk_id_idx = _build_chunk_id_index(chunks)

        try:
            graph = _get_call_graph(repo_id)
        except Exception:
            graph = None

        neighbor_fps: set[str] = set()
        seed_files = baseline_files[:5]
        if graph is not None:
            for seed_fp in seed_files:
                count = 0
                for ci in file_chunk_idx.get(seed_fp, []):
                    if count >= 10:
                        break
                    chunk_id = chunks[ci].chunk_id
                    if chunk_id not in graph:
                        continue
                    if graph.degree(chunk_id) > 50:
                        continue
                    for nid in list(graph.predecessors(chunk_id)) + list(
                        graph.successors(chunk_id)
                    ):
                        if count >= 10:
                            break
                        ni = chunk_id_idx.get(nid)
                        if ni is None:
                            continue
                        neighbor_fps.add(chunks[ni].file_path)
                        count += 1

        # Expanded pool: baseline top-10 + graph neighbors, capped at 20
        pool = list(baseline_top10)
        for fp in neighbor_fps:
            if fp not in baseline_top10 and len(pool) < 20:
                pool.append(fp)
        pool_set = set(pool)

        has_baseline = 1.0 if gt_files & baseline_top10 else 0.0
        has_pool = 1.0 if gt_files & pool_set else 0.0

        coverage_baseline_10 += has_baseline
        coverage_pool_20 += has_pool

        per_query.append(
            {
                "sample_id": sample.event_id,
                "repo": repo_name,
                "coverage_baseline_10": has_baseline,
                "coverage_pool_20": has_pool,
                "pool_size": len(pool_set),
                "graph_neighbors_added": len(pool_set - baseline_top10),
                "gt_in_pool_not_baseline": sorted(gt_files & pool_set - baseline_top10),
            }
        )

    n = len(queries)
    mean_baseline = coverage_baseline_10 / n
    mean_pool = coverage_pool_20 / n
    delta = mean_pool - mean_baseline
    go = delta >= 0.04

    result = {
        "coverage_at_10_baseline": round(mean_baseline, 4),
        "coverage_at_20_pool": round(mean_pool, 4),
        "delta": round(delta, 4),
        "go": go,
        "n_queries": n,
        "per_query": per_query,
    }

    print(f"Coverage@10 baseline: {mean_baseline:.4f}")
    print(f"Coverage@20 pool:     {mean_pool:.4f}")
    print(f"Delta:                {delta:+.4f}")
    print(f"GO: {go}")

    return result


def run_e1_safe_expansion(
    queries: list[tuple[str, BenchmarkSample]],
    cache: dict,
) -> dict:
    print("\n" + "=" * 60)
    print("E1: Safe Graph Expansion V2")
    print("=" * 60)

    baseline_results_list = []
    v2_results_list = []
    samples_list = []
    per_query_logs = []

    for qi, (repo_name, sample) in enumerate(queries):
        repo_id = repo_id_from_name(repo_name)
        gt_files = set(sample.changed_files)

        if qi % 10 == 0:
            print(f"[{qi + 1}/{len(queries)}] {repo_name}: {sample.query[:60]}...")

        baseline_ret = make_retriever(BM25FileAgg, repo_id, cache)
        try:
            bl_results = baseline_ret.retrieve(sample.query, top_k=TOP_K)
        except Exception as e:
            print(f"ERROR baseline: {e}")
            bl_results = []

        bl_files = []
        bl_scores = []
        seen = set()
        for chunk, score in bl_results:
            if chunk.file_path not in seen:
                bl_files.append(chunk.file_path)
                bl_scores.append(score)
                seen.add(chunk.file_path)

        v2_ret = make_retriever(
            SafeGraphExpansionV2, repo_id, cache, edge_direction="both"
        )
        try:
            v2_raw, diag = v2_ret.retrieve_with_diagnostics(sample.query, top_k=TOP_K)
        except Exception as e:
            print(f"ERROR v2: {e}")
            v2_raw = bl_results
            diag = {}

        v2_files = []
        v2_scores = []
        seen = set()
        for chunk, score in v2_raw:
            if chunk.file_path not in seen:
                v2_files.append(chunk.file_path)
                v2_scores.append(score)
                seen.add(chunk.file_path)

        samples_list.append(sample)

        baseline_results_list.append(
            RetrievalResult(
                sample_id=sample.event_id,
                retriever="A_baseline",
                retrieved_files=bl_files,
                retrieved_methods=[],
                scores=bl_scores,
                top_k=TOP_K,
            )
        )
        v2_results_list.append(
            RetrievalResult(
                sample_id=sample.event_id,
                retriever="D_safe_graph_v2",
                retrieved_files=v2_files,
                retrieved_methods=[],
                scores=v2_scores,
                top_k=TOP_K,
            )
        )

        per_query_logs.append(
            {
                "sample_id": sample.event_id,
                "repo": repo_name,
                "query": sample.query,
                "gt_files": sample.changed_files,
                "baseline_files": bl_files,
                "v2_files": v2_files,
                "v2_new_from_graph": sorted(set(v2_files) - set(bl_files)),
                "v2_gained_gt": sorted(
                    (gt_files & set(v2_files)) - (gt_files & set(bl_files))
                ),
                "v2_lost_gt": sorted(
                    (gt_files & set(bl_files)) - (gt_files & set(v2_files))
                ),
                "diagnostics": diag,
            }
        )

    bl_metrics = []
    v2_metrics = []
    for sample, bl_res, v2_res in zip(
        samples_list, baseline_results_list, v2_results_list
    ):
        bl_metrics.append(compute_sample_metrics(sample, bl_res, K_VALUES))
        v2_metrics.append(compute_sample_metrics(sample, v2_res, K_VALUES))

    bl_agg = aggregate_metrics(bl_metrics, K_VALUES)[0]
    v2_agg = aggregate_metrics(v2_metrics, K_VALUES)[0]

    d_hit5 = v2_agg.hit_at_k.get(5, 0) - bl_agg.hit_at_k.get(5, 0)
    d_hit10 = v2_agg.hit_at_k.get(10, 0) - bl_agg.hit_at_k.get(10, 0)
    d_mrr = v2_agg.mrr - bl_agg.mrr
    d_recall10 = v2_agg.recall_at_k.get(10, 0) - bl_agg.recall_at_k.get(10, 0)

    hit5_ok = d_hit5 >= -0.01
    improvement = d_hit10 >= 0.02 or d_mrr >= 0.01
    go = hit5_ok and improvement

    helped = sum(1 for l in per_query_logs if l["v2_gained_gt"])
    hurt = sum(1 for l in per_query_logs if l["v2_lost_gt"])

    result = {
        "baseline": {
            "hit5": round(bl_agg.hit_at_k.get(5, 0), 4),
            "hit10": round(bl_agg.hit_at_k.get(10, 0), 4),
            "mrr": round(bl_agg.mrr, 4),
            "recall10": round(bl_agg.recall_at_k.get(10, 0), 4),
        },
        "safe_graph_v2": {
            "hit5": round(v2_agg.hit_at_k.get(5, 0), 4),
            "hit10": round(v2_agg.hit_at_k.get(10, 0), 4),
            "mrr": round(v2_agg.mrr, 4),
            "recall10": round(v2_agg.recall_at_k.get(10, 0), 4),
        },
        "deltas": {
            "d_hit5": round(d_hit5, 4),
            "d_hit10": round(d_hit10, 4),
            "d_mrr": round(d_mrr, 4),
            "d_recall10": round(d_recall10, 4),
        },
        "helped": helped,
        "hurt": hurt,
        "go": go,
        "per_query": per_query_logs,
    }

    print(f"\n  {'Mode':<20} {'Hit@5':>8} {'Hit@10':>8} {'MRR':>8} {'Recall@10':>10}")
    print(f"{'-' * 56}")
    print(
        f"{'Baseline':<20} {bl_agg.hit_at_k.get(5, 0):>8.4f} {bl_agg.hit_at_k.get(10, 0):>8.4f} {bl_agg.mrr:>8.4f} {bl_agg.recall_at_k.get(10, 0):>10.4f}"
    )
    print(
        f"{'SafeGraphV2':<20} {v2_agg.hit_at_k.get(5, 0):>8.4f} {v2_agg.hit_at_k.get(10, 0):>8.4f} {v2_agg.mrr:>8.4f} {v2_agg.recall_at_k.get(10, 0):>10.4f}"
    )
    print(f"Delta:   dHit@5={d_hit5:+.4f}  dHit@10={d_hit10:+.4f}  dMRR={d_mrr:+.4f}")
    print(f"Helped: {helped}, Hurt: {hurt}")
    print(f"GO: {go}")

    return result


def run_e2_edge_direction(
    queries: list[tuple[str, BenchmarkSample]],
    cache: dict,
) -> dict:
    print("\n" + "=" * 60)
    print("E2: Edge Direction Ablation")
    print("=" * 60)

    directions = ["outgoing", "incoming", "both"]
    results_by_dir = {}

    for direction in directions:
        print(f"\n  --- Direction: {direction} ---")
        metrics_list = []
        samples_list = []

        for qi, (repo_name, sample) in enumerate(queries):
            repo_id = repo_id_from_name(repo_name)
            ret = make_retriever(
                SafeGraphExpansionV2, repo_id, cache, edge_direction=direction
            )
            try:
                raw_results = ret.retrieve(sample.query, top_k=TOP_K)
            except Exception as e:
                print(f"ERROR [{direction}]: {e}")
                raw_results = []

            files = []
            scores = []
            seen = set()
            for chunk, score in raw_results:
                if chunk.file_path not in seen:
                    files.append(chunk.file_path)
                    scores.append(score)
                    seen.add(chunk.file_path)

            rr = RetrievalResult(
                sample_id=sample.event_id,
                retriever=f"V2_{direction}",
                retrieved_files=files,
                retrieved_methods=[],
                scores=scores,
                top_k=TOP_K,
            )
            metrics_list.append(compute_sample_metrics(sample, rr, K_VALUES))
            samples_list.append(sample)

        agg = aggregate_metrics(metrics_list, K_VALUES)[0]
        results_by_dir[direction] = {
            "hit5": round(agg.hit_at_k.get(5, 0), 4),
            "hit10": round(agg.hit_at_k.get(10, 0), 4),
            "mrr": round(agg.mrr, 4),
            "recall10": round(agg.recall_at_k.get(10, 0), 4),
        }
        print(f"Hit@10={agg.hit_at_k.get(10, 0):.4f}  MRR={agg.mrr:.4f}")

    best_dir = max(
        directions,
        key=lambda d: (results_by_dir[d]["hit10"], results_by_dir[d]["mrr"]),
    )

    result = {
        "directions": results_by_dir,
        "best_direction": best_dir,
    }

    print(f"\n  Best direction: {best_dir}")
    return result


def run_e3_diagnostics(
    queries: list[tuple[str, BenchmarkSample]],
    cache: dict,
    best_direction: str,
) -> dict:
    print("\n" + "=" * 60)
    print("E3: Frontier Quality Diagnostics")
    print("=" * 60)

    all_candidates = []

    for qi, (repo_name, sample) in enumerate(queries):
        repo_id = repo_id_from_name(repo_name)
        gt_files = set(sample.changed_files)

        baseline_ret = make_retriever(BM25FileAgg, repo_id, cache)
        try:
            bl_results = baseline_ret.retrieve(sample.query, top_k=20)
        except Exception:
            bl_results = []
        bl_files = [c.file_path for c, _ in bl_results]
        bl_top10 = set(bl_files[:10])
        bl_rank_map = {fp: i + 1 for i, fp in enumerate(bl_files)}

        v2_ret = make_retriever(
            SafeGraphExpansionV2, repo_id, cache, edge_direction=best_direction
        )
        try:
            _, diag = v2_ret.retrieve_with_diagnostics(sample.query, top_k=TOP_K)
        except Exception:
            continue

        for nb in diag.get("filtered_neighbors", []):
            cand_fp = nb["candidate"]
            all_candidates.append(
                {
                    "query": sample.query,
                    "sample_id": sample.event_id,
                    "repo": repo_name,
                    "seed_file": nb["seed"],
                    "candidate_file": cand_fp,
                    "edge_direction": nb["direction"],
                    "edge_type": "method_invocation",
                    "is_gt": 1 if cand_fp in gt_files else 0,
                    "baseline_rank": bl_rank_map.get(cand_fp, -1),
                    "graph_only": 1 if cand_fp not in set(bl_files[:20]) else 0,
                    "prior_score": round(nb["prior"], 4),
                    "gt_in_baseline_top10": 1 if cand_fp in bl_top10 else 0,
                }
            )

    n_candidates = len(all_candidates)
    n_positive = sum(c["is_gt"] for c in all_candidates)
    n_graph_only = sum(c["graph_only"] for c in all_candidates)
    n_graph_only_positive = sum(c["is_gt"] for c in all_candidates if c["graph_only"])

    positive_rate = n_positive / max(1, n_candidates)
    graph_only_positive_rate = n_graph_only_positive / max(1, n_graph_only)

    query_gt_expansion = set()
    query_ids = set()
    for c in all_candidates:
        query_ids.add(c["sample_id"])
        if c["is_gt"] and not c["gt_in_baseline_top10"]:
            query_gt_expansion.add(c["sample_id"])

    n_queries = len(query_ids)
    queries_with_expansion_gt = len(query_gt_expansion)
    expansion_gt_rate = queries_with_expansion_gt / max(1, n_queries)

    go = expansion_gt_rate >= 0.10 and positive_rate >= 0.10 and n_candidates >= 100

    result = {
        "total_candidates": n_candidates,
        "total_positives": n_positive,
        "positive_rate": round(positive_rate, 4),
        "graph_only_candidates": n_graph_only,
        "graph_only_positives": n_graph_only_positive,
        "graph_only_positive_rate": round(graph_only_positive_rate, 4),
        "queries_with_gt_expansion": queries_with_expansion_gt,
        "expansion_gt_rate": round(expansion_gt_rate, 4),
        "n_queries_with_candidates": n_queries,
        "go": go,
        "candidates": all_candidates,
    }

    print(f"Total candidates:           {n_candidates}")
    print(f"Positives:                  {n_positive} ({positive_rate:.2%})")
    print(f"Graph-only candidates:      {n_graph_only}")
    print(
        f"Graph-only positives:       {n_graph_only_positive} ({graph_only_positive_rate:.2%})"
    )
    print(
        f"Queries with GT expansion:  {queries_with_expansion_gt} / {n_queries} ({expansion_gt_rate:.2%})"
    )
    print(f"GO: {go}")

    jsonl_path = EXPERIMENT_DIR / "frontier_dataset.jsonl"
    with open(jsonl_path, "w") as f:
        for c in all_candidates:
            f.write(json.dumps(c, default=str) + "\n")
    print(f"Saved {jsonl_path} ({n_candidates} entries)")

    return result


def run_e4_llm_probe(
    queries: list[tuple[str, BenchmarkSample]],
    cache: dict,
    candidates: list[dict],
) -> dict:
    print("\n" + "=" * 60)
    print("E4: LLM Probe (Qwen2.5-Coder logits)")
    print("=" * 60)

    query_candidates = defaultdict(list)
    for c in candidates:
        query_candidates[c["sample_id"]].append(c)

    probe_queries = []
    for sid, cands in query_candidates.items():
        has_positive = any(c["is_gt"] for c in cands)
        has_graph_only = any(c["graph_only"] for c in cands)
        if has_positive and has_graph_only:
            probe_queries.append(sid)
    if len(probe_queries) < 10:
        for sid, cands in query_candidates.items():
            if sid not in probe_queries and any(c["is_gt"] for c in cands):
                probe_queries.append(sid)
            if len(probe_queries) >= 20:
                break
    if len(probe_queries) < 10:
        for sid in query_candidates:
            if sid not in probe_queries:
                probe_queries.append(sid)
            if len(probe_queries) >= 20:
                break

    probe_queries = probe_queries[:20]
    print(f"Selected {len(probe_queries)} probe queries")

    probe_pairs = []
    for sid in probe_queries:
        for c in query_candidates[sid]:
            probe_pairs.append(c)
    print(f"Total probe pairs: {len(probe_pairs)}")

    if not probe_pairs:
        return {"go": False, "reason": "no probe pairs", "n_pairs": 0}

    MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
    print(f"Loading {MODEL_NAME}...")
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)
        model.eval()
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return {"go": False, "reason": f"model load failed: {e}", "n_pairs": 0}
    yes_ids = set()
    no_ids = set()
    for text in ["Yes", " Yes", "yes", " yes"]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            yes_ids.add(ids[0])
    for text in ["No", " No", "no", " no"]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            no_ids.add(ids[0])

    print(f"Yes token IDs: {yes_ids}, No token IDs: {no_ids}")

    chunk_cache: dict[str, dict] = {}  # repo_id -> {fp -> chunk_info}
    for pair in probe_pairs:
        repo_name = pair["repo"]
        repo_id = repo_id_from_name(repo_name)
        if repo_id not in chunk_cache:
            cached_data = get_cached(repo_id, cache)
            chunks = cached_data["_chunks"]
            by_fp: dict[str, dict] = {}
            for chunk in chunks:
                if chunk.file_path not in by_fp:
                    by_fp[chunk.file_path] = {
                        "classes": set(),
                        "methods": set(),
                    }
                if chunk.class_name:
                    by_fp[chunk.file_path]["classes"].add(chunk.class_name)
                if chunk.method_name:
                    by_fp[chunk.file_path]["methods"].add(chunk.method_name)
            for fp in by_fp:
                by_fp[fp]["classes"] = sorted(by_fp[fp]["classes"])
                by_fp[fp]["methods"] = sorted(by_fp[fp]["methods"])
            chunk_cache[repo_id] = by_fp

    probe_results = []
    print(f"Running probe on {len(probe_pairs)} pairs...")
    for pi, pair in enumerate(probe_pairs):
        if pi % 20 == 0:
            print(f"[{pi + 1}/{len(probe_pairs)}]...")

        repo_id = repo_id_from_name(pair["repo"])
        by_fp = chunk_cache.get(repo_id, {})

        seed_info = by_fp.get(pair["seed_file"], {"classes": [], "methods": []})
        cand_info = by_fp.get(pair["candidate_file"], {"classes": [], "methods": []})

        prompt = (
            f"Query: {pair['query']}\n"
            f"Seed file: {pair['seed_file']}\n"
            f"Classes: {', '.join(seed_info['classes'][:5])}\n"
            f"Methods: {', '.join(seed_info['methods'][:5])}\n"
            f"Candidate file: {pair['candidate_file']}\n"
            f"Classes: {', '.join(cand_info['classes'][:5])}\n"
            f"Methods: {', '.join(cand_info['methods'][:5])}\n"
            f"Graph relation: {pair['edge_direction']} / {pair['edge_type']}\n"
            f"Question: Is the candidate file relevant for solving the query in the context of the seed file?\n"
            f"Answer:"
        )

        try:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]

            yes_logit = (
                max(logits[tid].item() for tid in yes_ids) if yes_ids else -float("inf")
            )
            no_logit = (
                max(logits[tid].item() for tid in no_ids) if no_ids else -float("inf")
            )

            max_logit = max(yes_logit, no_logit)
            yes_exp = torch.exp(torch.tensor(yes_logit - max_logit)).item()
            no_exp = torch.exp(torch.tensor(no_logit - max_logit)).item()
            p_yes = yes_exp / (yes_exp + no_exp)

        except Exception as e:
            print(f"ERROR at pair {pi}: {e}")
            p_yes = 0.5

        probe_results.append(
            {
                "sample_id": pair["sample_id"],
                "candidate_file": pair["candidate_file"],
                "is_gt": pair["is_gt"],
                "p_yes": round(p_yes, 4),
                "graph_only": pair["graph_only"],
            }
        )

    positives = [r for r in probe_results if r["is_gt"]]
    negatives = [r for r in probe_results if not r["is_gt"]]

    avg_pos = sum(r["p_yes"] for r in positives) / max(1, len(positives))
    avg_neg = sum(r["p_yes"] for r in negatives) / max(1, len(negatives))
    score_gap = avg_pos - avg_neg

    pairwise_correct = 0
    pairwise_total = 0
    by_query = defaultdict(lambda: {"pos": [], "neg": []})
    for r in probe_results:
        if r["is_gt"]:
            by_query[r["sample_id"]]["pos"].append(r["p_yes"])
        else:
            by_query[r["sample_id"]]["neg"].append(r["p_yes"])

    for sid, groups in by_query.items():
        for p in groups["pos"]:
            for n in groups["neg"]:
                pairwise_total += 1
                if p > n:
                    pairwise_correct += 1
                elif p == n:
                    pairwise_correct += 0.5

    pairwise_accuracy = pairwise_correct / max(1, pairwise_total)

    auc = 0.5
    labels = [r["is_gt"] for r in probe_results]
    scores = [r["p_yes"] for r in probe_results]
    if len(set(labels)) > 1:
        auc = roc_auc_score(labels, scores)

    go = (pairwise_accuracy >= 0.60) or (auc >= 0.65)

    result = {
        "n_probe_queries": len(probe_queries),
        "n_probe_pairs": len(probe_pairs),
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "avg_pos_p_yes": round(avg_pos, 4),
        "avg_neg_p_yes": round(avg_neg, 4),
        "score_gap": round(score_gap, 4),
        "pairwise_accuracy": round(pairwise_accuracy, 4),
        "pairwise_total": pairwise_total,
        "auc": round(auc, 4),
        "go": go,
        "probe_results": probe_results,
    }

    print(f"\n  Positives: {len(positives)}, Negatives: {len(negatives)}")
    print(f"Avg p(Yes) pos: {avg_pos:.4f}")
    print(f"Avg p(Yes) neg: {avg_neg:.4f}")
    print(f"Score gap:      {score_gap:+.4f}")
    print(
        f"Pairwise acc:   {pairwise_accuracy:.4f} ({pairwise_correct}/{pairwise_total})"
    )
    print(f"AUC:            {auc:.4f}")
    print(f"GO: {go}")

    del model
    del tokenizer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_e5_smoke_lora(
    candidates: list[dict],
    e4_results: dict,
) -> dict:
    print("\n" + "=" * 60)
    print("E5: Smoke LoRA")
    print("=" * 60)

    train_samples = []
    for c in candidates:
        label = "Yes" if c["is_gt"] else "No"
        text = (
            f"Query: {c['query']}\n"
            f"Seed file: {c['seed_file']}\n"
            f"Candidate file: {c['candidate_file']}\n"
            f"Graph relation: {c['edge_direction']} / {c['edge_type']}\n"
            f"Question: Is the candidate file relevant for solving the query in the context of the seed file?\n"
            f"Answer: {label}"
        )
        train_samples.append({"text": text, "is_gt": c["is_gt"]})

    if len(train_samples) < 50:
        return {"success": False, "reason": f"too few samples ({len(train_samples)})"}

    random.seed(SEED)
    random.shuffle(train_samples)
    split_idx = int(len(train_samples) * 0.8)
    train_data = train_samples[:split_idx]
    val_data = train_samples[split_idx:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    train_pos = sum(s["is_gt"] for s in train_data)
    val_pos = sum(s["is_gt"] for s in val_data)
    print(f"Train positives: {train_pos}/{len(train_data)}")
    print(f"Val positives:   {val_pos}/{len(val_data)}")

    if val_pos == 0 or val_pos == len(val_data):
        return {"success": False, "reason": "val set has no class diversity"}

    train_ds = Dataset.from_list([{"text": s["text"]} for s in train_data])
    val_ds = Dataset.from_list([{"text": s["text"]} for s in val_data])

    MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Loading {MODEL_NAME} for LoRA training...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    output_dir = EXPERIMENT_DIR / "e5_lora_adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=False,
        fp16=False,
        dataloader_pin_memory=False,
        report_to="none",
        max_grad_norm=1.0,
    )
    training_args.max_seq_length = 256

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print("Training...")
    trainer.train()

    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Adapter saved to {final_path}")

    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics.get("eval_loss", float("inf"))
    print(f"Eval loss: {eval_loss:.4f}")

    model.eval()
    yes_ids = set()
    no_ids = set()
    for text in ["Yes", " Yes", "yes", " yes"]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            yes_ids.add(ids[0])
    for text in ["No", " No", "no", " no"]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            no_ids.add(ids[0])

    val_preds = []
    for s in val_data:
        prompt = s["text"].rsplit("Answer:", 1)[0] + "Answer:"
        try:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            yes_logit = (
                max(logits[tid].item() for tid in yes_ids) if yes_ids else -float("inf")
            )
            no_logit = (
                max(logits[tid].item() for tid in no_ids) if no_ids else -float("inf")
            )
            max_l = max(yes_logit, no_logit)
            yes_exp = torch.exp(torch.tensor(yes_logit - max_l)).item()
            no_exp = torch.exp(torch.tensor(no_logit - max_l)).item()
            p_yes = yes_exp / (yes_exp + no_exp)
        except Exception:
            p_yes = 0.5
        val_preds.append({"is_gt": s["is_gt"], "p_yes": round(p_yes, 4)})

    by_pred = {"pos": [], "neg": []}
    for p in val_preds:
        if p["is_gt"]:
            by_pred["pos"].append(p["p_yes"])
        else:
            by_pred["neg"].append(p["p_yes"])

    pw_correct = 0
    pw_total = 0
    for pos in by_pred["pos"]:
        for neg in by_pred["neg"]:
            pw_total += 1
            if pos > neg:
                pw_correct += 1
            elif pos == neg:
                pw_correct += 0.5
    lora_pw = pw_correct / max(1, pw_total)

    all_preds = [p["p_yes"] > 0.5 for p in val_preds]
    class_collapse = len(set(all_preds)) <= 1

    frozen_pw = e4_results.get("pairwise_accuracy", 0.5)
    improved = lora_pw > frozen_pw

    success = improved and not class_collapse

    result = {
        "eval_loss": round(eval_loss, 4),
        "lora_pairwise_accuracy": round(lora_pw, 4),
        "frozen_pairwise_accuracy": round(frozen_pw, 4),
        "class_collapse": class_collapse,
        "improved": improved,
        "success": success,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "val_predictions": val_preds,
    }

    print(f"LoRA pairwise acc:   {lora_pw:.4f}")
    print(f"Frozen pairwise acc: {frozen_pw:.4f}")
    print(f"Class collapse:      {class_collapse}")
    print(f"Success: {success}")

    del model, trainer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return result


def write_summary(decisions: dict, verdict: str):
    lines = [
        "# Graph Frontier Experiments — Summary\n",
        f"**Verdict: {verdict}**\n",
        "## Experiment Results\n",
        "| Experiment | Key Metric | GO? |",
        "|------------|------------|-----|",
    ]

    for exp_name, data in decisions.items():
        if isinstance(data, dict):
            go = data.get("go", data.get("success", "N/A"))
            metric = ""
            if "delta" in data:
                metric = f"delta={data['delta']:+.4f}"
            elif "pairwise_accuracy" in data:
                metric = f"pw_acc={data['pairwise_accuracy']:.4f}, AUC={data.get('auc', 'N/A')}"
            elif "d_hit10" in data.get("deltas", {}):
                metric = f"dHit@10={data['deltas']['d_hit10']:+.4f}"
            elif "best_direction" in data:
                metric = f"best={data['best_direction']}"
            elif "positive_rate" in data:
                metric = f"pos_rate={data['positive_rate']:.2%}"
            elif "lora_pairwise_accuracy" in data:
                metric = f"lora_pw={data['lora_pairwise_accuracy']:.4f}"
            lines.append(f"| {exp_name} | {metric} | {go} |")
        else:
            lines.append(f"| {exp_name} | — | {data} |")

    lines.append(f"\n## Verdict\n\n{verdict}\n")

    path = EXPERIMENT_DIR / "SUMMARY.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSummary written to {path}")


def run_all():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Graph Frontier Experiments E0–E5")
    print("=" * 60)

    dataset = load_dataset()
    print(f"Total: {dataset.total_samples} samples across {dataset.total_repos} repos")

    queries = sample_queries(dataset)
    print(
        f"Sampled {len(queries)} queries across {len(set(r for r, _ in queries))} repos"
    )

    cache: dict = {}
    decisions: dict = {}
    start = time.time()

    e0 = run_e0_oracle_coverage(queries, cache)
    decisions["E0_oracle_coverage"] = e0
    save_json(
        {k: v for k, v in e0.items() if k != "per_query"}, "e0_oracle_coverage.json"
    )
    save_json(e0["per_query"], "e0_per_query.json")

    if not e0["go"]:
        verdict = (
            f"NO-GO at E0: expanded pool delta = {e0['delta']:+.4f} (need >= +0.04)"
        )
        write_summary(decisions, verdict)
        print(f"\n{verdict}")
        print(f"Total time: {time.time() - start:.1f}s")
        return

    e1 = run_e1_safe_expansion(queries, cache)
    decisions["E1_safe_expansion"] = e1
    save_json(
        {k: v for k, v in e1.items() if k != "per_query"}, "e1_safe_expansion.json"
    )

    if not e1["go"]:
        verdict = (
            f"NO-GO at E1: dHit@10={e1['deltas']['d_hit10']:+.4f}, "
            f"dMRR={e1['deltas']['d_mrr']:+.4f}"
        )
        write_summary(decisions, verdict)
        print(f"\n{verdict}")
        print(f"Total time: {time.time() - start:.1f}s")
        return

    e2 = run_e2_edge_direction(queries, cache)
    decisions["E2_edge_ablation"] = e2
    save_json(e2, "e2_edge_ablation.json")

    e3 = run_e3_diagnostics(queries, cache, e2["best_direction"])
    decisions["E3_diagnostics"] = {k: v for k, v in e3.items() if k != "candidates"}
    save_json({k: v for k, v in e3.items() if k != "candidates"}, "e3_diagnostics.json")

    if not e3["go"]:
        verdict = (
            f"NO-GO at E3: positive_rate={e3['positive_rate']:.2%}, "
            f"expansion_gt_rate={e3['expansion_gt_rate']:.2%}"
        )
        write_summary(decisions, verdict)
        print(f"\n{verdict}")
        print(f"Total time: {time.time() - start:.1f}s")
        return

    e4 = run_e4_llm_probe(queries, cache, e3["candidates"])
    decisions["E4_llm_probe"] = {k: v for k, v in e4.items() if k != "probe_results"}
    save_json(e4, "e4_llm_probe.json")

    if not e4["go"]:
        verdict = (
            f"LIMITED GO: LLM probe pairwise_acc={e4['pairwise_accuracy']:.4f}, "
            f"AUC={e4['auc']:.4f} — LoRA likely not worth it"
        )
        write_summary(decisions, verdict)
        print(f"\n{verdict}")
        print(f"Total time: {time.time() - start:.1f}s")
        return

    try:
        e5 = run_e5_smoke_lora(e3["candidates"], e4)
        decisions["E5_smoke_lora"] = e5
        save_json(e5, "e5_lora_results.json")
    except Exception as e:
        print(f"E5 SKIPPED: {e}")
        decisions["E5_smoke_lora"] = {"success": False, "reason": str(e)}

    all_pass = all(
        (d.get("go") or d.get("success")) if isinstance(d, dict) else False
        for d in decisions.values()
    )
    if all_pass:
        verdict = (
            "GO — All experiments passed. LLM+LoRA as frontier reranker is viable."
        )
    elif e4.get("go"):
        verdict = "GO — E0-E4 passed. Proceed to full LoRA training."
    else:
        verdict = "LIMITED GO — Partial signal. Consider routing to subset of queries."

    write_summary(decisions, verdict)
    print(f"\n{'=' * 60}")
    print(f"FINAL: {verdict}")
    print(f"Total time: {time.time() - start:.1f}s")
    print(f"Artifacts in {EXPERIMENT_DIR}/")


if __name__ == "__main__":
    run_all()
