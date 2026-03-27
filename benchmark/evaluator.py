from collections import defaultdict

from benchmark.config import (
    AggregatedMetrics,
    BenchmarkSample,
    DEFAULT_TOP_K_VALUES,
    EvalResults,
    RetrievalResult,
    SampleMetrics,
)


def compute_sample_metrics(
    sample: BenchmarkSample,
    result: RetrievalResult,
    k_values: list[int] | None = None,
) -> SampleMetrics:
    k_values = k_values or DEFAULT_TOP_K_VALUES
    gt_files = set(sample.changed_files)
    gt_methods = set(sample.changed_methods)
    retrieved_files = result.retrieved_files
    retrieved_methods = result.retrieved_methods

    metrics = SampleMetrics(
        sample_id=sample.event_id,
        retriever=result.retriever,
        repo=sample.repo,
    )

    for k in k_values:
        top_k_files = set(retrieved_files[:k])

        if gt_files:
            recall = len(gt_files & top_k_files) / len(gt_files)
        else:
            recall = 0.0
        metrics.recall_at_k[k] = recall

        if top_k_files:
            precision = len(gt_files & top_k_files) / len(top_k_files)
        else:
            precision = 0.0
        metrics.precision_at_k[k] = precision
        metrics.hit_at_k[k] = 1.0 if gt_files & top_k_files else 0.0

    mrr = 0.0
    for rank, f in enumerate(retrieved_files, 1):
        if f in gt_files:
            mrr = 1.0 / rank
            break
    metrics.mrr = mrr

    if gt_methods:
        for k in k_values:
            top_k_methods = set(retrieved_methods[:k])
            if gt_methods:
                metrics.method_recall_at_k[k] = len(gt_methods & top_k_methods) / len(
                    gt_methods
                )
            metrics.method_hit_at_k[k] = 1.0 if gt_methods & top_k_methods else 0.0

    return metrics


def aggregate_metrics(
    per_sample: list[SampleMetrics],
    k_values: list[int] | None = None,
) -> list[AggregatedMetrics]:
    k_values = k_values or DEFAULT_TOP_K_VALUES

    by_retriever: dict[str, list[SampleMetrics]] = defaultdict(list)
    for m in per_sample:
        by_retriever[m.retriever].append(m)

    aggregated = []
    for retriever, samples in sorted(by_retriever.items()):
        n = len(samples)
        agg = AggregatedMetrics(retriever=retriever, num_samples=n)

        for k in k_values:
            agg.recall_at_k[k] = sum(s.recall_at_k.get(k, 0) for s in samples) / n
            agg.precision_at_k[k] = sum(s.precision_at_k.get(k, 0) for s in samples) / n
            agg.hit_at_k[k] = sum(s.hit_at_k.get(k, 0) for s in samples) / n
            agg.method_recall_at_k[k] = (
                sum(s.method_recall_at_k.get(k, 0) for s in samples) / n
            )
            agg.method_hit_at_k[k] = (
                sum(s.method_hit_at_k.get(k, 0) for s in samples) / n
            )

        agg.mrr = sum(s.mrr for s in samples) / n
        aggregated.append(agg)

    return aggregated


def aggregate_by_repo(
    per_sample: list[SampleMetrics],
    k_values: list[int] | None = None,
) -> dict[str, list[AggregatedMetrics]]:
    k_values = k_values or DEFAULT_TOP_K_VALUES

    by_repo: dict[str, list[SampleMetrics]] = defaultdict(list)
    for m in per_sample:
        by_repo[m.repo].append(m)

    result = {}
    for repo, samples in sorted(by_repo.items()):
        result[repo] = aggregate_metrics(samples, k_values)
    return result


def evaluate(
    samples: list[BenchmarkSample],
    results: list[RetrievalResult],
    k_values: list[int] | None = None,
) -> EvalResults:
    k_values = k_values or DEFAULT_TOP_K_VALUES

    result_lookup: dict[tuple[str, str], RetrievalResult] = {}
    for r in results:
        result_lookup[(r.sample_id, r.retriever)] = r

    per_sample_metrics = []
    for sample in samples:
        for retriever in {r.retriever for r in results}:
            key = (sample.event_id, retriever)
            if key in result_lookup:
                m = compute_sample_metrics(sample, result_lookup[key], k_values)
                per_sample_metrics.append(m)

    return EvalResults(
        per_sample=per_sample_metrics,
        per_retriever=aggregate_metrics(per_sample_metrics, k_values),
        per_repo=aggregate_by_repo(per_sample_metrics, k_values),
    )
