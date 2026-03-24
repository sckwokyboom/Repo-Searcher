"""Visualization: rich console tables and matplotlib charts."""

from pathlib import Path

from datetime import datetime

from benchmark.config import AggregatedMetrics, EvalResults, PLOTS_DIR, RESULTS_DIR


def print_summary_table(eval_results: EvalResults, k_values: list[int] | None = None):
    """Print a rich table with aggregated metrics per retriever."""
    try:
        from rich.console import Console
        from rich.table import Table
        _print_rich_table(eval_results, k_values)
    except ImportError:
        _print_plain_table(eval_results, k_values)


def _print_rich_table(eval_results: EvalResults, k_values: list[int] | None = None):
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console()
    k_values = k_values or [1, 3, 5, 10, 20]

    # Main summary table
    table = Table(title="Retrieval Benchmark Results (File-Level)", show_lines=True)
    table.add_column("Retriever", style="bold cyan", min_width=18)
    for k in k_values:
        table.add_column(f"Recall@{k}", justify="center", min_width=9)
    table.add_column("MRR", justify="center", min_width=7)
    for k in [5, 10]:
        table.add_column(f"Hit@{k}", justify="center", min_width=7)
    table.add_column("Samples", justify="center", min_width=8)

    # Find best values for highlighting
    best = {}
    for metric_name in [f"recall_{k}" for k in k_values] + ["mrr"] + [f"hit_{k}" for k in [5, 10]]:
        best[metric_name] = 0.0

    for agg in eval_results.per_retriever:
        for k in k_values:
            best[f"recall_{k}"] = max(best[f"recall_{k}"], agg.recall_at_k.get(k, 0))
        best["mrr"] = max(best["mrr"], agg.mrr)
        for k in [5, 10]:
            best[f"hit_{k}"] = max(best[f"hit_{k}"], agg.hit_at_k.get(k, 0))

    for agg in eval_results.per_retriever:
        row = [agg.retriever]
        for k in k_values:
            val = agg.recall_at_k.get(k, 0)
            cell = f"{val:.3f}"
            if val == best[f"recall_{k}"] and val > 0:
                cell = f"[bold green]{cell}[/bold green]"
            row.append(cell)

        val = agg.mrr
        cell = f"{val:.3f}"
        if val == best["mrr"] and val > 0:
            cell = f"[bold green]{cell}[/bold green]"
        row.append(cell)

        for k in [5, 10]:
            val = agg.hit_at_k.get(k, 0)
            cell = f"{val:.3f}"
            if val == best[f"hit_{k}"] and val > 0:
                cell = f"[bold green]{cell}[/bold green]"
            row.append(cell)

        row.append(str(agg.num_samples))
        table.add_row(*row)

    console.print()
    console.print(table)

    # Method-level table (if data available)
    has_method = any(
        any(v > 0 for v in agg.method_recall_at_k.values())
        for agg in eval_results.per_retriever
    )
    if has_method:
        mtable = Table(title="Retrieval Benchmark Results (Method-Level)", show_lines=True)
        mtable.add_column("Retriever", style="bold cyan", min_width=18)
        for k in k_values:
            mtable.add_column(f"Recall@{k}", justify="center", min_width=9)
        for k in [5, 10]:
            mtable.add_column(f"Hit@{k}", justify="center", min_width=7)

        for agg in eval_results.per_retriever:
            row = [agg.retriever]
            for k in k_values:
                row.append(f"{agg.method_recall_at_k.get(k, 0):.3f}")
            for k in [5, 10]:
                row.append(f"{agg.method_hit_at_k.get(k, 0):.3f}")
            mtable.add_row(*row)

        console.print()
        console.print(mtable)

    # Per-repo breakdown
    if eval_results.per_repo:
        console.print()
        console.print("[bold]Per-Repository Breakdown (Recall@5):[/bold]")

        repo_table = Table(show_lines=True)
        repo_table.add_column("Repository", style="bold", min_width=30)
        retrievers = [agg.retriever for agg in eval_results.per_retriever]
        for r in retrievers:
            repo_table.add_column(r, justify="center", min_width=10)
        repo_table.add_column("Samples", justify="center", min_width=8)

        for repo, aggs in sorted(eval_results.per_repo.items()):
            row = [repo]
            agg_by_name = {a.retriever: a for a in aggs}
            for r in retrievers:
                a = agg_by_name.get(r)
                row.append(f"{a.recall_at_k.get(5, 0):.3f}" if a else "-")
            row.append(str(aggs[0].num_samples if aggs else 0))
            repo_table.add_row(*row)

        console.print(repo_table)


def _print_plain_table(eval_results: EvalResults, k_values: list[int] | None = None):
    """Fallback plain text table when rich is not installed."""
    k_values = k_values or [1, 3, 5, 10, 20]

    header = f"{'Retriever':<20}"
    for k in k_values:
        header += f"{'R@'+str(k):>10}"
    header += f"{'MRR':>10}{'Hit@5':>10}{'Hit@10':>10}{'N':>8}"
    print("\n" + header)
    print("-" * len(header))

    for agg in eval_results.per_retriever:
        row = f"{agg.retriever:<20}"
        for k in k_values:
            row += f"{agg.recall_at_k.get(k, 0):>10.3f}"
        row += f"{agg.mrr:>10.3f}"
        row += f"{agg.hit_at_k.get(5, 0):>10.3f}"
        row += f"{agg.hit_at_k.get(10, 0):>10.3f}"
        row += f"{agg.num_samples:>8}"
        print(row)
    print()


def plot_results(eval_results: EvalResults, output_dir: Path | None = None):
    """Generate matplotlib charts."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    output_dir = output_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    retrievers = [agg.retriever for agg in eval_results.per_retriever]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4', '#795548']

    # 1. Grouped bar chart: key metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = {
        'Recall@1': [agg.recall_at_k.get(1, 0) for agg in eval_results.per_retriever],
        'Recall@5': [agg.recall_at_k.get(5, 0) for agg in eval_results.per_retriever],
        'Recall@10': [agg.recall_at_k.get(10, 0) for agg in eval_results.per_retriever],
        'MRR': [agg.mrr for agg in eval_results.per_retriever],
        'Hit@5': [agg.hit_at_k.get(5, 0) for agg in eval_results.per_retriever],
    }

    import numpy as np
    x = np.arange(len(metrics))
    width = 0.8 / len(retrievers)

    for i, (retriever, color) in enumerate(zip(retrievers, colors)):
        values = [metrics[m][i] for m in metrics]
        offset = (i - len(retrievers) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=retriever, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.set_ylabel('Score')
    ax.set_title('Retrieval Benchmark: Metrics Comparison')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=150)
    plt.close()

    # 2. Line chart: Recall@K curve
    fig, ax = plt.subplots(figsize=(10, 6))
    k_values = sorted(eval_results.per_retriever[0].recall_at_k.keys()) if eval_results.per_retriever else []

    for i, (agg, color) in enumerate(zip(eval_results.per_retriever, colors)):
        recalls = [agg.recall_at_k.get(k, 0) for k in k_values]
        ax.plot(k_values, recalls, marker='o', color=color, label=agg.retriever, linewidth=2, markersize=6)

    ax.set_xlabel('K (top-K)')
    ax.set_ylabel('Recall@K')
    ax.set_title('Recall@K Curve by Retriever')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'recall_at_k_curve.png', dpi=150)
    plt.close()

    # 3. Per-repo heatmap
    if eval_results.per_repo and len(eval_results.per_repo) > 1:
        repos = sorted(eval_results.per_repo.keys())
        data = []
        for repo in repos:
            aggs = {a.retriever: a for a in eval_results.per_repo[repo]}
            row = [aggs[r].recall_at_k.get(5, 0) if r in aggs else 0 for r in retrievers]
            data.append(row)

        fig, ax = plt.subplots(figsize=(max(8, len(retrievers) * 2), max(4, len(repos) * 0.6)))
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(range(len(retrievers)))
        ax.set_xticklabels(retrievers, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(repos)))
        # Shorten repo names for display
        short_repos = [r.split('/')[-1] if '/' in r else r for r in repos]
        ax.set_yticklabels(short_repos, fontsize=9)

        # Add values
        for i in range(len(repos)):
            for j in range(len(retrievers)):
                ax.text(j, i, f'{data[i][j]:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if data[i][j] > 0.5 else 'black')

        plt.colorbar(im, label='Recall@5')
        ax.set_title('Recall@5 per Repository')
        plt.tight_layout()
        plt.savefig(output_dir / 'per_repo_heatmap.png', dpi=150)
        plt.close()

    print(f"Plots saved to {output_dir}")


def save_results_markdown(
    eval_results: EvalResults,
    k_values: list[int] | None = None,
    output_path: Path | None = None,
):
    """Append benchmark results to a persistent markdown log."""
    k_values = k_values or [1, 3, 5, 10, 20]
    output_path = output_path or (RESULTS_DIR / "benchmark_log.md")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    retrievers = [agg.retriever for agg in eval_results.per_retriever]
    n_samples = eval_results.per_retriever[0].num_samples if eval_results.per_retriever else 0

    lines = []
    lines.append(f"\n## Run: {timestamp}")
    lines.append(f"Samples: {n_samples} | Retrievers: {', '.join(retrievers)}\n")

    # File-level table
    lines.append("### File-Level Metrics")
    header = "| Retriever |"
    sep = "|-----------|"
    for k in k_values:
        header += f" R@{k} |"
        sep += "------|"
    header += " MRR | Hit@5 | Hit@10 |"
    sep += "------|-------|--------|"
    lines.append(header)
    lines.append(sep)

    for agg in eval_results.per_retriever:
        row = f"| {agg.retriever} |"
        for k in k_values:
            row += f" {agg.recall_at_k.get(k, 0):.3f} |"
        row += f" {agg.mrr:.3f} | {agg.hit_at_k.get(5, 0):.3f} | {agg.hit_at_k.get(10, 0):.3f} |"
        lines.append(row)

    # Per-repo breakdown
    if eval_results.per_repo:
        lines.append("\n### Per-Repository (Recall@5)")
        header = "| Repository |"
        sep = "|------------|"
        for r in retrievers:
            header += f" {r} |"
            sep += "------|"
        header += " N |"
        sep += "---|"
        lines.append(header)
        lines.append(sep)

        for repo, aggs in sorted(eval_results.per_repo.items()):
            agg_by_name = {a.retriever: a for a in aggs}
            row = f"| {repo} |"
            for r in retrievers:
                a = agg_by_name.get(r)
                row += f" {a.recall_at_k.get(5, 0):.3f} |" if a else " - |"
            row += f" {aggs[0].num_samples if aggs else 0} |"
            lines.append(row)

    lines.append("")

    # Append to file (create with header if new)
    if not output_path.exists():
        with open(output_path, "w") as f:
            f.write("# Retrieval Benchmark Results Log\n")

    with open(output_path, "a") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Results appended to {output_path}")
