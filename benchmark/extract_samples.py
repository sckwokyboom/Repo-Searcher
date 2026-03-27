import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from benchmark.config import (
    SAMPLES_PATH,
    BenchmarkDataset,
    BenchmarkSample,
    DEFAULT_MAX_REPOS,
    DEFAULT_MIN_SAMPLES,
)
from benchmark.patch_parser import extract_java_files, extract_methods_from_patch
from benchmark.query_cleaner import clean_query

logger = logging.getLogger(__name__)


def extract_samples(
    jsonl_path: str | Path,
    max_repos: int = DEFAULT_MAX_REPOS,
    min_total_samples: int = DEFAULT_MIN_SAMPLES,
    output_path: str | Path | None = None,
) -> BenchmarkDataset:
    jsonl_path = Path(jsonl_path)
    output_path = Path(output_path) if output_path else SAMPLES_PATH

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    logger.info(f"Streaming {jsonl_path} for Java samples...")

    repo_samples: dict[str, list[dict]] = defaultdict(list)
    total_lines = 0
    java_lines = 0

    with open(jsonl_path, "r") as f:
        for line in f:
            total_lines += 1
            if total_lines % 50000 == 0:
                print(
                    f"Processed {total_lines} lines, found {java_lines} Java samples...",
                    flush=True,
                )

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            patch = entry.get("patch", "")
            java_files = extract_java_files(patch)
            if not java_files:
                continue

            java_lines += 1
            repo = entry.get("repo", "")
            if not repo:
                continue

            query, is_low_quality = clean_query(entry.get("description", ""))
            if not query:
                continue

            methods = extract_methods_from_patch(patch)

            repo_samples[repo].append(
                {
                    "event_id": entry.get("event_id", ""),
                    "repo": repo,
                    "sha": entry.get("sha", ""),
                    "query": query,
                    "raw_description": entry.get("description", ""),
                    "changed_files": java_files,
                    "changed_methods": methods,
                    "low_quality": is_low_quality,
                }
            )

    print(
        f"Done: {total_lines} total lines, {java_lines} Java samples across {len(repo_samples)} repos",
        flush=True,
    )

    repo_counts = sorted(repo_samples.items(), key=lambda x: len(x[1]), reverse=True)
    print("\nTop 20 repos by sample count:")
    for repo, samples in repo_counts[:20]:
        print(f"{repo}: {len(samples)} samples")

    selected_repos: dict[str, list[BenchmarkSample]] = {}
    total_selected = 0

    for repo, samples in repo_counts[:max_repos]:
        selected = [BenchmarkSample(**s) for s in samples]
        selected_repos[repo] = selected
        total_selected += len(selected)

        if total_selected >= min_total_samples and len(selected_repos) >= 3:
            if len(samples) < 3:
                break

    dataset = BenchmarkDataset(
        repos=selected_repos,
        total_samples=sum(len(s) for s in selected_repos.values()),
        total_repos=len(selected_repos),
    )

    print(
        f"\nSelected {dataset.total_repos} repos with {dataset.total_samples} total samples:"
    )
    for repo, samples in selected_repos.items():
        lq = sum(1 for s in samples if s.low_quality)
        print(f"{repo}: {len(samples)} samples ({lq} low-quality)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset.model_dump(), f, indent=2)
    print(f"\nSaved to {output_path}")

    return dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    jsonl_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/Users/sckwoky/Downloads/claude-20250813.jsonl"
    )
    extract_samples(jsonl_path)
