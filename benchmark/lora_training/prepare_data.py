import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from backend.app.indexer.bm25_builder import tokenize
from backend.app.indexer.store import load_bm25, load_chunks
from benchmark.query_cleaner import clean_query

REPOS_FULL_DIR = Path(__file__).parent.parent / "results" / "repos_full_history"
OUTPUT_DIR = Path(__file__).parent / "data"


def get_java_commits(repo_path: Path, max_commits: int = 2000) -> list[dict]:
    result = subprocess.run(
        [
            "git",
            "log",
            f"--max-count={max_commits}",
            "--diff-filter=M",
            "--name-only",
            "--format=COMMIT_SEP%n%H%n%s%n%b%nFILES_START",
            "--",
            "*.java",
        ],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
    )

    commits = []
    for block in result.stdout.split("COMMIT_SEP\n"):
        block = block.strip()
        if not block:
            continue

        parts = block.split("FILES_START\n", 1)
        if len(parts) != 2:
            continue

        header = parts[0].strip()
        files_text = parts[1].strip()

        lines = header.split("\n")
        if len(lines) < 2:
            continue

        sha = lines[0].strip()
        subject = lines[1].strip()
        body = "\n".join(lines[2:]).strip() if len(lines) > 2 else ""

        description = subject
        if body:
            description = f"{subject}\n{body}"

        java_files = [
            f.strip() for f in files_text.split("\n") if f.strip().endswith(".java")
        ]

        if not java_files:
            continue

        query, low_quality = clean_query(description)
        if low_quality or len(query) < 10:
            continue

        commits.append(
            {
                "sha": sha,
                "query": query,
                "raw_description": description,
                "changed_files": java_files,
            }
        )

    return commits


def generate_training_samples(
    repo_id: str,
    commits: list[dict],
    max_negatives: int = 5,
    max_positives: int = 5,
) -> list[dict]:
    chunks = load_chunks(repo_id)
    bm25, corpus = load_bm25(repo_id)
    file_to_chunks: dict[str, list[int]] = {}
    for i, chunk in enumerate(chunks):
        file_to_chunks.setdefault(chunk.file_path, []).append(i)

    all_file_paths = set(file_to_chunks.keys())

    samples = []
    skipped = 0

    for commit in commits:
        query = commit["query"]
        changed_in_index = [f for f in commit["changed_files"] if f in all_file_paths]
        if not changed_in_index:
            skipped += 1
            continue

        positive_chunks = []
        for f in changed_in_index[:max_positives]:
            chunk_indices = file_to_chunks[f]
            tokenized_query = tokenize(query)
            if tokenized_query:
                scores = bm25.get_scores(tokenized_query)
                best_idx = max(chunk_indices, key=lambda i: scores[i])
            else:
                best_idx = chunk_indices[0]
            positive_chunks.append(best_idx)

        changed_set = set(changed_in_index)
        tokenized_query = tokenize(query)
        if tokenized_query:
            scores = bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:50]
            negative_chunks = []
            for idx in top_indices:
                if chunks[idx].file_path not in changed_set and scores[idx] > 0:
                    negative_chunks.append(int(idx))
                    if len(negative_chunks) >= max_negatives:
                        break
        else:
            negative_chunks = []

        for idx in positive_chunks:
            chunk = chunks[idx]
            prompt = _format_prompt(query, chunk)
            score = 9
            samples.append(
                {
                    "prompt": prompt,
                    "completion": str(score),
                    "label": "positive",
                    "repo": repo_id,
                    "query": query,
                    "file": chunk.file_path,
                }
            )

        for idx in negative_chunks:
            chunk = chunks[idx]
            prompt = _format_prompt(query, chunk)
            score = 2
            samples.append(
                {
                    "prompt": prompt,
                    "completion": str(score),
                    "label": "negative",
                    "repo": repo_id,
                    "query": query,
                    "file": chunk.file_path,
                }
            )

    print(
        f"Generated {len(samples)} samples ({skipped} commits skipped - files not in index)"
    )
    return samples


def _format_prompt(query: str, chunk) -> str:
    """Format the scoring prompt for training."""
    parts = []
    parts.append(f"Query: {query}")
    if chunk.file_path:
        parts.append(f"File: {chunk.file_path}")
    if chunk.class_name:
        parts.append(f"Class: {chunk.class_name}")
    if chunk.method_name:
        parts.append(f"Method: {chunk.method_name}")
    if chunk.signature:
        parts.append(f"Signature: {chunk.signature[:300]}")
    if chunk.javadoc:
        parts.append(f"Doc: {chunk.javadoc[:150]}")

    code_context = "\n".join(parts)
    return (
        "Is this Java code relevant to the search query? "
        "Answer with a single number from 0 (irrelevant) to 10 (perfect match).\n\n"
        f"{code_context}\n\n"
        "Relevance score (0-10):"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    repos = {
        "jdereg--java-util": REPOS_FULL_DIR / "jdereg--java-util",
        "panghy--javaflow": REPOS_FULL_DIR / "panghy--javaflow",
    }

    all_samples = []
    for repo_id, repo_path in repos.items():
        if not repo_path.exists():
            print(f"Skipping {repo_id}: repo not found at {repo_path}")
            continue

        print(f"\nProcessing {repo_id}...")
        commits = get_java_commits(repo_path)
        print(f"Found {len(commits)} commits with Java changes")

        samples = generate_training_samples(repo_id, commits)
        all_samples.extend(samples)

    import random

    random.seed(42)
    random.shuffle(all_samples)
    split = int(len(all_samples) * 0.9)
    train = all_samples[:split]
    val = all_samples[split:]

    train_path = OUTPUT_DIR / "train_scorer.jsonl"
    val_path = OUTPUT_DIR / "val_scorer.jsonl"

    for path, data in [(train_path, train), (val_path, val)]:
        with open(path, "w") as f:
            for sample in data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(data)} samples to {path}")

    pos = sum(1 for s in all_samples if s["label"] == "positive")
    neg = sum(1 for s in all_samples if s["label"] == "negative")
    print(f"\nTotal: {len(all_samples)} samples ({pos} positive, {neg} negative)")
    print(f"Train: {len(train)}, Val: {len(val)}")


if __name__ == "__main__":
    main()
