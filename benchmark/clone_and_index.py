import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import asyncio
import logging
import shutil
import subprocess
from pathlib import Path

from backend.app.config import settings
from backend.app.indexer.bm25_builder import build_bm25_index
from backend.app.indexer.callgraph_builder import build_call_graph
from backend.app.indexer.parser import parse_repository
from backend.app.indexer.store import load_chunks, save_indexes
from backend.app.indexer.vector_builder import build_vector_index
from backend.app.models.repo import RepoInfo
from benchmark.config import REPOS_DIR, BenchmarkDataset, BenchmarkSample

logger = logging.getLogger(__name__)


def repo_id_from_name(repo_name: str) -> str:
    return repo_name.replace("/", "--")


def clone_repo_sync(repo_name: str, repos_dir: Path | None = None) -> Path:
    repos_dir = repos_dir or REPOS_DIR
    repos_dir.mkdir(parents=True, exist_ok=True)

    repo_id = repo_id_from_name(repo_name)
    target = repos_dir / repo_id

    if target.exists():
        logger.info(f"Repo already cloned: {target}")
        return target

    clone_url = f"https://github.com/{repo_name}.git"
    logger.info(f"Cloning {clone_url} -> {target}")

    result = subprocess.run(
        ["git", "clone", "--depth", "1", clone_url, str(target)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed for {repo_name}: {result.stderr}")

    return target


def index_repo(repo_name: str, repo_path: Path) -> dict:
    repo_id = repo_id_from_name(repo_name)
    index_dir = settings.indexes_dir / repo_id

    if (index_dir / "chunks.json").exists():
        logger.info(f"Repo already indexed: {repo_id}")
        chunks = load_chunks(repo_id)
        return {"repo_id": repo_id, "chunk_count": len(chunks), "cached": True}

    owner, name = repo_name.split("/", 1)
    repo_info = RepoInfo(
        repo_id=repo_id,
        owner=owner,
        name=name,
        full_name=repo_name,
        url=f"https://github.com/{repo_name}",
        language="Java",
    )

    print(f"Parsing {repo_name}...", flush=True)
    chunks, java_files_count = asyncio.run(parse_repository(repo_path))
    print(f"Found {len(chunks)} chunks from {java_files_count} Java files", flush=True)

    if not chunks:
        return {"repo_id": repo_id, "chunk_count": 0, "error": "No Java files"}
    print(f"Building BM25 index...", flush=True)
    bm25_index, tokenized_corpus = build_bm25_index(chunks)
    print(f"Building vector index...", flush=True)
    faiss_index = asyncio.run(build_vector_index(chunks))
    print(f"Building call graph...", flush=True)
    call_graph = build_call_graph(chunks, repo_path)
    print(f"Saving indexes...", flush=True)
    repo_info.chunk_count = len(chunks)
    save_indexes(
        repo_info, chunks, bm25_index, tokenized_corpus, faiss_index, call_graph
    )
    print(f"Done: {len(chunks)} chunks indexed", flush=True)
    return {"repo_id": repo_id, "chunk_count": len(chunks), "cached": False}


def validate_samples(
    dataset: BenchmarkDataset,
) -> tuple[BenchmarkDataset, dict[str, int]]:
    """Check which samples have ground-truth files in the index. Return filtered dataset + stats."""

    stats: dict[str, int] = {}
    filtered_repos: dict[str, list[BenchmarkSample]] = {}

    for repo_name, samples in dataset.repos.items():
        repo_id = repo_id_from_name(repo_name)
        try:
            chunks = load_chunks(repo_id)
        except FileNotFoundError:
            stats[repo_name] = 0
            continue

        indexed_files = {c.file_path for c in chunks}

        valid_samples = []
        for sample in samples:
            if any(f in indexed_files for f in sample.changed_files):
                valid_samples.append(sample)

        if valid_samples:
            filtered_repos[repo_name] = valid_samples
        stats[repo_name] = len(valid_samples)
        skipped = len(samples) - len(valid_samples)
        if skipped > 0:
            print(
                f"{repo_name}: {skipped}/{len(samples)} samples skipped (files not in index)"
            )

    filtered = BenchmarkDataset(
        repos=filtered_repos,
        total_samples=sum(len(s) for s in filtered_repos.values()),
        total_repos=len(filtered_repos),
    )
    return filtered, stats


def clone_and_index_all(dataset: BenchmarkDataset) -> dict[str, dict]:
    results = {}
    for i, repo_name in enumerate(dataset.repos.keys()):
        print(f"\n[{i + 1}/{dataset.total_repos}] Processing {repo_name}", flush=True)
        try:
            repo_path = clone_repo_sync(repo_name)
            result = index_repo(repo_name, repo_path)
            results[repo_name] = result
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            results[repo_name] = {
                "repo_id": repo_id_from_name(repo_name),
                "error": str(e),
            }
    return results
