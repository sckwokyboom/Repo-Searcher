import json
import pickle
from pathlib import Path

import faiss
import networkx as nx
import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.models.repo import RepoInfo
from app.models.search import CodeChunk


def save_indexes(
    repo: RepoInfo,
    chunks: list[CodeChunk],
    bm25_index: BM25Okapi,
    tokenized_corpus: list[list[str]],
    faiss_index: faiss.IndexFlatIP,
    call_graph: nx.DiGraph,
):
    index_dir = settings.indexes_dir / repo.repo_id
    index_dir.mkdir(parents=True, exist_ok=True)

    with open(index_dir / "chunks.json", "w") as f:
        json.dump([c.model_dump() for c in chunks], f)

    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25_index, "corpus": tokenized_corpus}, f)

    faiss.write_index(faiss_index, str(index_dir / "faiss.index"))

    with open(index_dir / "callgraph.pkl", "wb") as f:
        pickle.dump(call_graph, f)

    metadata = {
        "repo_info": repo.model_dump(mode="json"),
        "chunk_count": len(chunks),
    }
    with open(index_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_chunks(repo_id: str) -> list[CodeChunk]:
    path = settings.indexes_dir / repo_id / "chunks.json"
    with open(path) as f:
        data = json.load(f)
    return [CodeChunk(**c) for c in data]


def load_bm25(repo_id: str) -> tuple[BM25Okapi, list[list[str]]]:
    path = settings.indexes_dir / repo_id / "bm25.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["corpus"]


def load_faiss(repo_id: str) -> faiss.IndexFlatIP:
    path = settings.indexes_dir / repo_id / "faiss.index"
    return faiss.read_index(str(path))


def load_call_graph(repo_id: str) -> nx.DiGraph:
    path = settings.indexes_dir / repo_id / "callgraph.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
