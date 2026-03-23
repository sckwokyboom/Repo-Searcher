"""Pluggable retriever implementations for benchmarking."""

import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.config import settings
from app.indexer.bm25_builder import tokenize
from app.indexer.store import load_bm25, load_chunks, load_faiss, load_call_graph
from app.ml.model_manager import get_model_manager
from app.models.search import CodeChunk


class BaseRetriever(ABC):
    name: str

    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self._chunks = None
        self._bm25 = None
        self._corpus = None
        self._faiss = None

    @property
    def chunks(self) -> list[CodeChunk]:
        if self._chunks is None:
            self._chunks = load_chunks(self.repo_id)
        return self._chunks

    @property
    def bm25(self):
        if self._bm25 is None:
            self._bm25, self._corpus = load_bm25(self.repo_id)
        return self._bm25

    @property
    def corpus(self):
        if self._corpus is None:
            self._bm25, self._corpus = load_bm25(self.repo_id)
        return self._corpus

    @property
    def faiss_index(self):
        if self._faiss is None:
            self._faiss = load_faiss(self.repo_id)
        return self._faiss

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        """Return top-K (chunk, score) pairs."""
        ...

    def retrieve_timed(self, query: str, top_k: int = 20) -> tuple[list[tuple[CodeChunk, float]], float]:
        """Return results + elapsed time in seconds."""
        start = time.time()
        results = self.retrieve(query, top_k)
        elapsed = time.time() - start
        return results, elapsed


class BM25Only(BaseRetriever):
    name = "BM25Only"

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[int(i)], float(scores[i])) for i in top_indices if scores[i] > 0]


class VectorOnly(BaseRetriever):
    name = "VectorOnly"

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        manager = get_model_manager()
        query_vec = manager.encode_query(query)
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        distances, indices = self.faiss_index.search(query_vec.astype("float32"), top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                results.append((self.chunks[int(idx)], float(dist)))
        return results


class HybridRRF(BaseRetriever):
    name = "HybridRRF"

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        # BM25
        tokenized_query = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query) if tokenized_query else np.array([])
        bm25_top = np.argsort(bm25_scores)[::-1][:settings.bm25_top_k] if len(bm25_scores) > 0 else []

        # FAISS
        manager = get_model_manager()
        query_vec = manager.encode_query(query)
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        distances, faiss_indices = self.faiss_index.search(query_vec.astype("float32"), settings.faiss_top_k)

        # RRF fusion
        rrf_scores: dict[int, float] = defaultdict(float)
        for rank, idx in enumerate(bm25_top):
            rrf_scores[int(idx)] += 1.0 / (settings.rrf_k + rank + 1)
        for rank, idx in enumerate(faiss_indices[0]):
            if idx >= 0:
                rrf_scores[int(idx)] += 1.0 / (settings.rrf_k + rank + 1)

        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.chunks[idx], score) for idx, score in sorted_candidates]


class HybridRRFRerank(BaseRetriever):
    name = "HybridRRF+Rerank"

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        from app.search.reranker import rerank

        # Get RRF candidates (more than needed for reranking)
        rrf = HybridRRF(self.repo_id)
        rrf._chunks = self._chunks
        rrf._bm25 = self._bm25
        rrf._corpus = self._corpus
        rrf._faiss = self._faiss
        candidates = rrf.retrieve(query, top_k=max(top_k, settings.rrf_top_k))

        # Rerank top candidates
        reranked = rerank(query, candidates[:settings.rrf_top_k])
        # Append remaining candidates after reranked ones
        reranked_ids = {c.chunk_id for c, _ in reranked}
        remaining = [(c, s) for c, s in candidates if c.chunk_id not in reranked_ids]
        return (reranked + remaining)[:top_k]


class FullPipeline(BaseRetriever):
    name = "FullPipeline"

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        from app.search.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever(self.repo_id)
        result = retriever.search_sync(query, top_k=top_k)
        return [(r.chunk, r.score) for r in result["results"]]


RETRIEVER_REGISTRY: dict[str, type[BaseRetriever]] = {
    "bm25": BM25Only,
    "vector": VectorOnly,
    "rrf": HybridRRF,
    "rrf_rerank": HybridRRFRerank,
    "full": FullPipeline,
}


def get_retriever(name: str, repo_id: str) -> BaseRetriever:
    cls = RETRIEVER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown retriever: {name}. Available: {list(RETRIEVER_REGISTRY.keys())}")
    return cls(repo_id)
