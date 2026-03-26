import asyncio
from collections import defaultdict

import numpy as np

from app.config import settings
from app.indexer.bm25_builder import tokenize
from app.indexer.store import load_bm25, load_chunks, load_faiss
from app.ml.model_manager import get_model_manager
from app.models.search import SearchResult
from app.search.graph_expander import GraphExpander
from app.search.graph_mcts import GraphMCTS
from app.search.mcts_rewriter import mcts_rewrite
from app.search.reranker import rerank


_index_cache: dict[str, dict] = {}


def _get_indexes(repo_id: str) -> dict:
    if repo_id not in _index_cache:
        chunks = load_chunks(repo_id)
        bm25, corpus = load_bm25(repo_id)
        faiss_index = load_faiss(repo_id)
        _index_cache[repo_id] = {
            "chunks": chunks,
            "bm25": bm25,
            "corpus": corpus,
            "faiss": faiss_index,
        }
    return _index_cache[repo_id]


class HybridRetriever:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.indexes = _get_indexes(repo_id)

    def search_sync(self, query: str, top_k: int = 5) -> dict:
        """Synchronous search — designed to run in a thread pool."""
        return self._search_impl(query, top_k)

    async def search(self, query: str, top_k: int = 5) -> dict:
        return self._search_impl(query, top_k)

    def _search_impl(self, query: str, top_k: int = 5) -> dict:
        chunks = self.indexes["chunks"]
        bm25 = self.indexes["bm25"]
        corpus = self.indexes["corpus"]

        # MCTS query rewriting with hybrid reward (BM25 + semantic + LLM)
        faiss_index = self.indexes["faiss"]
        mcts_result = mcts_rewrite(query, bm25, corpus, chunks, faiss_index)
        expanded_query = mcts_result["best_query"]
        keywords = mcts_result["keywords"]
        mcts_trace = mcts_result["trace"]

        # BM25 retrieval
        tokenized_query = tokenize(expanded_query)
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][: settings.bm25_top_k]

        # FAISS retrieval (uses rewritten query for better semantic match)
        manager = get_model_manager()
        query_vector = manager.encode_query(expanded_query)
        query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
        faiss_index = self.indexes["faiss"]
        distances, faiss_top_indices = faiss_index.search(
            query_vector.astype("float32"), settings.faiss_top_k
        )
        faiss_top_indices = faiss_top_indices[0]
        faiss_distances = distances[0]

        # RRF Fusion
        rrf_scores: dict[int, float] = defaultdict(float)
        bm25_ranks: dict[int, int] = {}
        vector_ranks: dict[int, int] = {}

        for rank, idx in enumerate(bm25_top_indices):
            rrf_scores[idx] += 1.0 / (settings.rrf_k + rank + 1)
            bm25_ranks[idx] = rank + 1

        for rank, idx in enumerate(faiss_top_indices):
            if idx >= 0:
                rrf_scores[idx] += 1.0 / (settings.rrf_k + rank + 1)
                vector_ranks[idx] = rank + 1

        # Sort by RRF score, take top-K for reranking
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = sorted_candidates[: settings.rrf_top_k]

        # Call Graph expansion
        graph_expander = GraphExpander(self.repo_id)
        candidate_chunks = [(chunks[idx], rrf_score) for idx, rrf_score in top_candidates]

        # Reranking — use original query (not rewritten) for relevance scoring
        reranked = rerank(query, candidate_chunks)

        # Build final results
        results = []
        for chunk, score in reranked[:top_k]:
            idx = next(i for i, c in enumerate(chunks) if c.chunk_id == chunk.chunk_id)
            callers, callees = graph_expander.get_neighbors(chunk.chunk_id)
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    bm25_rank=bm25_ranks.get(idx),
                    vector_rank=vector_ranks.get(idx),
                    rrf_score=rrf_scores.get(idx, 0.0),
                    callers=callers,
                    callees=callees,
                    source="search",
                )
            )

        # Call Graph MCTS: explore graph from top results to find related code
        graph_mcts_trace = None
        seed_ids = [r.chunk.chunk_id for r in results]
        if seed_ids:
            call_graph = graph_expander.graph
            graph_mcts = GraphMCTS(call_graph, chunks, faiss_index)
            graph_result = graph_mcts.explore(query, seed_ids)

            # Add graph discoveries to results
            existing_ids = {r.chunk.chunk_id for r in results}
            for discovery in graph_result["discoveries"]:
                if discovery.chunk.chunk_id not in existing_ids:
                    # Enrich with call graph neighbors
                    callers, callees = graph_expander.get_neighbors(discovery.chunk.chunk_id)
                    discovery.callers = callers
                    discovery.callees = callees
                    results.append(discovery)
                    existing_ids.add(discovery.chunk.chunk_id)

            graph_mcts_trace = graph_result["trace"]

        return {
            "results": results,
            "expanded_keywords": keywords,
            "mcts_trace": mcts_trace,
            "graph_mcts_trace": graph_mcts_trace,
        }
