"""
Simple retriever: LLM query rewrite (Qwen + LoRA) → multi-query BM25 with RRF.
Call graph neighbors are attached for navigation.
"""

import json
import re
from collections import defaultdict

import numpy as np

from app.config import settings
from app.indexer.bm25_builder import tokenize
from app.indexer.store import load_bm25, load_chunks
from app.ml.lora_registry import get_adapter_path
from app.ml.model_manager import ensure_lora_adapter, get_model_manager
from app.models.search import RewriteDetails, SearchResult
from app.search.graph_expander import GraphExpander

# Cache loaded indexes
_index_cache: dict[str, dict] = {}

RRF_K = 60  # Reciprocal Rank Fusion constant


def _get_indexes(repo_id: str) -> dict:
    if repo_id not in _index_cache:
        chunks = load_chunks(repo_id)
        bm25, corpus = load_bm25(repo_id)
        _index_cache[repo_id] = {
            "chunks": chunks,
            "bm25": bm25,
            "corpus": corpus,
        }
    return _index_cache[repo_id]


def _parse_rewrite_json(response: str) -> dict | None:
    """Try to extract and parse a JSON object from LLM response."""
    text = response.strip()
    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _rewrite_query(query: str) -> tuple[RewriteDetails | None, list[str], list[str]]:
    """Rewrite query using LLM (Qwen + optional LoRA).

    Uses the same structured JSON prompt as the LoRA training data.
    Returns (rewrite_details, search_queries_for_bm25, keywords).
    """
    manager = get_model_manager()

    prompt = (
        "Rewrite this search query into structured retrieval hints "
        "for a Java codebase.\n\n"
        f"Query: {query}\n\n"
        "Output JSON:"
    )

    try:
        response = manager.generate(prompt, max_new_tokens=256)
        parsed = _parse_rewrite_json(response)

        if parsed:
            details = RewriteDetails(
                intent=parsed.get("intent"),
                search_scope=parsed.get("search_scope"),
                keywords=parsed.get("keywords", []),
                project_terms=parsed.get("project_terms", []),
                method_hints=parsed.get("method_hints", []),
                api_hints=parsed.get("api_hints", []),
                search_queries=parsed.get("search_queries", []),
            )

            # Build BM25 queries from structured output
            bm25_queries = list(details.search_queries)

            # Add a combined query from method_hints + project_terms if useful
            if details.method_hints and details.project_terms:
                combined = " ".join(details.project_terms[:2] + details.method_hints[:2])
                if combined not in bm25_queries:
                    bm25_queries.append(combined)

            # Ensure at least the original query is included
            if not bm25_queries:
                bm25_queries = [query]

            keywords = details.keywords or _extract_keywords(query)
            return details, bm25_queries, keywords

    except Exception as e:
        print(f"[Rewriter] LLM rewrite error: {e}", flush=True)

    # Fallback: no structured rewriting
    return None, [query], _extract_keywords(query)


def _extract_keywords(query: str) -> list[str]:
    words = re.split(r'[\s,;]+', query)
    keywords = []
    for w in words:
        w = w.strip().strip('"\'()[]{}')
        if w and len(w) > 1:
            keywords.append(w)
    return keywords[:10]


def _rrf_fusion(
    bm25, queries: list[str], num_docs: int, top_n: int
) -> list[tuple[int, float]]:
    """Run BM25 for each query and fuse results with Reciprocal Rank Fusion."""
    rrf_scores: dict[int, float] = defaultdict(float)

    for q in queries:
        tokens = tokenize(q)
        if not tokens:
            continue
        scores = bm25.get_scores(tokens)
        # Rank documents by BM25 score (descending)
        ranked_indices = np.argsort(scores)[::-1]

        for rank, idx in enumerate(ranked_indices[:top_n * 3]):
            idx = int(idx)
            if scores[idx] <= 0:
                break
            rrf_scores[idx] += 1.0 / (RRF_K + rank + 1)

    # Sort by RRF score descending
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_n]


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

        # Ensure correct LoRA adapter is loaded for this repo
        adapter_path = get_adapter_path(self.repo_id)
        lora_active = adapter_path is not None
        if adapter_path:
            ensure_lora_adapter(str(adapter_path))

        # LLM structured query rewriting
        rewrite_details, bm25_queries, keywords = _rewrite_query(query)

        if rewrite_details:
            print(
                f"[Search] Original: {query!r} → "
                f"Queries: {rewrite_details.search_queries}, "
                f"Methods: {rewrite_details.method_hints}, "
                f"Terms: {rewrite_details.project_terms}",
                flush=True,
            )
        else:
            print(f"[Search] Original: {query!r} (no structured rewrite)", flush=True)

        # Multi-query BM25 with RRF fusion
        ranked_docs = _rrf_fusion(bm25, bm25_queries, len(chunks), settings.bm25_top_k)

        # Call graph for neighbor info
        graph_expander = GraphExpander(self.repo_id)

        # Build results
        results = []
        for rank, (idx, rrf_score) in enumerate(ranked_docs[:top_k]):
            chunk = chunks[idx]
            callers, callees = graph_expander.get_neighbors(chunk.chunk_id)
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=rrf_score,
                    bm25_rank=rank + 1,
                    callers=callers,
                    callees=callees,
                )
            )

        # Build rewritten_query string for backward compatibility
        rewritten_query = None
        if rewrite_details and rewrite_details.search_queries:
            rewritten_query = rewrite_details.search_queries[0]
            if rewritten_query.lower() == query.lower():
                rewritten_query = None

        return {
            "results": results,
            "expanded_keywords": keywords,
            "rewritten_query": rewritten_query,
            "rewrite_details": rewrite_details,
            "lora_active": lora_active,
        }
