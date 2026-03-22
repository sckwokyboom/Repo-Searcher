import logging
import re

from app.ml.model_manager import get_model_manager
from app.models.search import CodeChunk

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    candidates: list[tuple[CodeChunk, float]],
) -> list[tuple[CodeChunk, float]]:
    if not candidates:
        return []

    manager = get_model_manager()
    scored = []
    total = len(candidates)
    print(f"[Reranker] Starting reranking of {total} candidates for: {query[:50]}...", flush=True)

    for i, (chunk, rrf_score) in enumerate(candidates):
        # Use signature + short body for faster inference
        code_snippet = chunk.signature
        if chunk.javadoc:
            code_snippet = chunk.javadoc[:200] + "\n" + code_snippet

        prompt = (
            "Rate relevance 0-10.\n"
            f"Query: {query}\n"
            f"Code: {code_snippet[:500]}\n"
            "Score:"
        )

        try:
            response = manager.generate(prompt, max_new_tokens=5)
            match = re.search(r'(\d+(?:\.\d+)?)', response)
            if match:
                score = float(match.group(1))
                score = min(10.0, max(0.0, score))
            else:
                score = rrf_score * 10
        except Exception as e:
            print(f"[Reranker] Error for {chunk.chunk_id}: {e}", flush=True)
            score = rrf_score * 10

        print(f"[Reranker] {i+1}/{total}: {chunk.chunk_id[:50]} -> {score:.1f}", flush=True)
        scored.append((chunk, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    print(f"[Reranker] Done!", flush=True)
    return scored
