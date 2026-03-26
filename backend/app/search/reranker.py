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
    logger.info(f"Reranker: {total} candidates")

    for i, (chunk, rrf_score) in enumerate(candidates):
        context_parts = []
        if chunk.file_path:
            context_parts.append(f"File: {chunk.file_path}")
        if chunk.class_name:
            context_parts.append(f"Class: {chunk.class_name}")
        if chunk.method_name:
            context_parts.append(f"Method: {chunk.method_name}")
        if chunk.signature:
            context_parts.append(f"Signature: {chunk.signature}")
        if chunk.javadoc:
            context_parts.append(f"Doc: {chunk.javadoc[:150]}")
        code_context = "\n".join(context_parts)

        prompt = (
            "Is this Java code relevant to the search query? "
            "Answer with a single number from 0 (irrelevant) to 10 (perfect match).\n\n"
            f"Query: {query}\n\n"
            f"{code_context[:600]}\n\n"
            "Relevance score (0-10):"
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
            logger.error(f"Reranker: LLM error: {e}")
            score = rrf_score * 10

        logger.info(
            f"Reranker: {i+1}/{total}: {chunk.signature} ({rrf_score:.2f} -> {score:.2f})",
        )
        scored.append((chunk, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Reranker: sorted {len(scored)} candidates")
    return scored
