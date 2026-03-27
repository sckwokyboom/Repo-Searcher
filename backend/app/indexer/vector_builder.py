from typing import Awaitable, Callable

import faiss
import numpy as np

from app.config import settings
from app.ml.model_manager import get_model_manager
from app.models.search import CodeChunk


async def build_vector_index(
    chunks: list[CodeChunk],
    progress_callback: Callable[[float], Awaitable[None]] | None = None,
) -> faiss.IndexFlatIP:
    manager = get_model_manager()
    texts = [chunk.text_representation for chunk in chunks]

    all_embeddings = []
    batch_size = settings.embedding_batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = manager.encode_code(batch)
        all_embeddings.append(embeddings)

        if progress_callback:
            await progress_callback(min(1.0, (i + batch_size) / len(texts)))

    vectors = np.vstack(all_embeddings).astype("float32")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

    index = faiss.IndexFlatIP(settings.embedding_dim)
    index.add(vectors)

    return index
