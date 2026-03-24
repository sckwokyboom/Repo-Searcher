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


class WeightedFusion(BaseRetriever):
    """Weighted normalized score fusion: BM25-dominant since it outperforms vector."""
    name = "WeightedFusion"
    bm25_weight = 0.7
    vector_weight = 0.3

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        # BM25 scores
        tokenized_query = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query) if tokenized_query else np.zeros(len(self.chunks))

        # Normalize BM25 scores to [0, 1]
        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_norm = bm25_scores / bm25_max

        # FAISS scores
        manager = get_model_manager()
        query_vec = manager.encode_query(query)
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        n_chunks = min(len(self.chunks), self.faiss_index.ntotal)
        # Retrieve all-ish scores for proper normalization
        fetch_k = min(n_chunks, max(top_k * 5, 200))
        distances, indices = self.faiss_index.search(query_vec.astype("float32"), fetch_k)

        # Build vector score array
        vector_scores = np.zeros(len(self.chunks))
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                vector_scores[int(idx)] = float(dist)

        # Combine
        combined = self.bm25_weight * bm25_norm + self.vector_weight * vector_scores
        top_indices = np.argsort(combined)[::-1][:top_k]
        return [(self.chunks[int(i)], float(combined[i])) for i in top_indices if combined[i] > 0]


class BM25FileAgg(BaseRetriever):
    """BM25 with file-level score aggregation: sum chunk scores per file, rank files."""
    name = "BM25+FileAgg"

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)

        # Aggregate scores by file path
        file_scores: dict[str, float] = defaultdict(float)
        file_best_chunk: dict[str, tuple[int, float]] = {}  # file -> (chunk_idx, max_score)
        for i, score in enumerate(scores):
            if score > 0:
                fp = self.chunks[i].file_path
                file_scores[fp] += score
                if fp not in file_best_chunk or score > file_best_chunk[fp][1]:
                    file_best_chunk[fp] = (i, score)

        # Rank files by aggregated score, return best chunk per file
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for fp, agg_score in sorted_files:
            chunk_idx, _ = file_best_chunk[fp]
            results.append((self.chunks[chunk_idx], agg_score))
        return results


class HybridFileAgg(BaseRetriever):
    """Weighted BM25+Vector with file-level score aggregation."""
    name = "Hybrid+FileAgg"
    bm25_weight = 0.7
    vector_weight = 0.3

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        # BM25 scores
        tokenized_query = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query) if tokenized_query else np.zeros(len(self.chunks))
        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_norm = bm25_scores / bm25_max

        # Vector scores
        manager = get_model_manager()
        query_vec = manager.encode_query(query)
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        fetch_k = min(self.faiss_index.ntotal, max(top_k * 5, 200))
        distances, indices = self.faiss_index.search(query_vec.astype("float32"), fetch_k)
        vector_scores = np.zeros(len(self.chunks))
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                vector_scores[int(idx)] = float(dist)

        combined = self.bm25_weight * bm25_norm + self.vector_weight * vector_scores

        # File-level aggregation
        file_scores: dict[str, float] = defaultdict(float)
        file_best_chunk: dict[str, tuple[int, float]] = {}
        for i, score in enumerate(combined):
            if score > 0:
                fp = self.chunks[i].file_path
                file_scores[fp] += score
                if fp not in file_best_chunk or score > file_best_chunk[fp][1]:
                    file_best_chunk[fp] = (i, score)

        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for fp, agg_score in sorted_files:
            chunk_idx, _ = file_best_chunk[fp]
            results.append((self.chunks[chunk_idx], agg_score))
        return results


# --- Qwen3-Embedding based retrievers ---

_qwen3_model = None
_qwen3_tokenizer = None
_qwen3_faiss_cache: dict[str, object] = {}  # repo_id -> faiss index


def _get_qwen3():
    """Lazy-load Qwen3-Embedding-0.6B."""
    global _qwen3_model, _qwen3_tokenizer
    if _qwen3_model is None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        print(f"[Qwen3Emb] Loading {model_name}...", flush=True)
        _qwen3_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _qwen3_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        _qwen3_model.eval()
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _qwen3_model = _qwen3_model.to(device)
        print(f"[Qwen3Emb] Loaded on {device}", flush=True)
    return _qwen3_model, _qwen3_tokenizer


def _qwen3_encode(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """Encode texts with Qwen3-Embedding-0.6B. Returns L2-normalized embeddings."""
    import torch
    model, tokenizer = _get_qwen3()
    device = next(model.parameters()).device
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu().numpy().astype("float32"))
    return np.vstack(all_embeddings)


def _get_qwen3_faiss(repo_id: str, chunks: list) -> object:
    """Build/cache a FAISS index with Qwen3 embeddings for a repo."""
    import faiss
    if repo_id not in _qwen3_faiss_cache:
        cache_path = Path(__file__).parent / "results" / "qwen3_faiss" / f"{repo_id}.npy"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            print(f"[Qwen3Emb] Loading cached embeddings for {repo_id}", flush=True)
            embeddings = np.load(str(cache_path))
        else:
            print(f"[Qwen3Emb] Encoding {len(chunks)} chunks for {repo_id}...", flush=True)
            texts = [c.text_representation for c in chunks]
            embeddings = _qwen3_encode(texts, batch_size=16)
            np.save(str(cache_path), embeddings)
            print(f"[Qwen3Emb] Saved to {cache_path}", flush=True)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        _qwen3_faiss_cache[repo_id] = index
    return _qwen3_faiss_cache[repo_id]


class Qwen3VectorOnly(BaseRetriever):
    """Vector search using Qwen3-Embedding-0.6B — better NL-to-code matching."""
    name = "Qwen3Vector"

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        faiss_idx = _get_qwen3_faiss(self.repo_id, self.chunks)
        query_vec = _qwen3_encode([query])
        distances, indices = faiss_idx.search(query_vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                results.append((self.chunks[int(idx)], float(dist)))
        return results


class Qwen3HybridFileAgg(BaseRetriever):
    """BM25 + Qwen3-Embedding with file-level aggregation."""
    name = "Qwen3Hybrid+FileAgg"
    bm25_weight = 0.6
    vector_weight = 0.4

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        # BM25 scores (normalized)
        tokenized_query = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query) if tokenized_query else np.zeros(len(self.chunks))
        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_norm = bm25_scores / bm25_max

        # Qwen3 vector scores
        faiss_idx = _get_qwen3_faiss(self.repo_id, self.chunks)
        query_vec = _qwen3_encode([query])
        fetch_k = min(faiss_idx.ntotal, max(top_k * 5, 200))
        distances, indices = faiss_idx.search(query_vec, fetch_k)
        vector_scores = np.zeros(len(self.chunks))
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                vector_scores[int(idx)] = float(dist)

        combined = self.bm25_weight * bm25_norm + self.vector_weight * vector_scores

        # File-level aggregation
        file_scores: dict[str, float] = defaultdict(float)
        file_best_chunk: dict[str, tuple[int, float]] = {}
        for i, score in enumerate(combined):
            if score > 0:
                fp = self.chunks[i].file_path
                file_scores[fp] += score
                if fp not in file_best_chunk or score > file_best_chunk[fp][1]:
                    file_best_chunk[fp] = (i, score)

        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.chunks[file_best_chunk[fp][0]], agg_score) for fp, agg_score in sorted_files]


def _chunk_summary(chunk: CodeChunk, max_len: int = 200) -> str:
    """Compact one-line summary of a chunk for batch scoring prompt."""
    parts = []
    if chunk.file_path:
        # Just filename, not full path
        parts.append(chunk.file_path.rsplit("/", 1)[-1])
    if chunk.class_name:
        parts.append(chunk.class_name)
    if chunk.method_name:
        parts.append(chunk.method_name)
    if chunk.signature:
        parts.append(chunk.signature[:120])
    elif chunk.javadoc:
        parts.append(chunk.javadoc[:80])
    return " | ".join(parts)[:max_len]


class BM25LoRARerank(BaseRetriever):
    """BM25 retrieval + LoRA-finetuned Qwen batch reranking."""
    name = "BM25+LoRA_Rerank"

    LORA_PATH = str(Path(__file__).parent / "lora_training" / "output" / "scorer_lora" / "final")
    RERANK_TOP = 30

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        from app.ml.model_manager import get_model_manager

        # Get model manager (LoRA should be pre-initialized before benchmark)
        manager = get_model_manager()

        # Step 1: BM25 retrieval
        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:self.RERANK_TOP]
        bm25_candidates = [(self.chunks[int(i)], float(scores[i]))
                           for i in top_indices if scores[i] > 0]

        if not bm25_candidates:
            return []

        # Step 2: Batch LLM reranking — one prompt for all candidates
        return self._batch_rerank(query, bm25_candidates, manager, top_k)

    def _batch_rerank(
        self,
        query: str,
        candidates: list[tuple[CodeChunk, float]],
        manager,
        top_k: int,
    ) -> list[tuple[CodeChunk, float]]:
        import re as re_mod

        # Build batch prompt with numbered candidates + few-shot examples
        lines = []
        for i, (chunk, _) in enumerate(candidates):
            lines.append(f"[{i}] {_chunk_summary(chunk)}")
        candidates_text = "\n".join(lines)

        prompt = (
            "Rate each Java code snippet's relevance to the search query (0=irrelevant, 10=perfect match).\n\n"
            "Example:\n"
            "Query: Fix NPE in user authentication\n"
            "Candidates:\n"
            "[0] AuthService.java | AuthService | authenticate | public User authenticate(String token)\n"
            "[1] MathUtils.java | MathUtils | factorial | public static long factorial(int n)\n"
            "[2] AuthController.java | AuthController | login | public Response login(LoginRequest req)\n"
            "Scores: [0] 9, [1] 0, [2] 6\n\n"
            "Now rate:\n"
            f"Query: {query}\n"
            f"Candidates:\n{candidates_text}\n"
            "Scores:"
        )

        try:
            response = manager.generate(prompt, max_new_tokens=min(len(candidates) * 8, 300))
            # Parse "[i] score" or "[i] score," patterns
            score_map: dict[int, float] = {}
            for match in re_mod.finditer(r'\[(\d+)\]\s*(\d+(?:\.\d+)?)', response):
                idx = int(match.group(1))
                score = float(match.group(2))
                if 0 <= idx < len(candidates):
                    score_map[idx] = min(10.0, max(0.0, score))
        except Exception as e:
            print(f"[LoRA Rerank] Error: {e}", flush=True)
            score_map = {}

        # Build results: use LLM score if available, else BM25-normalized fallback
        bm25_max = max(s for _, s in candidates) if candidates else 1.0
        reranked = []
        for i, (chunk, bm25_score) in enumerate(candidates):
            if i in score_map:
                reranked.append((chunk, score_map[i]))
            else:
                # Fallback: normalize BM25 to 0-10 scale
                reranked.append((chunk, (bm25_score / bm25_max) * 5.0))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]


class BM25BaseQwenRerank(BaseRetriever):
    """BM25 retrieval + base Qwen (no LoRA) batch reranking. Control for LoRA comparison."""
    name = "BM25+Qwen_Rerank"

    RERANK_TOP = 30

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        from app.ml.model_manager import get_model_manager

        manager = get_model_manager()

        # BM25 retrieval
        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:self.RERANK_TOP]
        bm25_candidates = [(self.chunks[int(i)], float(scores[i]))
                           for i in top_indices if scores[i] > 0]

        if not bm25_candidates:
            return []

        # Reuse the same batch reranking logic
        lora_ret = BM25LoRARerank.__new__(BM25LoRARerank)
        return lora_ret._batch_rerank(query, bm25_candidates, manager, top_k)


class BM25LoRAFileAgg(BaseRetriever):
    """BM25 + LoRA batch reranking + file-level aggregation."""
    name = "BM25+LoRA+FileAgg"

    LORA_PATH = str(Path(__file__).parent / "lora_training" / "output" / "scorer_lora" / "final")
    RERANK_TOP = 30

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        lora_retriever = BM25LoRARerank(self.repo_id)
        lora_retriever._chunks = self._chunks
        lora_retriever._bm25 = self._bm25
        lora_retriever._corpus = self._corpus
        candidates = lora_retriever.retrieve(query, top_k=self.RERANK_TOP)

        # File-level aggregation
        file_scores: dict[str, float] = defaultdict(float)
        file_best_chunk: dict[str, tuple[CodeChunk, float]] = {}
        for chunk, score in candidates:
            fp = chunk.file_path
            file_scores[fp] += score
            if fp not in file_best_chunk or score > file_best_chunk[fp][1]:
                file_best_chunk[fp] = (chunk, score)

        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(file_best_chunk[fp][0], agg_score) for fp, agg_score in sorted_files]


RETRIEVER_REGISTRY: dict[str, type[BaseRetriever]] = {
    "bm25": BM25Only,
    "vector": VectorOnly,
    "rrf": HybridRRF,
    "rrf_rerank": HybridRRFRerank,
    "full": FullPipeline,
    "weighted": WeightedFusion,
    "bm25_fileagg": BM25FileAgg,
    "hybrid_fileagg": HybridFileAgg,
    "qwen3_vector": Qwen3VectorOnly,
    "qwen3_hybrid_fileagg": Qwen3HybridFileAgg,
    "bm25_lora_rerank": BM25LoRARerank,
    "bm25_qwen_rerank": BM25BaseQwenRerank,
    "bm25_lora_fileagg": BM25LoRAFileAgg,
}


def get_retriever(name: str, repo_id: str) -> BaseRetriever:
    cls = RETRIEVER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown retriever: {name}. Available: {list(RETRIEVER_REGISTRY.keys())}")
    return cls(repo_id)
