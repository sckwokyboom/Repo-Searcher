import re
import re as re_mod
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import faiss
import networkx as nx
import torch
from transformers import AutoModel, AutoTokenizer

from backend.app.config import settings
from backend.app.indexer.bm25_builder import tokenize
from backend.app.indexer.store import (
    load_bm25,
    load_call_graph,
    load_chunks,
    load_faiss,
)
from backend.app.ml.model_manager import get_model_manager
from backend.app.models.search import CodeChunk
from backend.app.search.hybrid_retriever import HybridRetriever
from backend.app.search.reranker import rerank


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
        pass

    def retrieve_timed(
        self, query: str, top_k: int = 20
    ) -> tuple[list[tuple[CodeChunk, float]], float]:
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
        return [
            (self.chunks[int(i)], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]


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
        tokenized_query = tokenize(query)
        bm25_scores = (
            self.bm25.get_scores(tokenized_query) if tokenized_query else np.array([])
        )
        bm25_top = (
            np.argsort(bm25_scores)[::-1][: settings.bm25_top_k]
            if len(bm25_scores) > 0
            else []
        )

        manager = get_model_manager()
        query_vec = manager.encode_query(query)
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        distances, faiss_indices = self.faiss_index.search(
            query_vec.astype("float32"), settings.faiss_top_k
        )

        rrf_scores: dict[int, float] = defaultdict(float)
        for rank, idx in enumerate(bm25_top):
            rrf_scores[int(idx)] += 1.0 / (settings.rrf_k + rank + 1)
        for rank, idx in enumerate(faiss_indices[0]):
            if idx >= 0:
                rrf_scores[int(idx)] += 1.0 / (settings.rrf_k + rank + 1)

        sorted_candidates = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [(self.chunks[idx], score) for idx, score in sorted_candidates]


class HybridRRFRerank(BaseRetriever):
    name = "HybridRRF+Rerank"

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:

        rrf = HybridRRF(self.repo_id)
        rrf._chunks = self._chunks
        rrf._bm25 = self._bm25
        rrf._corpus = self._corpus
        rrf._faiss = self._faiss
        candidates = rrf.retrieve(query, top_k=max(top_k, settings.rrf_top_k))

        reranked = rerank(query, candidates[: settings.rrf_top_k])
        reranked_ids = {c.chunk_id for c, _ in reranked}
        remaining = [(c, s) for c, s in candidates if c.chunk_id not in reranked_ids]
        return (reranked + remaining)[:top_k]


class FullPipeline(BaseRetriever):
    name = "FullPipeline"

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:

        retriever = HybridRetriever(self.repo_id)
        result = retriever.search_sync(query, top_k=top_k)
        return [(r.chunk, r.score) for r in result["results"]]


class WeightedFusion(BaseRetriever):
    name = "WeightedFusion"
    bm25_weight = 0.7
    vector_weight = 0.3

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        tokenized_query = tokenize(query)
        bm25_scores = (
            self.bm25.get_scores(tokenized_query)
            if tokenized_query
            else np.zeros(len(self.chunks))
        )

        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_norm = bm25_scores / bm25_max

        manager = get_model_manager()
        query_vec = manager.encode_query(query)
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        n_chunks = min(len(self.chunks), self.faiss_index.ntotal)
        fetch_k = min(n_chunks, max(top_k * 5, 200))
        distances, indices = self.faiss_index.search(
            query_vec.astype("float32"), fetch_k
        )

        vector_scores = np.zeros(len(self.chunks))
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                vector_scores[int(idx)] = float(dist)

        combined = self.bm25_weight * bm25_norm + self.vector_weight * vector_scores
        top_indices = np.argsort(combined)[::-1][:top_k]
        return [
            (self.chunks[int(i)], float(combined[i]))
            for i in top_indices
            if combined[i] > 0
        ]


class BM25FileAgg(BaseRetriever):
    name = "BM25+FileAgg"

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)

        file_scores: dict[str, float] = defaultdict(float)
        file_best_chunk: dict[str, tuple[int, float]] = {}
        for i, score in enumerate(scores):
            if score > 0:
                fp = self.chunks[i].file_path
                file_scores[fp] += score
                if fp not in file_best_chunk or score > file_best_chunk[fp][1]:
                    file_best_chunk[fp] = (i, score)

        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
        results = []
        for fp, agg_score in sorted_files:
            chunk_idx, _ = file_best_chunk[fp]
            results.append((self.chunks[chunk_idx], agg_score))
        return results


class HybridFileAgg(BaseRetriever):
    name = "Hybrid+FileAgg"
    bm25_weight = 0.7
    vector_weight = 0.3

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        tokenized_query = tokenize(query)
        bm25_scores = (
            self.bm25.get_scores(tokenized_query)
            if tokenized_query
            else np.zeros(len(self.chunks))
        )
        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_norm = bm25_scores / bm25_max

        manager = get_model_manager()
        query_vec = manager.encode_query(query)
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        fetch_k = min(self.faiss_index.ntotal, max(top_k * 5, 200))
        distances, indices = self.faiss_index.search(
            query_vec.astype("float32"), fetch_k
        )
        vector_scores = np.zeros(len(self.chunks))
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                vector_scores[int(idx)] = float(dist)

        combined = self.bm25_weight * bm25_norm + self.vector_weight * vector_scores

        file_scores: dict[str, float] = defaultdict(float)
        file_best_chunk: dict[str, tuple[int, float]] = {}
        for i, score in enumerate(combined):
            if score > 0:
                fp = self.chunks[i].file_path
                file_scores[fp] += score
                if fp not in file_best_chunk or score > file_best_chunk[fp][1]:
                    file_best_chunk[fp] = (i, score)

        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
        results = []
        for fp, agg_score in sorted_files:
            chunk_idx, _ = file_best_chunk[fp]
            results.append((self.chunks[chunk_idx], agg_score))
        return results


_qwen3_model = None
_qwen3_tokenizer = None
_qwen3_faiss_cache: dict[str, object] = {}


def _get_qwen3():
    global _qwen3_model, _qwen3_tokenizer
    if _qwen3_model is None:

        model_name = "Qwen/Qwen3-Embedding-0.6B"
        print(f"[Qwen3Emb] Loading {model_name}...", flush=True)
        _qwen3_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        _qwen3_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        _qwen3_model.eval()
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _qwen3_model = _qwen3_model.to(device)
        print(f"[Qwen3Emb] Loaded on {device}", flush=True)
    return _qwen3_model, _qwen3_tokenizer


def _qwen3_encode(texts: list[str], batch_size: int = 16) -> np.ndarray:

    model, tokenizer = _get_qwen3()
    device = next(model.parameters()).device
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu().numpy().astype("float32"))
    return np.vstack(all_embeddings)


def _get_qwen3_faiss(repo_id: str, chunks: list) -> object:

    if repo_id not in _qwen3_faiss_cache:
        cache_path = (
            Path(__file__).parent / "results" / "qwen3_faiss" / f"{repo_id}.npy"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            print(f"[Qwen3Emb] Loading cached embeddings for {repo_id}", flush=True)
            embeddings = np.load(str(cache_path))
        else:
            print(
                f"[Qwen3Emb] Encoding {len(chunks)} chunks for {repo_id}...", flush=True
            )
            texts = [c.text_representation for c in chunks]
            embeddings = _qwen3_encode(texts, batch_size=16)
            np.save(str(cache_path), embeddings)
            print(f"[Qwen3Emb] Saved to {cache_path}", flush=True)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        _qwen3_faiss_cache[repo_id] = index
    return _qwen3_faiss_cache[repo_id]


class Qwen3VectorOnly(BaseRetriever):
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
    name = "Qwen3Hybrid+FileAgg"
    bm25_weight = 0.6
    vector_weight = 0.4

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        tokenized_query = tokenize(query)
        bm25_scores = (
            self.bm25.get_scores(tokenized_query)
            if tokenized_query
            else np.zeros(len(self.chunks))
        )
        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_norm = bm25_scores / bm25_max

        faiss_idx = _get_qwen3_faiss(self.repo_id, self.chunks)
        query_vec = _qwen3_encode([query])
        fetch_k = min(faiss_idx.ntotal, max(top_k * 5, 200))
        distances, indices = faiss_idx.search(query_vec, fetch_k)
        vector_scores = np.zeros(len(self.chunks))
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                vector_scores[int(idx)] = float(dist)

        combined = self.bm25_weight * bm25_norm + self.vector_weight * vector_scores

        file_scores: dict[str, float] = defaultdict(float)
        file_best_chunk: dict[str, tuple[int, float]] = {}
        for i, score in enumerate(combined):
            if score > 0:
                fp = self.chunks[i].file_path
                file_scores[fp] += score
                if fp not in file_best_chunk or score > file_best_chunk[fp][1]:
                    file_best_chunk[fp] = (i, score)

        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
        return [
            (self.chunks[file_best_chunk[fp][0]], agg_score)
            for fp, agg_score in sorted_files
        ]


def _chunk_summary(chunk: CodeChunk, max_len: int = 200) -> str:
    parts = []
    if chunk.file_path:
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
    name = "BM25+LoRA_Rerank"

    LORA_PATH = str(
        Path(__file__).parent / "lora_training" / "output" / "scorer_lora" / "final"
    )
    RERANK_TOP = 30

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        manager = get_model_manager()

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][: self.RERANK_TOP]
        bm25_candidates = [
            (self.chunks[int(i)], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

        if not bm25_candidates:
            return []

        return self._batch_rerank(query, bm25_candidates, manager, top_k)

    def _batch_rerank(
        self,
        query: str,
        candidates: list[tuple[CodeChunk, float]],
        manager,
        top_k: int,
    ) -> list[tuple[CodeChunk, float]]:

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
            response = manager.generate(
                prompt, max_new_tokens=min(len(candidates) * 8, 300)
            )
            score_map: dict[int, float] = {}
            for match in re_mod.finditer(r"\[(\d+)\]\s*(\d+(?:\.\d+)?)", response):
                idx = int(match.group(1))
                score = float(match.group(2))
                if 0 <= idx < len(candidates):
                    score_map[idx] = min(10.0, max(0.0, score))
        except Exception as e:
            print(f"[LoRA Rerank] Error: {e}", flush=True)
            score_map = {}

        bm25_max = max(s for _, s in candidates) if candidates else 1.0
        reranked = []
        for i, (chunk, bm25_score) in enumerate(candidates):
            if i in score_map:
                reranked.append((chunk, score_map[i]))
            else:
                reranked.append((chunk, (bm25_score / bm25_max) * 5.0))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]


class BM25BaseQwenRerank(BaseRetriever):
    name = "BM25+Qwen_Rerank"

    RERANK_TOP = 30

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:

        manager = get_model_manager()

        # BM25 retrieval
        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][: self.RERANK_TOP]
        bm25_candidates = [
            (self.chunks[int(i)], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

        if not bm25_candidates:
            return []

        lora_ret = BM25LoRARerank.__new__(BM25LoRARerank)
        return lora_ret._batch_rerank(query, bm25_candidates, manager, top_k)


class BM25LoRAFileAgg(BaseRetriever):
    name = "BM25+LoRA+FileAgg"

    LORA_PATH = str(
        Path(__file__).parent / "lora_training" / "output" / "scorer_lora" / "final"
    )
    RERANK_TOP = 30

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        lora_retriever = BM25LoRARerank(self.repo_id)
        lora_retriever._chunks = self._chunks
        lora_retriever._bm25 = self._bm25
        lora_retriever._corpus = self._corpus
        candidates = lora_retriever.retrieve(query, top_k=self.RERANK_TOP)

        file_scores: dict[str, float] = defaultdict(float)
        file_best_chunk: dict[str, tuple[CodeChunk, float]] = {}
        for chunk, score in candidates:
            fp = chunk.file_path
            file_scores[fp] += score
            if fp not in file_best_chunk or score > file_best_chunk[fp][1]:
                file_best_chunk[fp] = (chunk, score)

        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
        return [(file_best_chunk[fp][0], agg_score) for fp, agg_score in sorted_files]


def _build_file_chunk_index(chunks: list[CodeChunk]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for i, chunk in enumerate(chunks):
        index[chunk.file_path].append(i)
    return index


def _build_chunk_id_index(chunks: list[CodeChunk]) -> dict[str, int]:
    return {chunk.chunk_id: i for i, chunk in enumerate(chunks)}


_call_graph_cache: dict[str, nx.DiGraph] = {}


def _get_call_graph(repo_id: str) -> nx.DiGraph:
    if repo_id not in _call_graph_cache:
        _call_graph_cache[repo_id] = load_call_graph(repo_id)
    return _call_graph_cache[repo_id]


def compute_prior_score(
    query: str,
    candidate_chunk: CodeChunk,
    seed_file_path: str,
) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0

    id_text = (
        (candidate_chunk.file_path or "")
        + " "
        + (candidate_chunk.class_name or "")
        + " "
        + (candidate_chunk.method_name or "")
    )
    id_tokens = set(tokenize(id_text))
    union = query_tokens | id_tokens
    identifier_overlap = len(query_tokens & id_tokens) / max(1, len(union))

    synopsis = (candidate_chunk.text_representation or "")[:200]
    synopsis_tokens = set(tokenize(synopsis))
    token_overlap = len(query_tokens & synopsis_tokens) / max(1, len(query_tokens))

    seed_dir = seed_file_path.rsplit("/", 1)[0] if "/" in seed_file_path else ""
    cand_dir = (
        candidate_chunk.file_path.rsplit("/", 1)[0]
        if "/" in (candidate_chunk.file_path or "")
        else ""
    )
    package_match = 1.0 if seed_dir and cand_dir and seed_dir == cand_dir else 0.0

    test_penalty = 0.0
    if re.search(
        r"(?i)(Test|Coverage|Internal|Mock|Stub)", candidate_chunk.file_path or ""
    ):
        if "test" not in query.lower():
            test_penalty = 1.0

    return (
        2.0 * identifier_overlap
        + 1.0 * token_overlap
        + 0.5 * package_match
        - 1.0 * test_penalty
    )


class SafeGraphExpansionV2(BaseRetriever):
    name = "BM25+SafeGraphV2"

    FREEZE_K = 5
    SEED_K = 5
    RAW_NEIGHBORS_LIMIT = 10
    PRIOR_TOP_N = 5
    GRAPH_WEIGHT = 0.5
    ADDITIVE_ALPHA = 0.1
    HUB_DEGREE_LIMIT = 50
    EDGE_DIRECTION = "both"  # "outgoing", "incoming", "both"

    def __init__(self, repo_id: str, edge_direction: str = "both"):
        super().__init__(repo_id)
        self._call_graph = None
        self.EDGE_DIRECTION = edge_direction

    @property
    def call_graph(self) -> nx.DiGraph:
        if self._call_graph is None:
            self._call_graph = _get_call_graph(self.repo_id)
        return self._call_graph

    def _get_graph_neighbors(
        self, chunk_id: str, graph: nx.DiGraph
    ) -> list[tuple[str, str]]:
        neighbors = []
        if self.EDGE_DIRECTION in ("outgoing", "both"):
            for nid in graph.successors(chunk_id):
                neighbors.append((nid, "outgoing"))
        if self.EDGE_DIRECTION in ("incoming", "both"):
            for nid in graph.predecessors(chunk_id):
                neighbors.append((nid, "incoming"))
        return neighbors

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[CodeChunk, float]]:
        results, _ = self.retrieve_with_diagnostics(query, top_k)
        return results

    def retrieve_with_diagnostics(
        self,
        query: str,
        top_k: int = 10,
    ) -> tuple[list[tuple[CodeChunk, float]], dict]:
        diagnostics = {
            "baseline_top10": [],
            "frozen_top5": [],
            "raw_neighbors": [],
            "filtered_neighbors": [],
            "graph_candidates": [],
        }

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return [], diagnostics

        bm25_scores = self.bm25.get_scores(tokenized_query)

        file_agg: dict[str, float] = defaultdict(float)
        file_best_chunk: dict[str, int] = {}
        for i, score in enumerate(bm25_scores):
            if score > 0:
                fp = self.chunks[i].file_path
                file_agg[fp] += score
                if (
                    fp not in file_best_chunk
                    or score > bm25_scores[file_best_chunk[fp]]
                ):
                    file_best_chunk[fp] = i

        if not file_agg:
            return [], diagnostics

        max_file_score = max(file_agg.values())
        baseline_norm: dict[str, float] = {
            fp: s / max_file_score for fp, s in file_agg.items()
        }

        baseline_ranked = sorted(
            baseline_norm.items(), key=lambda x: x[1], reverse=True
        )

        # Freeze top-5
        frozen_top5 = baseline_ranked[: self.FREEZE_K]
        frozen_fps = {fp for fp, _ in frozen_top5}
        seed_files = baseline_ranked[: self.SEED_K]

        diagnostics["baseline_top10"] = [(fp, sc) for fp, sc in baseline_ranked[:10]]
        diagnostics["frozen_top5"] = [(fp, sc) for fp, sc in frozen_top5]

        file_chunk_idx = _build_file_chunk_index(self.chunks)
        chunk_id_idx = _build_chunk_id_index(self.chunks)

        try:
            graph = self.call_graph
        except Exception:
            results = []
            for fp, score in baseline_ranked[:top_k]:
                ci = file_best_chunk.get(fp)
                if ci is not None:
                    results.append((self.chunks[ci], score))
            return results, diagnostics

        raw_neighbors: list[
            dict
        ] = []  # {seed_fp, candidate_fp, candidate_chunk, direction, seed_norm}
        for seed_fp, seed_norm_score in seed_files:
            neighbor_count = 0
            seen_fps_this_seed: set[str] = set()
            for ci in file_chunk_idx.get(seed_fp, []):
                if neighbor_count >= self.RAW_NEIGHBORS_LIMIT:
                    break
                chunk_id = self.chunks[ci].chunk_id
                if chunk_id not in graph:
                    continue
                if graph.degree(chunk_id) > self.HUB_DEGREE_LIMIT:
                    continue
                for nid, direction in self._get_graph_neighbors(chunk_id, graph):
                    if neighbor_count >= self.RAW_NEIGHBORS_LIMIT:
                        break
                    ni = chunk_id_idx.get(nid)
                    if ni is None:
                        continue
                    nfp = self.chunks[ni].file_path
                    if nfp in frozen_fps or nfp == seed_fp:
                        continue
                    if nfp in seen_fps_this_seed:
                        continue
                    seen_fps_this_seed.add(nfp)
                    raw_neighbors.append(
                        {
                            "seed_fp": seed_fp,
                            "seed_norm": seed_norm_score,
                            "candidate_fp": nfp,
                            "candidate_chunk": self.chunks[ni],
                            "direction": direction,
                        }
                    )
                    neighbor_count += 1

        diagnostics["raw_neighbors"] = [
            {
                "seed": n["seed_fp"],
                "candidate": n["candidate_fp"],
                "direction": n["direction"],
            }
            for n in raw_neighbors
        ]

        candidate_map: dict[str, dict] = {}
        for n in raw_neighbors:
            prior = compute_prior_score(query, n["candidate_chunk"], n["seed_fp"])
            n["prior_score"] = prior
            fp = n["candidate_fp"]
            if fp not in candidate_map or prior > candidate_map[fp]["prior_score"]:
                candidate_map[fp] = n

        sorted_candidates = sorted(
            candidate_map.values(), key=lambda x: x["prior_score"], reverse=True
        )
        filtered = sorted_candidates[: self.PRIOR_TOP_N]

        diagnostics["filtered_neighbors"] = [
            {
                "candidate": c["candidate_fp"],
                "prior": c["prior_score"],
                "seed": c["seed_fp"],
                "direction": c["direction"],
            }
            for c in filtered
        ]

        max_prior = max((c["prior_score"] for c in filtered), default=1.0)
        if max_prior <= 0:
            max_prior = 1.0

        graph_scores: dict[str, float] = {}
        for c in filtered:
            prior_norm = max(0.0, c["prior_score"]) / max_prior
            gs = c["seed_norm"] * prior_norm * self.GRAPH_WEIGHT
            fp = c["candidate_fp"]
            graph_scores[fp] = max(graph_scores.get(fp, 0.0), gs)

        diagnostics["graph_candidates"] = [
            {"candidate": fp, "graph_score": gs} for fp, gs in graph_scores.items()
        ]

        remaining_scores: dict[str, float] = {}
        for fp, bl in baseline_norm.items():
            if fp in frozen_fps:
                continue
            gs = graph_scores.get(fp, 0.0)
            remaining_scores[fp] = bl + self.ADDITIVE_ALPHA * gs

        for fp, gs in graph_scores.items():
            if fp not in remaining_scores and fp not in frozen_fps:
                remaining_scores[fp] = gs

        remaining_ranked = sorted(
            remaining_scores.items(), key=lambda x: x[1], reverse=True
        )
        slots_left = top_k - len(frozen_top5)

        results = []
        for fp, score in frozen_top5:
            ci = file_best_chunk.get(fp)
            if ci is not None:
                results.append((self.chunks[ci], score))

        for fp, score in remaining_ranked[:slots_left]:
            ci = file_best_chunk.get(fp)
            if ci is not None:
                results.append((self.chunks[ci], score))
            else:
                idxs = file_chunk_idx.get(fp, [])
                if idxs:
                    results.append((self.chunks[idxs[0]], score))

        return results, diagnostics


class BM25GraphExpansion(BaseRetriever):
    name = "BM25+GraphExpand"

    SEED_K = 5
    NEIGHBORS_PER_SEED = 5
    EDGE_WEIGHT = 1.0
    BOOST = 0.2
    HUB_DEGREE_LIMIT = 50

    def __init__(self, repo_id: str):
        super().__init__(repo_id)
        self._call_graph = None

    @property
    def call_graph(self) -> nx.DiGraph:
        if self._call_graph is None:
            self._call_graph = _get_call_graph(self.repo_id)
        return self._call_graph

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[CodeChunk, float]]:
        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []
        bm25_scores = self.bm25.get_scores(tokenized_query)

        file_agg: dict[str, float] = defaultdict(float)
        file_best_chunk: dict[str, int] = {}
        for i, score in enumerate(bm25_scores):
            if score > 0:
                fp = self.chunks[i].file_path
                file_agg[fp] += score
                if (
                    fp not in file_best_chunk
                    or score > bm25_scores[file_best_chunk[fp]]
                ):
                    file_best_chunk[fp] = i

        max_file_score = max(file_agg.values()) if file_agg else 1.0
        baseline_norm: dict[str, float] = {
            fp: s / max_file_score for fp, s in file_agg.items()
        }

        baseline_ranked = sorted(
            baseline_norm.items(), key=lambda x: x[1], reverse=True
        )
        seed_files = baseline_ranked[: self.SEED_K]

        file_chunk_idx = _build_file_chunk_index(self.chunks)
        chunk_id_idx = _build_chunk_id_index(self.chunks)
        graph = self.call_graph

        graph_scores: dict[str, float] = {}
        for seed_fp, seed_norm_score in seed_files:
            neighbor_count = 0
            for ci in file_chunk_idx.get(seed_fp, []):
                chunk_id = self.chunks[ci].chunk_id
                if chunk_id not in graph:
                    continue
                if graph.degree(chunk_id) > self.HUB_DEGREE_LIMIT:
                    continue
                for nid in list(graph.predecessors(chunk_id)) + list(
                    graph.successors(chunk_id)
                ):
                    ni = chunk_id_idx.get(nid)
                    if ni is None:
                        continue
                    nfp = self.chunks[ni].file_path
                    gs = seed_norm_score * self.EDGE_WEIGHT
                    if nfp not in graph_scores or gs > graph_scores[nfp]:
                        graph_scores[nfp] = gs
                    neighbor_count += 1
                    if neighbor_count >= self.NEIGHBORS_PER_SEED:
                        break
                if neighbor_count >= self.NEIGHBORS_PER_SEED:
                    break

        final_scores: dict[str, float] = {}
        all_fps = set(baseline_norm.keys()) | set(graph_scores.keys())
        for fp in all_fps:
            bl = baseline_norm.get(fp, 0.0)
            gs = graph_scores.get(fp, 0.0)
            if bl > 0 and gs > 0:
                final_scores[fp] = bl + self.BOOST * gs
            elif bl > 0:
                final_scores[fp] = bl
            else:
                final_scores[fp] = gs

        sorted_files = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
        results = []
        for fp, score in sorted_files:
            ci = file_best_chunk.get(fp)
            if ci is not None:
                results.append((self.chunks[ci], score))
            else:
                idxs = file_chunk_idx.get(fp, [])
                if idxs:
                    results.append((self.chunks[idxs[0]], score))
        return results


class BM25PrioritizedExpansion(BaseRetriever):
    name = "BM25+PriorExpand"

    SEED_K = 5
    MAX_TOTAL_EXPANDED = 15
    GRAPH_BONUS = 0.3
    HUB_DEGREE_LIMIT = 50

    def __init__(self, repo_id: str):
        super().__init__(repo_id)
        self._call_graph = None

    @property
    def call_graph(self) -> nx.DiGraph:
        if self._call_graph is None:
            self._call_graph = _get_call_graph(self.repo_id)
        return self._call_graph

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[CodeChunk, float]]:
        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []
        bm25_scores = self.bm25.get_scores(tokenized_query)

        file_agg: dict[str, float] = defaultdict(float)
        file_best_chunk: dict[str, int] = {}
        for i, score in enumerate(bm25_scores):
            if score > 0:
                fp = self.chunks[i].file_path
                file_agg[fp] += score
                if (
                    fp not in file_best_chunk
                    or score > bm25_scores[file_best_chunk[fp]]
                ):
                    file_best_chunk[fp] = i

        max_file_score = max(file_agg.values()) if file_agg else 1.0
        baseline_norm: dict[str, float] = {
            fp: s / max_file_score for fp, s in file_agg.items()
        }

        baseline_ranked = sorted(
            baseline_norm.items(), key=lambda x: x[1], reverse=True
        )
        seed_files = baseline_ranked[: self.SEED_K]

        file_chunk_idx = _build_file_chunk_index(self.chunks)
        chunk_id_idx = _build_chunk_id_index(self.chunks)
        graph = self.call_graph

        graph_bonus: dict[str, float] = {}
        new_count = 0

        for seed_fp, seed_norm in seed_files:
            if new_count >= self.MAX_TOTAL_EXPANDED:
                break
            for ci in file_chunk_idx.get(seed_fp, []):
                if new_count >= self.MAX_TOTAL_EXPANDED:
                    break
                chunk_id = self.chunks[ci].chunk_id
                if chunk_id not in graph:
                    continue
                if graph.degree(chunk_id) > self.HUB_DEGREE_LIMIT:
                    continue
                for nid in list(graph.predecessors(chunk_id)) + list(
                    graph.successors(chunk_id)
                ):
                    if new_count >= self.MAX_TOTAL_EXPANDED:
                        break
                    ni = chunk_id_idx.get(nid)
                    if ni is None:
                        continue
                    nfp = self.chunks[ni].file_path
                    bonus = seed_norm * self.GRAPH_BONUS
                    if nfp not in graph_bonus:
                        new_count += 1
                    if nfp not in graph_bonus or bonus > graph_bonus[nfp]:
                        graph_bonus[nfp] = bonus

        final_scores: dict[str, float] = {}
        all_fps = set(baseline_norm.keys()) | set(graph_bonus.keys())
        for fp in all_fps:
            bl = baseline_norm.get(fp, 0.0)
            gb = graph_bonus.get(fp, 0.0)
            final_scores[fp] = bl + gb

        sorted_files = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
        results = []
        for fp, score in sorted_files:
            ci = file_best_chunk.get(fp)
            if ci is not None:
                results.append((self.chunks[ci], score))
            else:
                idxs = file_chunk_idx.get(fp, [])
                if idxs:
                    results.append((self.chunks[idxs[0]], score))
        return results


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
    "bm25_graph_expand": BM25GraphExpansion,
    "bm25_prior_expand": BM25PrioritizedExpansion,
    "bm25_safe_graph_v2": SafeGraphExpansionV2,
}


def get_retriever(name: str, repo_id: str) -> BaseRetriever:
    cls = RETRIEVER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown retriever: {name}. Available: {list(RETRIEVER_REGISTRY.keys())}"
        )
    return cls(repo_id)
