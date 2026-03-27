import logging
import math
import re

import numpy as np

from app.config import settings
from app.indexer.bm25_builder import tokenize
from app.ml.model_manager import get_model_manager
from app.models.search import CodeChunk

logger = logging.getLogger(__name__)


class _Node:
    __slots__ = (
        "id",
        "query",
        "parent_id",
        "children_ids",
        "visits",
        "total_reward",
        "top_hits",
        "bm25_reward",
        "semantic_reward",
        "llm_reward",
    )

    def __init__(self, node_id: int, query: str, parent_id: int | None = None):
        self.id = node_id
        self.query = query
        self.parent_id = parent_id
        self.children_ids: list[int] = []
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.top_hits: list[dict] = []
        self.bm25_reward: float = 0.0
        self.semantic_reward: float = 0.0
        self.llm_reward: float = 0.0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0


class MCTSRewriter:
    C = 1.41

    W_BM25 = 0.25
    W_SEMANTIC = 0.50
    W_LLM = 0.25

    def __init__(
        self,
        bm25,
        corpus: list[list[str]],
        chunks: list[CodeChunk],
        faiss_index=None,
        n_iterations: int | None = None,
        n_children: int | None = None,
    ):
        self.bm25 = bm25
        self.corpus = corpus
        self.chunks = chunks
        self.faiss_index = faiss_index
        self.n_iterations = n_iterations or settings.mcts_iterations
        self.n_children = n_children or settings.mcts_children
        self.nodes: list[_Node] = []
        self._manager = get_model_manager()

    def rewrite(self, query: str) -> dict:
        logger.info(f"MCTS: rewriting for <{query!r}>...")
        logger.info(
            f"MCTS: weights: BM25={self.W_BM25}, Semantic={self.W_SEMANTIC}, LLM={self.W_LLM}"
        )

        root = self._add_node(query, parent_id=None)
        root_reward, root_hits, components = self._simulate(root.query)
        root.top_hits = root_hits
        root.bm25_reward = components["bm25"]
        root.semantic_reward = components["semantic"]
        root.llm_reward = components["llm"]
        self._backpropagate(root.id, root_reward)
        logger.info(
            f"MCTS: root node: reward={root_reward:.3f} "
            f"(bm25={components['bm25']:.3f}, sem={components['semantic']:.3f}, llm={components['llm']:.3f}), "
            f"hits={[h['name'] for h in root_hits]}",
        )

        for iteration in range(self.n_iterations):
            logger.info(f"MCTS: iteration {iteration + 1}/{self.n_iterations}")

            leaf = self._select(root.id)
            children = self._expand(leaf, query)
            logger.info(f"MCTS: expanded {len(children)} children for {leaf.query!r}")

            for child in children:
                reward, hits, components = self._simulate(child.query)
                child.top_hits = hits
                child.bm25_reward = components["bm25"]
                child.semantic_reward = components["semantic"]
                child.llm_reward = components["llm"]
                self._backpropagate(child.id, reward)
                logger.info(
                    f"MCTS: node {child.id}: reward={reward:.3f} "
                    f"(bm25={components['bm25']:.3f}, sem={components['semantic']:.3f}, llm={components['llm']:.3f}), "
                    f"hits={[h['name'] for h in hits]}",
                )

        best_leaf = self._best_leaf()
        best_path = self._path_to_root(best_leaf.id)
        keywords = self._extract_keywords(best_leaf.query)

        logger.info(f"MCTS: best leaf: {best_leaf.query!r}")
        logger.info(f"MCTS: best path: {[n for n in best_path]}")
        logger.info(f"MCTS: best keywords: {keywords}")

        return {
            "trace": self._build_trace(best_path, best_leaf.query, query),
            "best_query": best_leaf.query,
            "keywords": keywords,
        }

    def _select(self, node_id: int) -> _Node:
        node = self.nodes[node_id]
        while not node.is_leaf():
            best_child_id = max(
                node.children_ids,
                key=lambda cid: self._ucb1(cid, node.visits),
            )
            node = self.nodes[best_child_id]
        return node

    def _ucb1(self, node_id: int, parent_visits: int) -> float:
        node = self.nodes[node_id]
        if node.visits == 0:
            return float("inf")
        return node.avg_reward + self.C * math.sqrt(
            math.log(parent_visits) / node.visits
        )

    def _expand(self, node: _Node, original_query: str) -> list[_Node]:
        prompt = (
            "You are a code search query optimizer for Java repositories. "
            "The user's query may be in any language (English, Russian, Chinese, etc). "
            "Your task: rewrite it into "
            f"{self.n_children} different search queries that will find the most relevant Java source code. "
            "Each rewrite MUST:\n"
            "- Be in English (since Java code uses English identifiers)\n"
            "- Use specific Java class names, method names, design patterns, or API terms\n"
            "- Target different aspects or synonyms of the original intent\n\n"
            f'User\'s original query: "{original_query}"\n'
        )
        if node.query != original_query:
            prompt += f'Current best rewrite: "{node.query}"\n'
        prompt += (
            f"\nProvide exactly {self.n_children} rewrites, one per line, numbered:\n"
        )

        response = self._manager.generate(prompt, max_new_tokens=200)
        variants = self._parse_variants(response, node.query)

        children = []
        for variant in variants[: self.n_children]:
            child = self._add_node(variant, parent_id=node.id)
            children.append(child)
        return children

    def _simulate(self, query: str) -> tuple[float, list[dict], dict]:
        bm25_reward = 0.0
        bm25_hits: list[tuple[int, float]] = []  # (chunk_idx, score)

        tokenized = tokenize(query)
        if tokenized:
            scores = self.bm25.get_scores(tokenized)
            if len(scores) > 0:
                top_indices = np.argsort(scores)[::-1][:5]
                for idx in top_indices:
                    idx = int(idx)
                    s = float(scores[idx])
                    if s > 0:
                        bm25_hits.append((idx, s))
                if bm25_hits:
                    avg_bm25 = np.mean([s for _, s in bm25_hits])
                    bm25_reward = float(avg_bm25 / (avg_bm25 + 1.0))

        semantic_reward = 0.0
        semantic_hits: list[tuple[int, float]] = []

        if self.faiss_index is not None:
            query_vec = self._manager.encode_query(query)
            query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
            distances, indices = self.faiss_index.search(query_vec.astype("float32"), 5)
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0:
                    semantic_hits.append((int(idx), float(dist)))
            if semantic_hits:
                avg_sim = np.mean([s for _, s in semantic_hits])
                semantic_reward = float(max(0.0, avg_sim))

        seen: dict[int, dict] = {}
        for idx, bm25_score in bm25_hits:
            if idx not in seen:
                seen[idx] = {"bm25_score": bm25_score, "semantic_score": 0.0}
            else:
                seen[idx]["bm25_score"] = max(seen[idx]["bm25_score"], bm25_score)
        for idx, sem_score in semantic_hits:
            if idx not in seen:
                seen[idx] = {"bm25_score": 0.0, "semantic_score": sem_score}
            else:
                seen[idx]["semantic_score"] = max(
                    seen[idx]["semantic_score"], sem_score
                )

        ranked = sorted(
            seen.items(),
            key=lambda x: x[1]["bm25_score"] * 0.3 + x[1]["semantic_score"] * 0.7,
            reverse=True,
        )[:5]

        llm_reward = 0.0
        if ranked:
            top_idx = ranked[0][0]
            top_chunk = self.chunks[top_idx]
            llm_reward = self._llm_score(query, top_chunk)

        top_hits = []
        for idx, scores_dict in ranked:
            chunk = self.chunks[idx]
            display = (
                f"{chunk.class_name}.{chunk.method_name}"
                if chunk.class_name and chunk.method_name
                else chunk.method_name or chunk.chunk_id
            )
            top_hits.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "name": display,
                    "file_path": chunk.file_path,
                    "chunk_type": chunk.chunk_type,
                    "signature": chunk.signature,
                    "bm25_score": round(scores_dict["bm25_score"], 3),
                    "semantic_score": round(scores_dict["semantic_score"], 3),
                }
            )

        combined = (
            self.W_BM25 * bm25_reward
            + self.W_SEMANTIC * semantic_reward
            + self.W_LLM * llm_reward
        )

        components = {
            "bm25": round(bm25_reward, 4),
            "semantic": round(semantic_reward, 4),
            "llm": round(llm_reward, 4),
        }

        return combined, top_hits, components

    def _llm_score(self, query: str, chunk: CodeChunk) -> float:
        """Ask Qwen to rate relevance of a code chunk to the query (0-10 → 0-1)."""
        code_snippet = chunk.signature
        if chunk.javadoc:
            code_snippet = chunk.javadoc[:200] + "\n" + code_snippet

        prompt = (
            "Rate how relevant this Java code is to the search query. "
            "The query may be in any language. Score 0-10.\n"
            f"Query: {query}\n"
            f"Code: {code_snippet[:400]}\n"
            "Score:"
        )

        try:
            response = self._manager.generate(prompt, max_new_tokens=5)
            match = re.search(r"(\d+(?:\.\d+)?)", response)
            if match:
                score = float(match.group(1))
                return min(1.0, max(0.0, score / 10.0))
        except Exception as e:
            logger.error(f"MCTS: LLM error: {e}")

        return 0.0

    def _backpropagate(self, node_id: int, reward: float) -> None:
        current_id: int | None = node_id
        while current_id is not None:
            node = self.nodes[current_id]
            node.visits += 1
            node.total_reward += reward
            current_id = node.parent_id

    def _add_node(self, query: str, parent_id: int | None) -> _Node:
        node_id = len(self.nodes)
        node = _Node(node_id, query, parent_id)
        self.nodes.append(node)
        if parent_id is not None:
            self.nodes[parent_id].children_ids.append(node_id)
        return node

    def _best_leaf(self) -> _Node:
        leaves = [n for n in self.nodes if n.is_leaf()]
        return max(leaves, key=lambda n: n.avg_reward)

    def _path_to_root(self, node_id: int) -> list[int]:
        path = []
        current_id: int | None = node_id
        while current_id is not None:
            path.append(current_id)
            current_id = self.nodes[current_id].parent_id
        path.reverse()
        return path

    def _parse_variants(self, response: str, original: str) -> list[str]:
        variants = []
        for line in response.strip().split("\n"):
            cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
            cleaned = re.sub(r"^[-•]\s*", "", cleaned).strip()
            cleaned = re.sub(
                r"^(?:Rewrite\s*\d+\s*[:.]?\s*)", "", cleaned, flags=re.IGNORECASE
            ).strip()
            cleaned = cleaned.strip("\"'")
            if (
                cleaned
                and len(cleaned) > 5
                and cleaned.lower() != original.lower()
                and not re.match(r"^rewrite\s*\d*", cleaned, re.IGNORECASE)
            ):
                variants.append(cleaned)
        if not variants:
            variants.append(original)
        return variants

    def _extract_keywords(self, query: str) -> list[str]:
        words = re.split(r"[\s,;]+", query)
        keywords = []
        for w in words:
            w = w.strip().strip("\"'()[]{}")
            if w and len(w) > 1:
                keywords.append(w)
        return keywords[:10]

    def _build_trace(
        self, best_path: list[int], best_query: str, original_query: str
    ) -> dict:
        best_set = set(best_path)
        root_entity_ids = (
            {h["chunk_id"] for h in self.nodes[0].top_hits} if self.nodes else set()
        )

        nodes = []
        for n in self.nodes:
            hits_with_flags = []
            for h in n.top_hits:
                hits_with_flags.append(
                    {
                        **h,
                        "is_new": h["chunk_id"] not in root_entity_ids and n.id != 0,
                    }
                )

            nodes.append(
                {
                    "id": n.id,
                    "query": n.query,
                    "parent_id": n.parent_id,
                    "children_ids": list(n.children_ids),
                    "visits": n.visits,
                    "avg_reward": round(n.avg_reward, 4),
                    "is_best": n.id in best_set,
                    "top_hits": hits_with_flags,
                    "reward_components": {
                        "bm25": n.bm25_reward,
                        "semantic": n.semantic_reward,
                        "llm": n.llm_reward,
                    },
                }
            )

        return {
            "nodes": nodes,
            "iterations": self.n_iterations,
            "best_path": best_path,
            "best_query": best_query,
            "original_query": original_query,
        }


def mcts_rewrite(
    query: str,
    bm25,
    corpus: list[list[str]],
    chunks: list[CodeChunk],
    faiss_index=None,
) -> dict:
    rewriter = MCTSRewriter(bm25, corpus, chunks, faiss_index)
    return rewriter.rewrite(query)
