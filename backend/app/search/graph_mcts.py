"""
Monte Carlo Tree Search over Call Graph.

After initial search results are found, this module explores the call graph
starting from seed nodes (top search hits) to discover structurally related
methods that may be relevant to the query.

Example: if `authenticate()` is found relevant, MCTS explores its callees
(`hashPassword()`, `generateJWT()`) and callers (`loginController()`),
scoring them by semantic similarity to the original query.
"""

import math

import networkx as nx
import numpy as np

from app.config import settings
from app.ml.model_manager import get_model_manager
from app.models.search import CodeChunk, SearchResult


class _GraphNode:
    __slots__ = (
        "chunk_id", "parent_chunk_id", "relation",
        "visits", "total_reward", "children_expanded",
    )

    def __init__(
        self,
        chunk_id: str,
        parent_chunk_id: str | None = None,
        relation: str = "",
    ):
        self.chunk_id = chunk_id
        self.parent_chunk_id = parent_chunk_id
        self.relation = relation  # "calls" | "called_by" | "seed"
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.children_expanded: bool = False

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0


class GraphMCTS:
    """
    Explores the call graph using MCTS to find relevant code
    that is structurally connected to initial search results.
    """

    C = 1.41  # UCB1 exploration constant

    def __init__(
        self,
        call_graph: nx.DiGraph,
        chunks: list[CodeChunk],
        faiss_index,
    ):
        self.graph = call_graph
        self.chunks = chunks
        self.chunk_map: dict[str, CodeChunk] = {c.chunk_id: c for c in chunks}
        self.faiss_index = faiss_index
        self._manager = get_model_manager()

        # MCTS state
        self.nodes: dict[str, _GraphNode] = {}  # chunk_id -> node
        self.seed_ids: list[str] = []

    def explore(
        self,
        query: str,
        seed_chunk_ids: list[str],
    ) -> dict:
        """
        Run MCTS exploration from seed nodes.

        Returns dict with:
          - discoveries: list of SearchResult (source="graph_mcts")
          - trace: GraphMCTSTraceInfo dict
        """
        print(f"[GraphMCTS] Starting exploration from {len(seed_chunk_ids)} seeds", flush=True)

        # Filter seeds to those that exist in the call graph
        self.seed_ids = [cid for cid in seed_chunk_ids if cid in self.graph]
        if not self.seed_ids:
            print("[GraphMCTS] No seeds found in call graph, skipping", flush=True)
            return {"discoveries": [], "trace": self._build_trace()}

        # Encode query once for semantic scoring
        query_vec = self._manager.encode_query(query)
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        self._query_vec = query_vec

        # Initialize seed nodes
        for cid in self.seed_ids:
            node = _GraphNode(cid, parent_chunk_id=None, relation="seed")
            reward = self._semantic_reward(cid)
            node.visits = 1
            node.total_reward = reward
            self.nodes[cid] = node
            print(f"[GraphMCTS]   Seed {cid.split('::')[-1]}: reward={reward:.3f}", flush=True)

        # MCTS iterations
        n_iterations = settings.graph_mcts_iterations
        for iteration in range(n_iterations):
            print(f"[GraphMCTS] === Iteration {iteration + 1}/{n_iterations} ===", flush=True)

            # Selection: pick the most promising unexpanded node
            selected = self._select()
            if selected is None:
                print("[GraphMCTS]   No more nodes to expand", flush=True)
                break

            print(f"[GraphMCTS]   Selected: {selected.chunk_id.split('::')[-1]}", flush=True)

            # Expansion: get neighbors from call graph
            children = self._expand(selected)
            print(f"[GraphMCTS]   Found {len(children)} neighbors", flush=True)

            # Simulation + backpropagation for each child
            for child in children:
                reward = self._semantic_reward(child.chunk_id)
                child.visits += 1
                child.total_reward += reward
                # Backpropagate to parent chain
                self._backpropagate(selected.chunk_id, reward)
                print(
                    f"[GraphMCTS]   {child.chunk_id.split('::')[-1]} "
                    f"({child.relation}) -> reward={reward:.3f}",
                    flush=True,
                )

        # Collect discoveries: non-seed nodes with high enough reward
        threshold = settings.graph_mcts_reward_threshold
        max_disc = settings.graph_mcts_max_discoveries
        seed_set = set(self.seed_ids)

        candidates = [
            n for n in self.nodes.values()
            if n.chunk_id not in seed_set and n.avg_reward >= threshold
        ]
        candidates.sort(key=lambda n: n.avg_reward, reverse=True)
        discoveries = candidates[:max_disc]

        print(
            f"[GraphMCTS] Done: {len(self.nodes)} nodes explored, "
            f"{len(discoveries)} discoveries (threshold={threshold})",
            flush=True,
        )

        # Build SearchResult objects for discoveries
        result_items = []
        for disc in discoveries:
            chunk = self.chunk_map.get(disc.chunk_id)
            if chunk is None:
                continue

            # Find display name for discovered_via
            via_name = ""
            if disc.parent_chunk_id and disc.parent_chunk_id in self.chunk_map:
                via_chunk = self.chunk_map[disc.parent_chunk_id]
                via_name = (
                    f"{via_chunk.class_name}.{via_chunk.method_name}"
                    if via_chunk.class_name and via_chunk.method_name
                    else via_chunk.method_name or disc.parent_chunk_id
                )

            result_items.append(
                SearchResult(
                    chunk=chunk,
                    score=round(disc.avg_reward * 10, 1),  # Scale to 0-10 range
                    source="graph_mcts",
                    discovered_via=via_name,
                    relation=disc.relation,
                )
            )

        return {
            "discoveries": result_items,
            "trace": self._build_trace(),
        }

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _select(self) -> _GraphNode | None:
        """Select the most promising unexpanded node using UCB1."""
        best_node = None
        best_score = -1.0

        for node in self.nodes.values():
            if node.children_expanded:
                continue  # Already expanded

            if node.visits == 0:
                return node  # Unvisited → explore immediately

            # UCB1 with total visits as parent
            total_visits = sum(n.visits for n in self.nodes.values())
            ucb = node.avg_reward + self.C * math.sqrt(
                math.log(total_visits) / node.visits
            )

            if ucb > best_score:
                best_score = ucb
                best_node = node

        return best_node

    def _expand(self, node: _GraphNode) -> list[_GraphNode]:
        """Get callers and callees from the call graph as new MCTS nodes."""
        node.children_expanded = True
        children = []

        chunk_id = node.chunk_id
        if chunk_id not in self.graph:
            return children

        # Callees (methods this node calls)
        for callee_id in self.graph.successors(chunk_id):
            if callee_id not in self.nodes and callee_id in self.chunk_map:
                child = _GraphNode(callee_id, parent_chunk_id=chunk_id, relation="calls")
                self.nodes[callee_id] = child
                children.append(child)

        # Callers (methods that call this node)
        for caller_id in self.graph.predecessors(chunk_id):
            if caller_id not in self.nodes and caller_id in self.chunk_map:
                child = _GraphNode(caller_id, parent_chunk_id=chunk_id, relation="called_by")
                self.nodes[caller_id] = child
                children.append(child)

        return children

    def _semantic_reward(self, chunk_id: str) -> float:
        """Compute cosine similarity between query and chunk using UniXcoder."""
        chunk = self.chunk_map.get(chunk_id)
        if chunk is None:
            return 0.0

        # Encode chunk text
        chunk_vec = self._manager.encode_code([chunk.text_representation])
        chunk_vec = chunk_vec / np.linalg.norm(chunk_vec, axis=1, keepdims=True)

        # Cosine similarity via dot product (both normalized)
        similarity = float(np.dot(self._query_vec, chunk_vec.T)[0, 0])
        return max(0.0, similarity)

    def _backpropagate(self, chunk_id: str, reward: float) -> None:
        """Propagate reward up through parent chain."""
        current_id: str | None = chunk_id
        while current_id is not None:
            node = self.nodes.get(current_id)
            if node is None:
                break
            node.visits += 1
            node.total_reward += reward
            current_id = node.parent_chunk_id

    # ------------------------------------------------------------------
    # Trace building
    # ------------------------------------------------------------------

    def _build_trace(self) -> dict:
        explored = []
        seed_set = set(self.seed_ids)

        for node in self.nodes.values():
            if node.chunk_id in seed_set:
                continue  # Don't include seeds in trace
            chunk = self.chunk_map.get(node.chunk_id)
            if chunk is None:
                continue
            display = (
                f"{chunk.class_name}.{chunk.method_name}"
                if chunk.class_name and chunk.method_name
                else chunk.method_name or chunk.chunk_id
            )
            explored.append({
                "chunk_id": node.chunk_id,
                "name": display,
                "file_path": chunk.file_path,
                "visits": node.visits,
                "avg_reward": round(node.avg_reward, 4),
                "discovered_via": node.parent_chunk_id or "",
                "relation": node.relation,
            })

        # Sort by reward descending
        explored.sort(key=lambda x: x["avg_reward"], reverse=True)

        return {
            "explored_nodes": explored,
            "total_nodes_visited": len(self.nodes),
            "discoveries_count": len([
                n for n in self.nodes.values()
                if n.chunk_id not in set(self.seed_ids)
                and n.avg_reward >= settings.graph_mcts_reward_threshold
            ]),
        }
