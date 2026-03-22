import networkx as nx

from app.indexer.store import load_call_graph, load_chunks
from app.models.graph import CallGraphEdge, CallGraphNode, CallGraphResponse

_graph_cache: dict[str, nx.DiGraph] = {}


class GraphExpander:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        if repo_id not in _graph_cache:
            _graph_cache[repo_id] = load_call_graph(repo_id)
        self.graph = _graph_cache[repo_id]

    def get_neighbors(self, chunk_id: str) -> tuple[list[str], list[str]]:
        if chunk_id not in self.graph:
            return [], []
        callers = list(self.graph.predecessors(chunk_id))
        callees = list(self.graph.successors(chunk_id))
        return callers, callees

    def get_subgraph(self, method_id: str, hops: int = 1) -> CallGraphResponse:
        if method_id not in self.graph:
            return CallGraphResponse(nodes=[], edges=[])

        # Collect nodes within N hops
        visited = {method_id}
        frontier = {method_id}

        for _ in range(hops):
            new_frontier = set()
            for node in frontier:
                if node in self.graph:
                    for pred in self.graph.predecessors(node):
                        if pred not in visited:
                            visited.add(pred)
                            new_frontier.add(pred)
                    for succ in self.graph.successors(node):
                        if succ not in visited:
                            visited.add(succ)
                            new_frontier.add(succ)
            frontier = new_frontier

        # Build response
        nodes = []
        for node_id in visited:
            data = self.graph.nodes.get(node_id, {})
            label = data.get("label", node_id.split("::")[-1])
            file_path = data.get("file_path", "")
            nodes.append(
                CallGraphNode(
                    id=node_id,
                    label=label,
                    file_path=file_path,
                    is_result=(node_id == method_id),
                )
            )

        edges = []
        for u, v in self.graph.edges():
            if u in visited and v in visited:
                edges.append(CallGraphEdge(source=u, target=v))

        return CallGraphResponse(nodes=nodes, edges=edges)
