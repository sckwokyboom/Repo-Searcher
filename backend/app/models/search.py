from typing import Literal, TypedDict

from pydantic import BaseModel


class JavaClassInfo(TypedDict):
    javadoc: str | None
    name: str


class CodeChunk(BaseModel):
    chunk_id: str
    chunk_type: Literal["method", "class"]
    file_path: str
    class_name: str | None = None
    method_name: str | None = None
    signature: str
    javadoc: str | None = None
    body: str
    start_line: int
    end_line: int
    text_representation: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    chunk: CodeChunk
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rrf_score: float | None = None
    callers: list[str] = []
    callees: list[str] = []
    source: str = "search"  # "search" | "graph_mcts"
    discovered_via: str | None = None  # chunk_id of the node that led to discovery
    relation: str | None = None  # "calls" | "called_by"


# --- MCTS Trace models ---


class MCTSHitInfo(BaseModel):
    """A code entity found by a particular MCTS query variant."""

    chunk_id: str
    name: str  # e.g. "UserService.findById"
    file_path: str
    chunk_type: str  # "method" | "class"
    signature: str
    bm25_score: float
    semantic_score: float = 0.0
    is_new: bool = False  # True if this entity wasn't found by root query


class MCTSRewardComponents(BaseModel):
    """Decomposed reward signal for visualization."""

    bm25: float = 0.0
    semantic: float = 0.0
    llm: float = 0.0


class MCTSNodeInfo(BaseModel):
    id: int
    query: str
    parent_id: int | None = None
    children_ids: list[int] = []
    visits: int = 0
    avg_reward: float = 0.0
    is_best: bool = False
    top_hits: list[MCTSHitInfo] = []
    reward_components: MCTSRewardComponents = MCTSRewardComponents()


class MCTSTraceInfo(BaseModel):
    nodes: list[MCTSNodeInfo] = []
    iterations: int = 0
    best_path: list[int] = []
    best_query: str = ""
    original_query: str = ""


class GraphMCTSNodeInfo(BaseModel):
    """A node explored during Call Graph MCTS."""

    chunk_id: str
    name: str
    file_path: str
    visits: int = 0
    avg_reward: float = 0.0
    discovered_via: str = ""  # chunk_id of parent in exploration
    relation: str = ""  # "calls" | "called_by"


class GraphMCTSTraceInfo(BaseModel):
    """Trace of the Call Graph MCTS exploration."""

    explored_nodes: list[GraphMCTSNodeInfo] = []
    total_nodes_visited: int = 0
    discoveries_count: int = 0


class SearchResponse(BaseModel):
    query: str
    expanded_keywords: list[str] = []
    results: list[SearchResult]
    search_time_ms: float
    mcts_trace: MCTSTraceInfo | None = None
    graph_mcts_trace: GraphMCTSTraceInfo | None = None
