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
    source: str = "search"
    discovered_via: str | None = None
    relation: str | None = None


class MCTSHitInfo(BaseModel):
    chunk_id: str
    name: str
    file_path: str
    chunk_type: str
    signature: str
    bm25_score: float
    semantic_score: float = 0.0
    is_new: bool = False


class MCTSRewardComponents(BaseModel):
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
    chunk_id: str
    name: str
    file_path: str
    visits: int = 0
    avg_reward: float = 0.0
    discovered_via: str = ""
    relation: str = ""


class GraphMCTSTraceInfo(BaseModel):
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
