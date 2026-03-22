from pydantic import BaseModel


class CallGraphNode(BaseModel):
    id: str
    label: str
    file_path: str
    is_result: bool = False


class CallGraphEdge(BaseModel):
    source: str
    target: str


class CallGraphResponse(BaseModel):
    nodes: list[CallGraphNode]
    edges: list[CallGraphEdge]
