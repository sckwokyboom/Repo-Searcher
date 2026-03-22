from typing import Literal

from pydantic import BaseModel


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


class SearchResponse(BaseModel):
    query: str
    expanded_keywords: list[str] = []
    results: list[SearchResult]
    search_time_ms: float
