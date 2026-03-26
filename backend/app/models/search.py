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
    callers: list[str] = []
    callees: list[str] = []


class RewriteDetails(BaseModel):
    """Structured query rewriting output from LLM (matches LoRA training format)."""
    intent: str | None = None
    search_scope: str | None = None
    keywords: list[str] = []
    project_terms: list[str] = []
    method_hints: list[str] = []
    api_hints: list[str] = []
    search_queries: list[str] = []


class SearchResponse(BaseModel):
    query: str
    rewritten_query: str | None = None
    expanded_keywords: list[str] = []
    rewrite_details: RewriteDetails | None = None
    results: list[SearchResult]
    search_time_ms: float
    lora_active: bool = False
