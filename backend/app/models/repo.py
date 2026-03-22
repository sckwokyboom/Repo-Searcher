from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class IndexingStep(str, Enum):
    CLONING = "cloning"
    PARSING = "parsing"
    BUILDING_BM25 = "building_bm25"
    BUILDING_VECTORS = "building_vectors"
    BUILDING_CALLGRAPH = "building_callgraph"
    SAVING = "saving"
    DONE = "done"
    FAILED = "failed"


class RepoInfo(BaseModel):
    repo_id: str
    owner: str
    name: str
    full_name: str
    description: str | None = None
    stars: int = 0
    url: str
    language: str | None = None
    indexed_at: datetime | None = None
    chunk_count: int = 0


class IndexingProgress(BaseModel):
    repo_id: str
    step: IndexingStep
    progress: float = 0.0
    message: str = ""
    files_processed: int = 0
    files_total: int = 0


class IndexingStatusResponse(BaseModel):
    repo_id: str
    status: IndexingStep
    repo_info: RepoInfo | None = None


class GitHubSearchResult(BaseModel):
    full_name: str
    description: str | None = None
    stars: int = 0
    url: str
    owner: str
    owner_avatar: str = ""
    language: str | None = None
