from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class IndexingStep(str, Enum):
    CLONING = "cloning"
    PARSING = "parsing"
    BUILDING_BM25 = "building_bm25"
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
    has_lora_adapter: bool = False


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


class LoRATrainingStep(str, Enum):
    PREPARING_DATA = "preparing_data"
    TRAINING = "training"
    SAVING = "saving"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LoRATrainingProgress(BaseModel):
    repo_id: str
    step: LoRATrainingStep
    progress: float = 0.0
    message: str = ""
    epoch: int = 0
    total_epochs: int = 0
    train_loss: float | None = None
    eval_loss: float | None = None
    estimated_time_remaining_sec: int | None = None


class LoRAAdapterInfo(BaseModel):
    adapter_id: str
    name: str
    description: str
    source: str  # "bundled" | "trained"
    trained_for_repo: str | None = None


class LoRAStatusResponse(BaseModel):
    repo_id: str
    has_adapter: bool = False
    active_adapter_id: str | None = None
    is_training: bool = False
    estimated_minutes: float | None = None


class GitHubSearchResult(BaseModel):
    full_name: str
    description: str | None = None
    stars: int = 0
    url: str
    owner: str
    owner_avatar: str = ""
    language: str | None = None
