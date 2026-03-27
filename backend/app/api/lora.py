import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.ws import connection_manager
from app.config import settings
from app.indexer.store import load_chunks
from app.ml.lora_data_generator import estimate_training_time, fast_estimate_samples
from app.ml.lora_registry import (
    assign_adapter,
    get_active_adapter_id,
    has_adapter,
    list_adapters,
    unassign_adapter,
)
from app.ml.lora_trainer import LoRATrainer
from app.ml.model_manager import reset_model_manager
from app.models.repo import LoRAAdapterInfo, LoRAStatusResponse, LoRATrainingProgress

router = APIRouter(tags=["lora"])

_executor = ThreadPoolExecutor(max_workers=1)
active_lora_tasks: dict[str, asyncio.Task] = {}
cancel_events: dict[str, threading.Event] = {}
latest_lora_progress: dict[str, LoRATrainingProgress] = {}


@router.get("/lora/adapters", response_model=list[LoRAAdapterInfo])
async def get_available_adapters():
    return [
        LoRAAdapterInfo(
            adapter_id=a.adapter_id,
            name=a.name,
            description=a.description,
            source=a.source,
            trained_for_repo=a.trained_for_repo,
        )
        for a in list_adapters()
    ]


class SelectAdapterRequest(BaseModel):
    adapter_id: str | None = None


@router.post("/repos/{repo_id}/lora/select")
async def select_adapter(repo_id: str, request: SelectAdapterRequest):
    if request.adapter_id is None:
        unassign_adapter(repo_id)
        reset_model_manager()
        return {"repo_id": repo_id, "status": "adapter_removed"}

    if not assign_adapter(repo_id, request.adapter_id):
        raise HTTPException(status_code=404, detail="Adapter not found")

    reset_model_manager()
    return {
        "repo_id": repo_id,
        "status": "adapter_assigned",
        "adapter_id": request.adapter_id,
    }


@router.post("/repos/{repo_id}/lora/train", status_code=202)
async def start_lora_training(repo_id: str):
    metadata_path = settings.indexes_dir / repo_id / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Repository not indexed")

    if repo_id in active_lora_tasks and not active_lora_tasks[repo_id].done():
        raise HTTPException(status_code=409, detail="LoRA training already in progress")

    chunks = load_chunks(repo_id)
    estimated_samples = fast_estimate_samples(len(chunks))
    estimated_min = estimate_training_time(
        estimated_samples,
        epochs=settings.lora_epochs,
        batch_size=settings.lora_batch_size,
        grad_accum=settings.lora_gradient_accumulation,
    )

    cancel_event = threading.Event()
    cancel_events[repo_id] = cancel_event

    loop = asyncio.get_event_loop()

    def sync_progress_fn(progress: LoRATrainingProgress):
        latest_lora_progress[repo_id] = progress
        asyncio.run_coroutine_threadsafe(
            connection_manager.broadcast(f"lora_{repo_id}", progress),
            loop,
        )

    trainer = LoRATrainer(
        repo_id=repo_id,
        chunks=chunks,
        progress_fn=sync_progress_fn,
        cancel_event=cancel_event,
    )

    async def run_training():
        await loop.run_in_executor(_executor, trainer.run)

    task = asyncio.create_task(run_training())
    active_lora_tasks[repo_id] = task

    return {
        "repo_id": repo_id,
        "status": "training_started",
        "estimated_minutes": estimated_min,
        "num_samples": estimated_samples,
    }


@router.post("/repos/{repo_id}/lora/cancel")
async def cancel_lora_training(repo_id: str):
    if repo_id not in cancel_events:
        raise HTTPException(status_code=404, detail="No active training for this repo")

    cancel_events[repo_id].set()
    return {"repo_id": repo_id, "status": "cancelling"}


@router.get("/repos/{repo_id}/lora/status", response_model=LoRAStatusResponse)
async def get_lora_status(repo_id: str):
    is_training = repo_id in active_lora_tasks and not active_lora_tasks[repo_id].done()

    estimated_min = None
    if not has_adapter(repo_id) and not is_training:
        metadata_path = settings.indexes_dir / repo_id / "metadata.json"
        if metadata_path.exists():
            try:
                chunks = load_chunks(repo_id)
                estimated_samples = fast_estimate_samples(len(chunks))
                estimated_min = estimate_training_time(
                    estimated_samples,
                    epochs=settings.lora_epochs,
                    batch_size=settings.lora_batch_size,
                    grad_accum=settings.lora_gradient_accumulation,
                )
            except Exception:
                pass

    return LoRAStatusResponse(
        repo_id=repo_id,
        has_adapter=has_adapter(repo_id),
        active_adapter_id=get_active_adapter_id(repo_id),
        is_training=is_training,
        estimated_minutes=estimated_min,
    )
