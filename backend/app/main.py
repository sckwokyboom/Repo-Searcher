import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import repos, search, graph, ws, lora
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-initialize ModelManager with default LoRA adapter if available
    from app.ml.lora_registry import get_adapter_path
    default_adapter = get_adapter_path(settings.default_lora_repo_id)
    if default_adapter:
        from app.ml.model_manager import get_model_manager
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Default LoRA adapter found: {default_adapter}")
        # Don't eagerly load — just configure the path for first use
        get_model_manager(lora_adapter_path=str(default_adapter))
    yield


app = FastAPI(
    title="CodeGraph Search",
    description="Natural language code search for Java repositories",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(repos.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(graph.router, prefix="/api")
app.include_router(ws.router, prefix="/api")
app.include_router(lora.router, prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
