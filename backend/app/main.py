import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import graph, lora, repos, search, ws
from app.config import settings
from app.ml.lora_registry import get_adapter_path
from app.ml.model_manager import get_model_manager

logger = logging.getLogger("uvicorn.startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    default_adapter = get_adapter_path(settings.default_lora_repo_id)
    if default_adapter:
        logger.info(f"Default LoRA adapter found: {default_adapter}")
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
    allow_origins=["*"],
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


if settings.dist_path.is_dir():
    logger.info(f"Found dist path: {settings.dist_path}")
    app.mount(
        "/assets", StaticFiles(directory=settings.dist_path / "assets"), name="static"
    )

    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        file_path = settings.dist_path / full_path

        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        return FileResponse(settings.dist_path / "index.html")
