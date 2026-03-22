import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import torch before faiss to avoid OpenMP crash on macOS
import torch  # noqa: F401, E402
import faiss  # noqa: F401, E402

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import repos, search, graph, ws
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
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


@app.get("/api/health")
async def health():
    return {"status": "ok"}
