import asyncio
import json
import shutil
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Query

from app.api.ws import connection_manager
from app.config import settings
from app.indexer.orchestrator import IndexingOrchestrator
from app.ml.lora_registry import has_adapter
from app.models.repo import (
    GitHubSearchResult,
    IndexingProgress,
    IndexingStatusResponse,
    IndexingStep,
    RepoInfo,
)

router = APIRouter(tags=["repos"])

active_tasks: dict[str, asyncio.Task] = {}
latest_progress: dict[str, IndexingProgress] = {}


@router.get("/repos/search", response_model=list[GitHubSearchResult])
async def search_repos(q: str = Query(..., min_length=1)):
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.github.com/search/repositories",
            params={"q": f"{q} language:java", "per_page": 10, "sort": "stars"},
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="GitHub API error")

        data = resp.json()
        return [
            GitHubSearchResult(
                full_name=item["full_name"],
                description=item.get("description"),
                stars=item.get("stargazers_count", 0),
                url=item["html_url"],
                owner=item["owner"]["login"],
                owner_avatar=item["owner"].get("avatar_url", ""),
                language=item.get("language"),
            )
            for item in data.get("items", [])
        ]


@router.get("/repos/indexed", response_model=list[RepoInfo])
async def list_indexed_repos():

    registry_path = settings.indexes_dir / "registry.json"
    if not registry_path.exists():
        return []
    with open(registry_path) as f:
        repos = json.load(f)

    result = []
    for r in repos:
        info = RepoInfo(**r)
        info.has_lora_adapter = has_adapter(info.repo_id)
        result.append(info)
    return result


@router.post("/repos/index", status_code=202)
async def index_repo(repo: RepoInfo):

    if repo.repo_id in active_tasks and not active_tasks[repo.repo_id].done():
        raise HTTPException(status_code=409, detail="Indexing already in progress")

    metadata_path = settings.indexes_dir / repo.repo_id / "metadata.json"
    if metadata_path.exists():
        raise HTTPException(status_code=409, detail="Repository already indexed")

    async def progress_callback(progress: IndexingProgress):
        latest_progress[repo.repo_id] = progress
        await connection_manager.broadcast(repo.repo_id, progress)

    orchestrator = IndexingOrchestrator(repo, progress_callback)
    task = asyncio.create_task(orchestrator.run())
    active_tasks[repo.repo_id] = task

    return {"repo_id": repo.repo_id, "status": "indexing_started"}


@router.get("/repos/{repo_id}/status", response_model=IndexingStatusResponse)
async def get_repo_status(repo_id: str):
    metadata_path = settings.indexes_dir / repo_id / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        return IndexingStatusResponse(
            repo_id=repo_id,
            status=IndexingStep.DONE,
            repo_info=RepoInfo(**meta["repo_info"]),
        )

    if repo_id in active_tasks and not active_tasks[repo_id].done():
        progress = latest_progress.get(repo_id)
        return IndexingStatusResponse(
            repo_id=repo_id,
            status=progress.step if progress else IndexingStep.CLONING,
        )

    return IndexingStatusResponse(repo_id=repo_id, status=IndexingStep.FAILED)


@router.delete("/repos/{repo_id}")
async def delete_repo_index(repo_id: str):

    index_dir = settings.indexes_dir / repo_id
    if index_dir.exists():
        shutil.rmtree(index_dir)

    registry_path = settings.indexes_dir / "registry.json"
    if registry_path.exists():
        with open(registry_path) as f:
            repos = json.load(f)
        repos = [r for r in repos if r["repo_id"] != repo_id]
        with open(registry_path, "w") as f:
            json.dump(repos, f)

    return {"status": "deleted"}
