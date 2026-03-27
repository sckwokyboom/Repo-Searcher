import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.search import SearchRequest, SearchResponse

router = APIRouter(tags=["search"])

_executor = ThreadPoolExecutor(max_workers=2)


@router.post("/repos/{repo_id}/search", response_model=SearchResponse)
async def search_code(repo_id: str, request: SearchRequest):
    metadata_path = settings.indexes_dir / repo_id / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Repository not indexed")

    from app.search.hybrid_retriever import HybridRetriever

    start = time.time()

    def _run_search():
        retriever = HybridRetriever(repo_id)
        return retriever.search_sync(request.query, request.top_k)

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(_executor, _run_search)
    elapsed = (time.time() - start) * 1000

    return SearchResponse(
        query=request.query,
        rewritten_query=results.get("rewritten_query"),
        expanded_keywords=results.get("expanded_keywords", []),
        rewrite_details=results.get("rewrite_details"),
        results=results["results"],
        search_time_ms=round(elapsed, 1),
        lora_active=results.get("lora_active", False),
    )
