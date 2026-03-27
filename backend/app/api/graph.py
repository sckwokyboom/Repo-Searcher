from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.graph import CallGraphResponse

router = APIRouter(tags=["graph"])


@router.get("/repos/{repo_id}/graph/{method_id:path}", response_model=CallGraphResponse)
async def get_call_graph(repo_id: str, method_id: str):
    metadata_path = settings.indexes_dir / repo_id / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Repository not indexed")

    from app.search.graph_expander import GraphExpander

    expander = GraphExpander(repo_id)
    return expander.get_subgraph(method_id)
