from collections import defaultdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.models.repo import IndexingProgress

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    def __init__(self):
        self.connections: dict[str, list[WebSocket]] = defaultdict(list)

    async def connect(self, repo_id: str, ws: WebSocket):
        await ws.accept()
        self.connections[repo_id].append(ws)

    def disconnect(self, repo_id: str, ws: WebSocket):
        if ws in self.connections[repo_id]:
            self.connections[repo_id].remove(ws)

    async def broadcast(self, repo_id: str, progress: IndexingProgress):
        dead = []
        for ws in self.connections[repo_id]:
            try:
                await ws.send_json(progress.model_dump(mode="json"))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connections[repo_id].remove(ws)


connection_manager = ConnectionManager()


@router.websocket("/ws/indexing/{repo_id}")
async def indexing_ws(websocket: WebSocket, repo_id: str):
    await connection_manager.connect(repo_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(repo_id, websocket)
