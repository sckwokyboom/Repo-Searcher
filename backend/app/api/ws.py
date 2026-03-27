from collections import defaultdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from pydantic import BaseModel

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

    async def broadcast(self, channel: str, progress: BaseModel):
        dead = []
        for ws in self.connections[channel]:
            try:
                await ws.send_json(progress.model_dump(mode="json"))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connections[channel].remove(ws)


connection_manager = ConnectionManager()


@router.websocket("/ws/indexing/{repo_id}")
async def indexing_ws(websocket: WebSocket, repo_id: str):
    await connection_manager.connect(repo_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(repo_id, websocket)


@router.websocket("/ws/lora/{repo_id}")
async def lora_ws(websocket: WebSocket, repo_id: str):
    channel = f"lora_{repo_id}"
    await connection_manager.connect(channel, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(channel, websocket)
