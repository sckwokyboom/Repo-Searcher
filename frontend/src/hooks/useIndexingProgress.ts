import { useState, useEffect, useRef, useCallback } from "react";
import type { IndexingProgress } from "../types";
import { WS_BASE } from "../services/api";

export function useIndexingProgress(repoId: string | null) {
  const [progress, setProgress] = useState<IndexingProgress | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    if (!repoId) return;

    const ws = new WebSocket(`${WS_BASE}/api/ws/indexing/${repoId}`);
    wsRef.current = ws;

    ws.onopen = () => setIsConnected(true);

    ws.onmessage = (event) => {
      const data: IndexingProgress = JSON.parse(event.data);
      setProgress(data);
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [repoId]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  return { progress, isConnected };
}
