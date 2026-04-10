import { useState, useEffect, useRef, useCallback } from "react";
import type { LoRATrainingProgress } from "../types";
import { WS_BASE } from "../services/api";

export function useLoRATraining(repoId: string | null) {
  const [progress, setProgress] = useState<LoRATrainingProgress | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    if (!repoId) return;

    let ws: WebSocket;
    try {
      ws = new WebSocket(`${WS_BASE}/api/ws/lora/${repoId}`);
    } catch {
      return;
    }
    wsRef.current = ws;

    ws.onopen = () => setIsConnected(true);

    ws.onmessage = (event) => {
      const data: LoRATrainingProgress = JSON.parse(event.data);
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
