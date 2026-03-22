import { useEffect, useRef, useState } from "react";
import {
  Box,
  CircularProgress,
  IconButton,
  Paper,
  Typography,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import type { CallGraphData } from "../types";
import { getCallGraph } from "../services/graphService";

interface Props {
  repoId: string;
  methodId: string | null;
  onClose: () => void;
}

interface GraphNode {
  id: string;
  label: string;
  file_path: string;
  is_result: boolean;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
}

interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
}

export default function CallGraphPanel({ repoId, methodId, onClose }: Props) {
  const [data, setData] = useState<CallGraphData | null>(null);
  const [loading, setLoading] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<any>(null);

  useEffect(() => {
    if (!methodId) return;
    setLoading(true);
    getCallGraph(repoId, methodId)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [repoId, methodId]);

  useEffect(() => {
    if (!data || !containerRef.current) return;

    let ForceGraph: any;
    import("react-force-graph-2d").then((mod) => {
      ForceGraph = mod.default;
      renderGraph(ForceGraph);
    });

    function renderGraph(FG: any) {
      // We'll use a canvas-based approach instead
    }
  }, [data]);

  if (!methodId) return null;

  const graphData = data
    ? {
        nodes: data.nodes.map((n) => ({
          ...n,
          val: n.is_result ? 3 : 1,
          color: n.is_result ? "#7c4dff" : "#64748b",
        })),
        links: data.edges.map((e) => ({
          source: e.source,
          target: e.target,
        })),
      }
    : { nodes: [], links: [] };

  return (
    <Paper
      sx={{
        mt: 3,
        p: 2,
        position: "relative",
        minHeight: 300,
      }}
    >
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          mb: 1,
        }}
      >
        <Typography variant="subtitle2" color="text.secondary">
          Call Graph
        </Typography>
        <IconButton size="small" onClick={onClose}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>

      {loading && (
        <Box
          sx={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            height: 250,
          }}
        >
          <CircularProgress />
        </Box>
      )}

      {!loading && data && (
        <Box ref={containerRef} sx={{ height: 300, position: "relative" }}>
          <CallGraphCanvas data={graphData} />
        </Box>
      )}

      {!loading && data && data.nodes.length === 0 && (
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ textAlign: "center", py: 4 }}
        >
          No call graph data available for this method.
        </Typography>
      )}
    </Paper>
  );
}

function CallGraphCanvas({
  data,
}: {
  data: { nodes: any[]; links: any[] };
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.nodes.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.parentElement?.clientWidth || 600;
    const height = 280;
    canvas.width = width;
    canvas.height = height;

    // Simple force-directed layout
    const nodes: GraphNode[] = data.nodes.map((n, i) => ({
      ...n,
      x: width / 2 + Math.cos((2 * Math.PI * i) / data.nodes.length) * 100,
      y: height / 2 + Math.sin((2 * Math.PI * i) / data.nodes.length) * 80,
      vx: 0,
      vy: 0,
    }));

    const nodeMap = new Map(nodes.map((n) => [n.id, n]));

    const links: { source: GraphNode; target: GraphNode }[] = data.links
      .map((l: any) => ({
        source: nodeMap.get(typeof l.source === "string" ? l.source : l.source.id)!,
        target: nodeMap.get(typeof l.target === "string" ? l.target : l.target.id)!,
      }))
      .filter((l) => l.source && l.target);

    // Simulate
    for (let tick = 0; tick < 100; tick++) {
      // Repulsion
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[j].x! - nodes[i].x!;
          const dy = nodes[j].y! - nodes[i].y!;
          const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
          const force = 2000 / (dist * dist);
          nodes[i].vx! -= (dx / dist) * force;
          nodes[i].vy! -= (dy / dist) * force;
          nodes[j].vx! += (dx / dist) * force;
          nodes[j].vy! += (dy / dist) * force;
        }
      }
      // Attraction (edges)
      for (const link of links) {
        const dx = link.target.x! - link.source.x!;
        const dy = link.target.y! - link.source.y!;
        const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
        const force = (dist - 120) * 0.01;
        link.source.vx! += (dx / dist) * force;
        link.source.vy! += (dy / dist) * force;
        link.target.vx! -= (dx / dist) * force;
        link.target.vy! -= (dy / dist) * force;
      }
      // Center gravity
      for (const n of nodes) {
        n.vx! += (width / 2 - n.x!) * 0.005;
        n.vy! += (height / 2 - n.y!) * 0.005;
        n.vx! *= 0.9;
        n.vy! *= 0.9;
        n.x! += n.vx!;
        n.y! += n.vy!;
        n.x = Math.max(40, Math.min(width - 40, n.x!));
        n.y = Math.max(30, Math.min(height - 30, n.y!));
      }
    }

    // Draw
    ctx.clearRect(0, 0, width, height);

    // Edges with arrows
    for (const link of links) {
      ctx.beginPath();
      ctx.strokeStyle = "rgba(148, 163, 184, 0.3)";
      ctx.lineWidth = 1.5;
      ctx.moveTo(link.source.x!, link.source.y!);
      ctx.lineTo(link.target.x!, link.target.y!);
      ctx.stroke();

      // Arrow
      const angle = Math.atan2(
        link.target.y! - link.source.y!,
        link.target.x! - link.source.x!
      );
      const arrowLen = 8;
      const tx = link.target.x! - Math.cos(angle) * 14;
      const ty = link.target.y! - Math.sin(angle) * 14;
      ctx.beginPath();
      ctx.fillStyle = "rgba(148, 163, 184, 0.4)";
      ctx.moveTo(tx, ty);
      ctx.lineTo(
        tx - arrowLen * Math.cos(angle - Math.PI / 6),
        ty - arrowLen * Math.sin(angle - Math.PI / 6)
      );
      ctx.lineTo(
        tx - arrowLen * Math.cos(angle + Math.PI / 6),
        ty - arrowLen * Math.sin(angle + Math.PI / 6)
      );
      ctx.closePath();
      ctx.fill();
    }

    // Nodes
    for (const node of nodes) {
      const radius = node.is_result ? 10 : 7;
      ctx.beginPath();
      ctx.arc(node.x!, node.y!, radius, 0, Math.PI * 2);
      ctx.fillStyle = node.is_result ? "#7c4dff" : "#475569";
      ctx.fill();
      ctx.strokeStyle = node.is_result ? "#b47cff" : "#64748b";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label
      ctx.fillStyle = "#e2e8f0";
      ctx.font = node.is_result ? "bold 11px Inter, sans-serif" : "10px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(node.label, node.x!, node.y! + radius + 14);
    }
  }, [data]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height: 280, display: "block" }}
    />
  );
}
