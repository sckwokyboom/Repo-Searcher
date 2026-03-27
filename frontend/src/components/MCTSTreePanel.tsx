import { useState } from "react";
import {
  Box,
  Card,
  CardContent,
  Chip,
  Collapse,
  Divider,
  IconButton,
  LinearProgress,
  Tooltip,
  Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import PsychologyIcon from "@mui/icons-material/Psychology";
import ArrowRightAltIcon from "@mui/icons-material/ArrowRightAlt";
import StarIcon from "@mui/icons-material/Star";
import FiberNewIcon from "@mui/icons-material/FiberNew";
import type { MCTSTrace, MCTSNode, MCTSHit, MCTSRewardComponents } from "../types";

interface Props {
  trace: MCTSTrace;
}

export default function MCTSTreePanel({ trace }: Props) {
  const [expanded, setExpanded] = useState(true);
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null);

  const root = trace.nodes.find((n: MCTSNode) => n.parent_id === null);
  if (!root) return null;

  const bestNode = trace.nodes.find(
    (n: MCTSNode) => n.is_best && n.children_ids.length === 0
  );

  const iterations = getIterationGroups(trace.nodes);
  const maxReward = Math.max(...trace.nodes.map((n: MCTSNode) => n.avg_reward), 0.001);

  const detailNode =
    selectedNodeId !== null
      ? trace.nodes.find((n: MCTSNode) => n.id === selectedNodeId)
      : null;

  return (
    <Card
      sx={{
        mb: 3,
        border: "1px solid rgba(124, 77, 255, 0.25)",
        bgcolor: "rgba(124, 77, 255, 0.02)",
      }}
    >
      <CardContent sx={{ pb: 1 }}>
        {/* Header */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <PsychologyIcon sx={{ color: "primary.main", fontSize: 22 }} />
            <Typography variant="subtitle2" fontWeight={700}>
              MCTS Query Rewriting
            </Typography>
            <Chip
              label={`${trace.iterations} iter`}
              size="small"
              sx={{ height: 20, fontSize: 10, bgcolor: "rgba(124,77,255,0.1)" }}
            />
          </Box>
          <IconButton
            size="small"
            onClick={() => setExpanded(!expanded)}
            sx={{
              transform: expanded ? "rotate(180deg)" : "none",
              transition: "transform 0.2s",
            }}
          >
            <ExpandMoreIcon fontSize="small" />
          </IconButton>
        </Box>

        {/* Summary: original → best */}
        <Box
          sx={{
            mt: 1.5,
            display: "flex",
            alignItems: "center",
            gap: 1,
            flexWrap: "wrap",
          }}
        >
          <QueryChip query={trace.original_query} variant="original" />
          <ArrowRightAltIcon
            sx={{ fontSize: 18, color: "text.secondary", mx: 0.5 }}
          />
          <QueryChip query={trace.best_query} variant="best" />
          {bestNode && (
            <Chip
              label={`+${((bestNode.avg_reward / (root.avg_reward || 0.001) - 1) * 100).toFixed(0)}% reward`}
              size="small"
              sx={{
                height: 20,
                fontSize: 10,
                fontWeight: 600,
                bgcolor:
                  bestNode.avg_reward > root.avg_reward
                    ? "rgba(76,175,80,0.15)"
                    : "rgba(255,152,0,0.15)",
                color:
                  bestNode.avg_reward > root.avg_reward
                    ? "#4caf50"
                    : "#ff9800",
              }}
            />
          )}
        </Box>
      </CardContent>

      <Collapse in={expanded}>
        <Divider sx={{ opacity: 0.1 }} />
        <Box sx={{ p: 2 }}>
          {/* Iteration rows */}
          {iterations.map((iterNodes, iterIdx) => (
            <Box key={iterIdx} sx={{ mb: iterIdx < iterations.length - 1 ? 2 : 0 }}>
              <Typography
                variant="overline"
                sx={{
                  fontSize: 10,
                  color: "text.secondary",
                  letterSpacing: 1.2,
                  mb: 0.5,
                  display: "block",
                }}
              >
                {iterIdx === 0 ? "Original Query" : `Iteration ${iterIdx}`}
              </Typography>

              <Box sx={{ display: "flex", gap: 1.5, flexWrap: "wrap" }}>
                {iterNodes.map((node) => (
                  <NodeCard
                    key={node.id}
                    node={node}
                    maxReward={maxReward}
                    isSelected={selectedNodeId === node.id}
                    onClick={() =>
                      setSelectedNodeId(
                        selectedNodeId === node.id ? null : node.id
                      )
                    }
                  />
                ))}
              </Box>
            </Box>
          ))}

          {/* Detail panel for selected node */}
          {detailNode && (
            <Box sx={{ mt: 2 }}>
              <Divider sx={{ opacity: 0.1, mb: 1.5 }} />
              <NodeDetailPanel node={detailNode} />
            </Box>
          )}
        </Box>
      </Collapse>
    </Card>
  );
}

// ------------------------------------------------------------------
// Sub-components
// ------------------------------------------------------------------

function QueryChip({
  query,
  variant,
}: {
  query: string;
  variant: "original" | "best";
}) {
  const isOriginal = variant === "original";
  return (
    <Typography
      variant="caption"
      sx={{
        fontFamily: "monospace",
        fontSize: "0.78rem",
        fontWeight: isOriginal ? 400 : 600,
        color: isOriginal ? "text.secondary" : "#4caf50",
        bgcolor: isOriginal
          ? "rgba(255,255,255,0.04)"
          : "rgba(76,175,80,0.08)",
        px: 1,
        py: 0.3,
        borderRadius: 1,
        border: isOriginal
          ? "1px solid rgba(255,255,255,0.08)"
          : "1px solid rgba(76,175,80,0.25)",
      }}
    >
      {isOriginal && "Q: "}
      {query}
    </Typography>
  );
}

// ------------------------------------------------------------------
// Reward decomposition bar
// ------------------------------------------------------------------

function RewardBar({ components, height = 6 }: { components: MCTSRewardComponents; height?: number }) {
  const total = components.bm25 + components.semantic + components.llm;
  if (total === 0) return null;

  const bm25Pct = (components.bm25 / total) * 100;
  const semPct = (components.semantic / total) * 100;
  const llmPct = (components.llm / total) * 100;

  return (
    <Tooltip
      title={`BM25: ${(components.bm25 * 100).toFixed(0)}% | Semantic: ${(components.semantic * 100).toFixed(0)}% | LLM: ${(components.llm * 100).toFixed(0)}%`}
      arrow
    >
      <Box
        sx={{
          display: "flex",
          height,
          borderRadius: height / 2,
          overflow: "hidden",
          width: "100%",
        }}
      >
        <Box sx={{ width: `${bm25Pct}%`, bgcolor: "#ff9800", transition: "width 0.3s" }} />
        <Box sx={{ width: `${semPct}%`, bgcolor: "#2196f3", transition: "width 0.3s" }} />
        <Box sx={{ width: `${llmPct}%`, bgcolor: "#9c27b0", transition: "width 0.3s" }} />
      </Box>
    </Tooltip>
  );
}

// ------------------------------------------------------------------

interface NodeCardProps {
  node: MCTSNode;
  maxReward: number;
  isSelected: boolean;
  onClick: () => void;
}

function NodeCard({
  node,
  maxReward,
  isSelected,
  onClick,
}: NodeCardProps) {
  const isBest = node.is_best;
  const isRoot = node.parent_id === null;
  const newHits = node.top_hits.filter((h) => h.is_new);
  const rewardPct = (node.avg_reward / maxReward) * 100;

  return (
    <Box
      onClick={onClick}
      sx={{
        flex: "1 1 220px",
        maxWidth: 340,
        minWidth: 200,
        p: 1.5,
        borderRadius: 1.5,
        cursor: "pointer",
        border: isSelected
          ? "1px solid rgba(124,77,255,0.6)"
          : isBest
            ? "1px solid rgba(76,175,80,0.4)"
            : "1px solid rgba(255,255,255,0.08)",
        bgcolor: isSelected
          ? "rgba(124,77,255,0.06)"
          : isBest
            ? "rgba(76,175,80,0.04)"
            : "rgba(255,255,255,0.02)",
        transition: "all 0.15s",
        "&:hover": {
          borderColor: isSelected
            ? "rgba(124,77,255,0.8)"
            : "rgba(124,77,255,0.3)",
          bgcolor: "rgba(124,77,255,0.04)",
        },
      }}
    >
      {/* Query text */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 0.75 }}>
        {isBest && !isRoot && (
          <StarIcon sx={{ fontSize: 14, color: "#4caf50", flexShrink: 0 }} />
        )}
        {isRoot && (
          <Chip
            label="ROOT"
            size="small"
            color="primary"
            sx={{ height: 18, fontSize: 9, fontWeight: 700, mr: 0.5 }}
          />
        )}
        <Typography
          variant="caption"
          sx={{
            fontFamily: "monospace",
            fontSize: "0.75rem",
            fontWeight: isBest ? 600 : 400,
            color: isBest ? "#4caf50" : "text.primary",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {node.query}
        </Typography>
      </Box>

      {/* Reward bar + value */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.75 }}>
        <Box sx={{ flex: 1 }}>
          <LinearProgress
            variant="determinate"
            value={rewardPct}
            sx={{
              height: 4,
              borderRadius: 2,
              bgcolor: "rgba(255,255,255,0.05)",
              "& .MuiLinearProgress-bar": {
                bgcolor: isBest ? "#4caf50" : "primary.main",
                borderRadius: 2,
              },
            }}
          />
        </Box>
        <Typography
          variant="caption"
          sx={{ fontSize: 10, fontFamily: "monospace", color: "text.secondary", minWidth: 35 }}
        >
          {node.avg_reward.toFixed(3)}
        </Typography>
      </Box>

      {/* Reward decomposition */}
      {node.reward_components && (
        <Box sx={{ mb: 0.75 }}>
          <RewardBar components={node.reward_components} height={4} />
          <Box sx={{ display: "flex", gap: 1, mt: 0.3 }}>
            <Typography variant="caption" sx={{ fontSize: 9, color: "#ff9800" }}>
              BM25
            </Typography>
            <Typography variant="caption" sx={{ fontSize: 9, color: "#2196f3" }}>
              Semantic
            </Typography>
            <Typography variant="caption" sx={{ fontSize: 9, color: "#9c27b0" }}>
              LLM
            </Typography>
          </Box>
        </Box>
      )}

      {/* Top entities found */}
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.4 }}>
        {node.top_hits.slice(0, 3).map((hit) => (
          <Tooltip
            key={hit.chunk_id}
            title={`${hit.signature}\nBM25: ${hit.bm25_score} | Cosine: ${hit.semantic_score.toFixed(3)}`}
            arrow
            placement="top"
          >
            <Chip
              icon={
                hit.is_new ? (
                  <FiberNewIcon sx={{ fontSize: "14px !important" }} />
                ) : undefined
              }
              label={shortName(hit.name)}
              size="small"
              sx={{
                height: 22,
                fontSize: 10,
                fontFamily: "monospace",
                maxWidth: 150,
                bgcolor: hit.is_new
                  ? "rgba(76,175,80,0.12)"
                  : "rgba(255,255,255,0.04)",
                borderColor: hit.is_new
                  ? "rgba(76,175,80,0.3)"
                  : "rgba(255,255,255,0.1)",
                color: hit.is_new ? "#4caf50" : "text.secondary",
                "& .MuiChip-label": {
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                },
              }}
              variant="outlined"
            />
          </Tooltip>
        ))}
        {newHits.length > 0 && !isRoot && (
          <Typography
            variant="caption"
            sx={{ fontSize: 10, color: "#4caf50", alignSelf: "center" }}
          >
            +{newHits.length} new
          </Typography>
        )}
      </Box>
    </Box>
  );
}

// ------------------------------------------------------------------
// Detail panel
// ------------------------------------------------------------------

function NodeDetailPanel({ node }: { node: MCTSNode }) {
  return (
    <Box>
      <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 1 }}>
        {node.parent_id === null ? "Original Query" : "Rewritten Query"} — Found Entities
      </Typography>

      <Typography
        variant="body2"
        sx={{
          fontFamily: "monospace",
          mb: 1.5,
          color: node.is_best ? "#4caf50" : "text.primary",
        }}
      >
        "{node.query}"
      </Typography>

      {/* Reward decomposition detail */}
      {node.reward_components && (
        <Box sx={{ mb: 1.5, display: "flex", gap: 2 }}>
          <RewardChip label="BM25" value={node.reward_components.bm25} color="#ff9800" />
          <RewardChip label="Semantic" value={node.reward_components.semantic} color="#2196f3" />
          <RewardChip label="LLM" value={node.reward_components.llm} color="#9c27b0" />
        </Box>
      )}

      {node.top_hits.length === 0 ? (
        <Typography variant="caption" color="text.secondary">
          No code entities matched this query variant.
        </Typography>
      ) : (
        <Box sx={{ display: "flex", flexDirection: "column", gap: 0.75 }}>
          {node.top_hits.map((hit, idx) => (
            <HitRow key={hit.chunk_id} hit={hit} rank={idx + 1} />
          ))}
        </Box>
      )}

      {/* Stats */}
      <Box
        sx={{
          mt: 1.5,
          pt: 1,
          borderTop: "1px solid rgba(255,255,255,0.06)",
          display: "flex",
          gap: 2,
        }}
      >
        <StatLabel label="Avg Reward" value={node.avg_reward.toFixed(4)} />
        <StatLabel label="Visits" value={String(node.visits)} />
        <StatLabel
          label="New Entities"
          value={String(node.top_hits.filter((h) => h.is_new).length)}
          highlight
        />
      </Box>
    </Box>
  );
}

function RewardChip({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 0.5,
        px: 1,
        py: 0.3,
        borderRadius: 1,
        bgcolor: `${color}15`,
        border: `1px solid ${color}30`,
      }}
    >
      <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: color }} />
      <Typography variant="caption" sx={{ fontSize: 10, color: "text.secondary" }}>
        {label}:
      </Typography>
      <Typography
        variant="caption"
        sx={{ fontSize: 11, fontWeight: 600, fontFamily: "monospace", color }}
      >
        {(value * 100).toFixed(0)}%
      </Typography>
    </Box>
  );
}

function HitRow({ hit, rank }: { hit: MCTSHit; rank: number }) {
  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 1,
        py: 0.5,
        px: 1,
        borderRadius: 1,
        bgcolor: hit.is_new
          ? "rgba(76,175,80,0.06)"
          : "rgba(255,255,255,0.02)",
        border: hit.is_new
          ? "1px solid rgba(76,175,80,0.15)"
          : "1px solid rgba(255,255,255,0.04)",
      }}
    >
      <Typography
        variant="caption"
        sx={{ fontFamily: "monospace", fontSize: 10, color: "text.secondary", minWidth: 16 }}
      >
        #{rank}
      </Typography>

      {hit.is_new && (
        <Chip
          label="NEW"
          size="small"
          sx={{
            height: 16,
            fontSize: 9,
            fontWeight: 700,
            bgcolor: "rgba(76,175,80,0.2)",
            color: "#4caf50",
          }}
        />
      )}

      <Chip
        label={hit.chunk_type}
        size="small"
        variant="outlined"
        sx={{ height: 18, fontSize: 9, borderColor: "rgba(255,255,255,0.1)" }}
      />

      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Typography
          variant="caption"
          fontWeight={600}
          sx={{
            fontFamily: "monospace",
            fontSize: "0.78rem",
            color: hit.is_new ? "#4caf50" : "text.primary",
          }}
          noWrap
        >
          {hit.name}
        </Typography>
        <Typography
          variant="caption"
          sx={{ display: "block", fontFamily: "monospace", fontSize: 10, color: "text.secondary" }}
          noWrap
        >
          {hit.file_path}
        </Typography>
      </Box>

      {/* Dual score */}
      <Box sx={{ display: "flex", gap: 1, flexShrink: 0, alignItems: "center" }}>
        <Tooltip title={`BM25 score: ${hit.bm25_score}`} arrow>
          <Typography
            variant="caption"
            sx={{ fontFamily: "monospace", fontSize: 10, color: "#ff9800" }}
          >
            B:{hit.bm25_score.toFixed(1)}
          </Typography>
        </Tooltip>
        <Tooltip title={`Cosine similarity: ${hit.semantic_score}`} arrow>
          <Typography
            variant="caption"
            sx={{ fontFamily: "monospace", fontSize: 10, color: "#2196f3" }}
          >
            S:{hit.semantic_score.toFixed(3)}
          </Typography>
        </Tooltip>
      </Box>
    </Box>
  );
}

function StatLabel({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <Box>
      <Typography
        variant="caption"
        sx={{ fontSize: 10, color: "text.secondary", display: "block" }}
      >
        {label}
      </Typography>
      <Typography
        variant="caption"
        sx={{
          fontFamily: "monospace",
          fontWeight: 600,
          fontSize: 12,
          color: highlight ? "#4caf50" : "text.primary",
        }}
      >
        {value}
      </Typography>
    </Box>
  );
}

// ------------------------------------------------------------------
// Utilities
// ------------------------------------------------------------------

function getIterationGroups(nodes: MCTSNode[]): MCTSNode[][] {
  const depthMap = new Map<number, number>();

  function computeDepth(node: MCTSNode): number {
    if (depthMap.has(node.id)) return depthMap.get(node.id)!;
    if (node.parent_id === null) {
      depthMap.set(node.id, 0);
      return 0;
    }
    const parent = nodes.find((n) => n.id === node.parent_id);
    const d = parent ? computeDepth(parent) + 1 : 0;
    depthMap.set(node.id, d);
    return d;
  }

  nodes.forEach(computeDepth);
  const maxDepth = Math.max(...Array.from(depthMap.values()), 0);
  const groups: MCTSNode[][] = [];
  for (let d = 0; d <= maxDepth; d++) {
    groups.push(nodes.filter((n) => depthMap.get(n.id) === d));
  }
  return groups;
}

function shortName(name: string): string {
  const parts = name.split(".");
  if (parts.length <= 2) return name;
  return parts.slice(-2).join(".");
}
