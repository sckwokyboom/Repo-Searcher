import { useState } from "react";
import {
  Box,
  Card,
  CardContent,
  Chip,
  Collapse,
  IconButton,
  Tooltip,
  Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import AccountTreeIcon from "@mui/icons-material/AccountTree";
import InsertDriveFileIcon from "@mui/icons-material/InsertDriveFile";
import type { SearchResult } from "../types";
import CodeBlock from "./CodeBlock";

interface Props {
  result: SearchResult;
  rank: number;
  onShowGraph: (methodId: string) => void;
}

export default function SearchResultCard({ result, rank, onShowGraph }: Props) {
  const [expanded, setExpanded] = useState(rank === 0);
  const { chunk } = result;

  const displayName = chunk.class_name
    ? `${chunk.class_name}.${chunk.method_name}`
    : chunk.method_name || chunk.chunk_id;

  return (
    <Card
      sx={{
        mb: 2,
        transition: "all 0.2s",
        "&:hover": {
          borderColor: "rgba(124, 77, 255, 0.3)",
        },
      }}
    >
      <CardContent sx={{ pb: expanded ? 0 : undefined }}>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, flex: 1, minWidth: 0 }}>
            <Chip
              label={`#${rank + 1}`}
              size="small"
              color="primary"
              sx={{ fontWeight: 700, minWidth: 40 }}
            />
            <Typography variant="subtitle1" fontWeight={600} noWrap>
              {displayName}
            </Typography>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            <Chip
              label={`${result.score.toFixed(1)}`}
              size="small"
              variant="outlined"
              color="secondary"
              sx={{ fontFamily: "monospace" }}
            />
            <Tooltip title="Show Call Graph">
              <IconButton
                size="small"
                onClick={() => onShowGraph(chunk.chunk_id)}
                color="primary"
              >
                <AccountTreeIcon fontSize="small" />
              </IconButton>
            </Tooltip>
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
        </Box>

        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1,
            mt: 0.5,
            mb: expanded ? 1.5 : 0,
          }}
        >
          <InsertDriveFileIcon
            sx={{ fontSize: 14, color: "text.secondary" }}
          />
          <Typography
            variant="caption"
            color="text.secondary"
            fontFamily="monospace"
          >
            {chunk.file_path}:{chunk.start_line}-{chunk.end_line}
          </Typography>
          {chunk.chunk_type === "method" && (
            <Chip label="method" size="small" variant="outlined" sx={{ height: 20, fontSize: 11 }} />
          )}
          {result.bm25_rank && (
            <Typography variant="caption" color="text.secondary">
              BM25: #{result.bm25_rank}
            </Typography>
          )}
          {result.vector_rank && (
            <Typography variant="caption" color="text.secondary">
              Vec: #{result.vector_rank}
            </Typography>
          )}
        </Box>
      </CardContent>

      <Collapse in={expanded}>
        <Box sx={{ px: 2, pb: 2 }}>
          {chunk.javadoc && (
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{
                mb: 1,
                p: 1.5,
                borderRadius: 1,
                bgcolor: "rgba(255,255,255,0.02)",
                fontStyle: "italic",
                fontSize: "0.82rem",
              }}
            >
              {chunk.javadoc.slice(0, 300)}
            </Typography>
          )}
          <CodeBlock
            code={chunk.body}
            startLine={chunk.start_line}
          />
          {(result.callers.length > 0 || result.callees.length > 0) && (
            <Box sx={{ mt: 1.5, display: "flex", gap: 2, flexWrap: "wrap" }}>
              {result.callers.length > 0 && (
                <Typography variant="caption" color="text.secondary">
                  Called by: {result.callers.slice(0, 3).map(c => c.split("::").pop()).join(", ")}
                  {result.callers.length > 3 && ` +${result.callers.length - 3} more`}
                </Typography>
              )}
              {result.callees.length > 0 && (
                <Typography variant="caption" color="text.secondary">
                  Calls: {result.callees.slice(0, 3).map(c => c.split("::").pop()).join(", ")}
                  {result.callees.length > 3 && ` +${result.callees.length - 3} more`}
                </Typography>
              )}
            </Box>
          )}
        </Box>
      </Collapse>
    </Card>
  );
}
