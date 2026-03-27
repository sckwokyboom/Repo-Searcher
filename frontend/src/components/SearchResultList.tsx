import { Box, Chip, Typography } from "@mui/material";
import AutoFixHighIcon from "@mui/icons-material/AutoFixHigh";
import ArrowRightAltIcon from "@mui/icons-material/ArrowRightAlt";
import SearchIcon from "@mui/icons-material/Search";
import CodeIcon from "@mui/icons-material/Code";
import AccountTreeIcon from "@mui/icons-material/AccountTree";
import ApiIcon from "@mui/icons-material/Api";
import type { SearchResponse, RewriteDetails } from "../types";
import SearchResultCard from "./SearchResultCard";

interface Props {
  response: SearchResponse;
  onShowGraph: (methodId: string) => void;
}

function RewriteDetailsCard({ details, originalQuery }: { details: RewriteDetails; originalQuery: string }) {
  return (
    <Box
      sx={{
        mb: 2.5,
        p: 2,
        borderRadius: 2,
        bgcolor: "rgba(124,77,255,0.03)",
        border: "1px solid rgba(124,77,255,0.15)",
      }}
    >
      {/* Header: Query Analysis + intent/scope badges */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5, flexWrap: "wrap" }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 700, color: "text.primary", fontSize: 13 }}>
          Query Analysis
        </Typography>
        {details.intent && (
          <Chip
            label={details.intent.replace("_", " ")}
            size="small"
            sx={{
              height: 20, fontSize: 10, fontWeight: 600,
              bgcolor: "rgba(124,77,255,0.1)", color: "#7c4dff",
            }}
          />
        )}
        {details.search_scope && (
          <Chip
            label={details.search_scope}
            size="small"
            sx={{
              height: 20, fontSize: 10, fontWeight: 600,
              bgcolor: "rgba(0,229,255,0.08)", color: "#00e5ff",
            }}
          />
        )}
        <Box sx={{ flex: 1 }} />
        <Typography variant="caption" sx={{ color: "text.secondary", fontStyle: "italic", fontSize: 11 }}>
          &laquo;{originalQuery}&raquo;
        </Typography>
      </Box>

      {/* Keywords */}
      {details.keywords.length > 0 && (
        <FieldRow
          icon={<SearchIcon sx={{ fontSize: 14, color: "text.secondary" }} />}
          label="Keywords"
          items={details.keywords}
          chipSx={{ bgcolor: "rgba(255,255,255,0.06)", color: "text.secondary" }}
        />
      )}

      {/* Project terms */}
      {details.project_terms.length > 0 && (
        <FieldRow
          icon={<AccountTreeIcon sx={{ fontSize: 14, color: "#4caf50" }} />}
          label="Project"
          items={details.project_terms}
          chipSx={{ bgcolor: "rgba(76,175,80,0.1)", color: "#4caf50", fontFamily: "monospace" }}
        />
      )}

      {/* Method hints */}
      {details.method_hints.length > 0 && (
        <FieldRow
          icon={<CodeIcon sx={{ fontSize: 14, color: "#ff9800" }} />}
          label="Methods"
          items={details.method_hints}
          chipSx={{ bgcolor: "rgba(255,152,0,0.1)", color: "#ff9800", fontFamily: "monospace" }}
        />
      )}

      {/* API hints */}
      {details.api_hints.length > 0 && (
        <FieldRow
          icon={<ApiIcon sx={{ fontSize: 14, color: "#2196f3" }} />}
          label="API"
          items={details.api_hints}
          chipSx={{ bgcolor: "rgba(33,150,243,0.1)", color: "#2196f3", fontFamily: "monospace" }}
        />
      )}

      {/* Search queries */}
      {details.search_queries.length > 0 && (
        <Box sx={{ mt: 1.5, pt: 1, borderTop: "1px solid rgba(255,255,255,0.06)" }}>
          <Typography variant="caption" sx={{ color: "text.secondary", fontWeight: 600, fontSize: 11, mb: 0.5, display: "block" }}>
            Search Queries
          </Typography>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
            {details.search_queries.map((q, i) => (
              <Box key={i} sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                <Typography
                  variant="caption"
                  sx={{ color: "text.secondary", fontSize: 10, width: 16, textAlign: "right", opacity: 0.5 }}
                >
                  {i + 1}.
                </Typography>
                <Typography
                  variant="caption"
                  sx={{ fontFamily: "monospace", fontSize: 12, color: "#b0bec5" }}
                >
                  {q}
                </Typography>
              </Box>
            ))}
          </Box>
        </Box>
      )}
    </Box>
  );
}

function FieldRow({
  icon,
  label,
  items,
  chipSx,
}: {
  icon: React.ReactNode;
  label: string;
  items: string[];
  chipSx: object;
}) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, mb: 0.75, flexWrap: "wrap" }}>
      {icon}
      <Typography variant="caption" sx={{ color: "text.secondary", fontWeight: 600, fontSize: 11, minWidth: 50 }}>
        {label}
      </Typography>
      {items.map((item) => (
        <Chip
          key={item}
          label={item}
          size="small"
          sx={{ height: 22, fontSize: 11, fontWeight: 500, ...chipSx }}
        />
      ))}
    </Box>
  );
}

/** Fallback: simple "old → new" display when no structured details */
function SimplRewriteInfo({ query, rewrittenQuery }: { query: string; rewrittenQuery: string }) {
  return (
    <Box
      sx={{
        mb: 2,
        p: 1.5,
        borderRadius: 1.5,
        bgcolor: "rgba(76,175,80,0.04)",
        border: "1px solid rgba(76,175,80,0.15)",
        display: "flex",
        alignItems: "center",
        gap: 1,
        flexWrap: "wrap",
      }}
    >
      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
        Query rewritten:
      </Typography>
      <Typography
        variant="caption"
        sx={{
          fontFamily: "monospace", fontSize: "0.78rem",
          color: "text.secondary", textDecoration: "line-through", opacity: 0.6,
        }}
      >
        {query}
      </Typography>
      <ArrowRightAltIcon sx={{ fontSize: 16, color: "text.secondary" }} />
      <Typography
        variant="caption"
        sx={{ fontFamily: "monospace", fontSize: "0.78rem", fontWeight: 600, color: "#4caf50" }}
      >
        {rewrittenQuery}
      </Typography>
    </Box>
  );
}

export default function SearchResultList({ response, onShowGraph }: Props) {
  const hasStructuredRewrite = response.rewrite_details != null;
  const wasRewritten =
    !hasStructuredRewrite &&
    response.rewritten_query &&
    response.rewritten_query !== response.query;

  return (
    <Box>
      {/* Structured rewrite details */}
      {hasStructuredRewrite && (
        <RewriteDetailsCard details={response.rewrite_details!} originalQuery={response.query} />
      )}

      {/* Fallback: simple rewrite display */}
      {wasRewritten && (
        <SimplRewriteInfo query={response.query} rewrittenQuery={response.rewritten_query!} />
      )}

      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          mb: 2,
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Typography variant="body2" color="text.secondary">
            {response.results.length} results in {response.search_time_ms.toFixed(0)}ms
          </Typography>
          {response.lora_active && (
            <Chip
              icon={<AutoFixHighIcon sx={{ fontSize: "14px !important" }} />}
              label="LoRA Enhanced"
              size="small"
              sx={{
                height: 22,
                fontSize: 10,
                fontWeight: 600,
                bgcolor: "rgba(124,77,255,0.12)",
                color: "#7c4dff",
                "& .MuiChip-icon": { color: "#7c4dff" },
              }}
            />
          )}
        </Box>
        {/* Show keywords from rewrite_details or fallback to expanded_keywords */}
        {!hasStructuredRewrite && response.expanded_keywords.length > 0 && (
          <Box sx={{ display: "flex", gap: 0.5, flexWrap: "wrap" }}>
            <Typography variant="caption" color="text.secondary" sx={{ mr: 0.5 }}>
              Keywords:
            </Typography>
            {response.expanded_keywords.slice(0, 5).map((kw) => (
              <Chip
                key={kw}
                label={kw}
                size="small"
                variant="outlined"
                sx={{ height: 22, fontSize: 11 }}
              />
            ))}
          </Box>
        )}
      </Box>

      {/* Search results */}
      {response.results.map((result, index) => (
        <SearchResultCard
          key={result.chunk.chunk_id}
          result={result}
          rank={index}
          onShowGraph={onShowGraph}
        />
      ))}

      {response.results.length === 0 && (
        <Box sx={{ textAlign: "center", py: 6 }}>
          <Typography color="text.secondary">
            No results found. Try rephrasing your query.
          </Typography>
        </Box>
      )}
    </Box>
  );
}
