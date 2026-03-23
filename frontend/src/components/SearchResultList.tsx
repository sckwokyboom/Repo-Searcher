import { Box, Chip, Typography } from "@mui/material";
import AccountTreeIcon from "@mui/icons-material/AccountTree";
import type { SearchResponse } from "../types";
import SearchResultCard from "./SearchResultCard";
import MCTSTreePanel from "./MCTSTreePanel";

interface Props {
  response: SearchResponse;
  onShowGraph: (methodId: string) => void;
}

export default function SearchResultList({ response, onShowGraph }: Props) {
  const searchResults = response.results.filter((r) => r.source !== "graph_mcts");
  const graphResults = response.results.filter((r) => r.source === "graph_mcts");

  return (
    <Box>
      {/* MCTS Tree Visualization */}
      {response.mcts_trace && (
        <MCTSTreePanel trace={response.mcts_trace} />
      )}

      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          mb: 2,
        }}
      >
        <Typography variant="body2" color="text.secondary">
          {searchResults.length} results
          {graphResults.length > 0 && ` + ${graphResults.length} via graph`}
          {" "}in {response.search_time_ms.toFixed(0)}ms
        </Typography>
        {response.expanded_keywords.length > 0 && (
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

      {/* Direct search results */}
      {searchResults.map((result, index) => (
        <SearchResultCard
          key={result.chunk.chunk_id}
          result={result}
          rank={index}
          onShowGraph={onShowGraph}
        />
      ))}

      {/* Graph MCTS discoveries */}
      {graphResults.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
            <AccountTreeIcon sx={{ fontSize: 18, color: "#ff9800" }} />
            <Typography variant="subtitle2" fontWeight={600} sx={{ color: "#ff9800" }}>
              Discovered via Call Graph
            </Typography>
            {response.graph_mcts_trace && (
              <Typography variant="caption" color="text.secondary">
                ({response.graph_mcts_trace.total_nodes_visited} nodes explored)
              </Typography>
            )}
          </Box>
          {graphResults.map((result, index) => (
            <SearchResultCard
              key={result.chunk.chunk_id}
              result={result}
              rank={searchResults.length + index}
              onShowGraph={onShowGraph}
            />
          ))}
        </Box>
      )}

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
