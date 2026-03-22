import { Box, Chip, Typography } from "@mui/material";
import type { SearchResponse } from "../types";
import SearchResultCard from "./SearchResultCard";

interface Props {
  response: SearchResponse;
  onShowGraph: (methodId: string) => void;
}

export default function SearchResultList({ response, onShowGraph }: Props) {
  return (
    <Box>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          mb: 2,
        }}
      >
        <Typography variant="body2" color="text.secondary">
          {response.results.length} results in {response.search_time_ms.toFixed(0)}ms
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
