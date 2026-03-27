import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Box, Grid, Typography } from "@mui/material";
import SearchIcon from "@mui/icons-material/Search";
import type { GitHubRepo, RepoInfo } from "../types";
import RepoSearchBar from "../components/RepoSearchBar";
import RepoCard from "../components/RepoCard";
import { getIndexedRepos } from "../services/repoService";

export default function HomePage() {
  const navigate = useNavigate();
  const [indexedRepos, setIndexedRepos] = useState<RepoInfo[]>([]);

  useEffect(() => {
    getIndexedRepos()
      .then(setIndexedRepos)
      .catch(() => {});
  }, []);

  const handleSelect = (repo: GitHubRepo) => {
    navigate(`/repo/${repo.full_name}`);
  };

  return (
    <Box>
      {/* Hero section */}
      <Box sx={{ textAlign: "center", mb: 6, mt: 4 }}>
        <Box
          sx={{
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            width: 72,
            height: 72,
            borderRadius: "50%",
            background: "linear-gradient(135deg, rgba(124, 77, 255, 0.2) 0%, rgba(0, 229, 255, 0.2) 100%)",
            mb: 3,
          }}
        >
          <SearchIcon sx={{ fontSize: 36, color: "primary.main" }} />
        </Box>
        <Typography
          variant="h4"
          sx={{
            mb: 1,
            background: "linear-gradient(135deg, #e2e8f0 0%, #94a3b8 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          Search Code with Natural Language
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 4, maxWidth: 600, mx: "auto" }}>
          Find relevant code in Java repositories using natural language queries.
          Powered by hybrid BM25 + vector search with call graph analysis.
        </Typography>
        <Box sx={{ maxWidth: 640, mx: "auto" }}>
          <RepoSearchBar onSelect={handleSelect} />
        </Box>
      </Box>

      {/* Indexed repos */}
      {indexedRepos.length > 0 && (
        <Box>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Recently Indexed
          </Typography>
          <Grid container spacing={2}>
            {indexedRepos.map((repo) => (
              <Grid size={{ xs: 12, sm: 6, md: 4 }} key={repo.repo_id}>
                <RepoCard
                  repo={repo}
                  onClick={() => navigate(`/repo/${repo.full_name}`)}
                />
              </Grid>
            ))}
          </Grid>
        </Box>
      )}
    </Box>
  );
}
