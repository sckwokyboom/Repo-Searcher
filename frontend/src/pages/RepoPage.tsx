import { useEffect, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  LinearProgress,
  Typography,
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import type { RepoInfo } from "../types";
import { getRepoStatus, indexRepo } from "../services/repoService";
import { useIndexingProgress } from "../hooks/useIndexingProgress";
import { useCodeSearch } from "../hooks/useCodeSearch";
import IndexingProgress from "../components/IndexingProgress";
import CodeSearchBar from "../components/CodeSearchBar";
import SearchResultList from "../components/SearchResultList";
import CallGraphPanel from "../components/CallGraphPanel";

export default function RepoPage() {
  const { owner, name } = useParams<{ owner: string; name: string }>();
  const fullName = `${owner}/${name}`;
  const repoId = `${owner}__${name}`;

  const [status, setStatus] = useState<string | null>(null);
  const [repoInfo, setRepoInfo] = useState<RepoInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [isIndexing, setIsIndexing] = useState(false);
  const [selectedMethod, setSelectedMethod] = useState<string | null>(null);

  const { progress } = useIndexingProgress(isIndexing ? repoId : null);
  const { response, loading: searching, error, search } = useCodeSearch(repoId);

  const checkStatus = useCallback(async () => {
    try {
      const data = await getRepoStatus(repoId);
      setStatus(data.status);
      setRepoInfo(data.repo_info);
      if (data.status === "done") {
        setIsIndexing(false);
      }
    } catch {
      setStatus(null);
    } finally {
      setLoading(false);
    }
  }, [repoId]);

  useEffect(() => {
    checkStatus();
  }, [checkStatus]);

  useEffect(() => {
    if (progress?.step === "done") {
      setStatus("done");
      setIsIndexing(false);
      checkStatus();
    }
  }, [progress?.step, checkStatus]);

  const handleStartIndex = async () => {
    const repo: RepoInfo = {
      repo_id: repoId,
      owner: owner!,
      name: name!,
      full_name: fullName,
      description: null,
      stars: 0,
      url: `https://github.com/${fullName}`,
      language: "Java",
      indexed_at: null,
      chunk_count: 0,
    };

    try {
      await indexRepo(repo);
      setIsIndexing(true);
    } catch (err: any) {
      if (err?.response?.status === 409) {
        checkStatus();
      }
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", py: 8 }}>
        <CircularProgress />
      </Box>
    );
  }

  const isReady = status === "done";

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h5" sx={{ mb: 0.5 }}>
          {fullName}
        </Typography>
        {repoInfo?.description && (
          <Typography variant="body2" color="text.secondary">
            {repoInfo.description}
          </Typography>
        )}
      </Box>

      {/* Not indexed yet */}
      {!isReady && !isIndexing && (
        <Box sx={{ textAlign: "center", py: 6 }}>
          <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
            This repository hasn't been indexed yet
          </Typography>
          <Button
            variant="contained"
            size="large"
            startIcon={<PlayArrowIcon />}
            onClick={handleStartIndex}
            sx={{
              px: 4,
              py: 1.5,
              background: "linear-gradient(135deg, #7c4dff 0%, #651fff 100%)",
            }}
          >
            Start Indexing
          </Button>
        </Box>
      )}

      {/* Indexing in progress */}
      {isIndexing && <IndexingProgress progress={progress} />}

      {/* Ready for search */}
      {isReady && (
        <>
          <Box sx={{ mb: 3 }}>
            <CodeSearchBar onSearch={search} loading={searching} />
          </Box>

          {searching && <LinearProgress sx={{ mb: 2 }} />}

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {response && (
            <SearchResultList
              response={response}
              onShowGraph={(id) =>
                setSelectedMethod(selectedMethod === id ? null : id)
              }
            />
          )}

          {selectedMethod && (
            <CallGraphPanel
              repoId={repoId}
              methodId={selectedMethod}
              onClose={() => setSelectedMethod(null)}
            />
          )}
        </>
      )}
    </Box>
  );
}
