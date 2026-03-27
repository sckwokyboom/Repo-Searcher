import { useEffect, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  LinearProgress,
  Typography,
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import AutoFixHighIcon from "@mui/icons-material/AutoFixHigh";
import type { RepoInfo, LoRAStatus } from "../types";
import { getRepoStatus, indexRepo } from "../services/repoService";
import { getLoRAStatus } from "../services/loraService";
import { useIndexingProgress } from "../hooks/useIndexingProgress";
import { useCodeSearch } from "../hooks/useCodeSearch";
import { useLoRATraining } from "../hooks/useLoRATraining";
import IndexingProgress from "../components/IndexingProgress";
import CodeSearchBar from "../components/CodeSearchBar";
import SearchResultList from "../components/SearchResultList";
import CallGraphPanel from "../components/CallGraphPanel";
import LoRATrainingPanel from "../components/LoRATrainingPanel";

export default function RepoPage() {
  const { owner, name } = useParams<{ owner: string; name: string }>();
  const fullName = `${owner}/${name}`;
  const repoId = `${owner}__${name}`;

  const [status, setStatus] = useState<string | null>(null);
  const [repoInfo, setRepoInfo] = useState<RepoInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [isIndexing, setIsIndexing] = useState(false);
  const [selectedMethod, setSelectedMethod] = useState<string | null>(null);
  const [loraStatus, setLoraStatus] = useState<LoRAStatus | null>(null);
  const [isLoRATraining, setIsLoRATraining] = useState(false);

  const { progress } = useIndexingProgress(isIndexing ? repoId : null);
  const { response, loading: searching, error, search } = useCodeSearch(repoId);
  const { progress: loraProgress } = useLoRATraining(status === "done" ? repoId : null);

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

  const fetchLoRAStatus = useCallback(async () => {
    try {
      const data = await getLoRAStatus(repoId);
      setLoraStatus(data);
      setIsLoRATraining(data.is_training);
    } catch {
      // Repo might not be indexed yet
    }
  }, [repoId]);

  useEffect(() => {
    checkStatus();
  }, [checkStatus]);

  useEffect(() => {
    if (status === "done") {
      fetchLoRAStatus();
    }
  }, [status, fetchLoRAStatus]);

  useEffect(() => {
    if (progress?.step === "done") {
      setStatus("done");
      setIsIndexing(false);
      checkStatus();
    }
  }, [progress?.step, checkStatus]);

  // Watch LoRA training completion
  useEffect(() => {
    if (loraProgress?.step === "done" || loraProgress?.step === "failed" || loraProgress?.step === "cancelled") {
      setIsLoRATraining(false);
      fetchLoRAStatus();
    }
  }, [loraProgress?.step, fetchLoRAStatus]);

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
      has_lora_adapter: false,
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
        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 0.5 }}>
          <Typography variant="h5">
            {fullName}
          </Typography>
          {loraStatus?.has_adapter && (
            <Chip
              icon={<AutoFixHighIcon sx={{ fontSize: "16px !important" }} />}
              label="LoRA"
              size="small"
              sx={{
                height: 24,
                fontSize: 11,
                fontWeight: 700,
                bgcolor: "rgba(124,77,255,0.12)",
                color: "#7c4dff",
                "& .MuiChip-icon": { color: "#7c4dff" },
              }}
            />
          )}
        </Box>
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
          {/* LoRA Training Panel */}
          <LoRATrainingPanel
            repoId={repoId}
            loraStatus={loraStatus}
            trainingProgress={loraProgress}
            onTrainingStarted={() => setIsLoRATraining(true)}
            onTrainingDone={fetchLoRAStatus}
          />

          {!isLoRATraining && (
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
            </>
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
