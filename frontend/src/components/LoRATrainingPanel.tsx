import { useEffect, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Divider,
  LinearProgress,
  MenuItem,
  Select,
  Typography,
} from "@mui/material";
import AutoFixHighIcon from "@mui/icons-material/AutoFixHigh";
import CancelIcon from "@mui/icons-material/Cancel";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import SchoolIcon from "@mui/icons-material/School";
import SwapHorizIcon from "@mui/icons-material/SwapHoriz";
import type { LoRATrainingProgress, LoRAStatus, LoRAAdapterInfo } from "../types";
import {
  startLoRATraining,
  cancelLoRATraining,
  getAvailableAdapters,
  selectAdapter,
} from "../services/loraService";

interface Props {
  repoId: string;
  loraStatus: LoRAStatus | null;
  trainingProgress: LoRATrainingProgress | null;
  onTrainingStarted: () => void;
  onTrainingDone: () => void;
}

export default function LoRATrainingPanel({
  repoId,
  loraStatus,
  trainingProgress,
  onTrainingStarted,
  onTrainingDone,
}: Props) {
  const [confirming, setConfirming] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [adapters, setAdapters] = useState<LoRAAdapterInfo[]>([]);
  const [showPicker, setShowPicker] = useState(false);
  const [selectedAdapterId, setSelectedAdapterId] = useState<string>("");
  const [applying, setApplying] = useState(false);

  const isTraining = loraStatus?.is_training || false;
  const hasAdapter = loraStatus?.has_adapter || false;
  const activeAdapterId = loraStatus?.active_adapter_id || null;
  const step = trainingProgress?.step;
  const isDone = step === "done";
  const isFailed = step === "failed";
  const isCancelled = step === "cancelled";

  // Load available adapters when picker is opened
  useEffect(() => {
    if (showPicker && adapters.length === 0) {
      getAvailableAdapters()
        .then(setAdapters)
        .catch(() => {});
    }
  }, [showPicker, adapters.length]);

  // Pre-select current adapter in dropdown
  useEffect(() => {
    if (activeAdapterId && !selectedAdapterId) {
      setSelectedAdapterId(activeAdapterId);
    }
  }, [activeAdapterId, selectedAdapterId]);

  const handleStart = async () => {
    setStarting(true);
    setError(null);
    try {
      await startLoRATraining(repoId);
      onTrainingStarted();
      setConfirming(false);
    } catch (err: any) {
      setError(err?.response?.data?.detail || "Failed to start training");
    } finally {
      setStarting(false);
    }
  };

  const handleCancel = async () => {
    setCancelling(true);
    try {
      await cancelLoRATraining(repoId);
    } catch {
      // ignore
    }
  };

  const handleApplyAdapter = async () => {
    setApplying(true);
    setError(null);
    try {
      await selectAdapter(repoId, selectedAdapterId || null);
      onTrainingDone(); // refresh status
      setShowPicker(false);
    } catch (err: any) {
      setError(err?.response?.data?.detail || "Failed to apply adapter");
    } finally {
      setApplying(false);
    }
  };

  const handleRemoveAdapter = async () => {
    setApplying(true);
    try {
      await selectAdapter(repoId, null);
      onTrainingDone(); // refresh status
    } catch {
      // ignore
    } finally {
      setApplying(false);
    }
  };

  // Training in progress
  if (isTraining && trainingProgress) {
    const progress = trainingProgress.progress * 100;
    const eta = trainingProgress.estimated_time_remaining_sec;
    const etaText = eta != null
      ? eta > 60
        ? `~${Math.ceil(eta / 60)} min remaining`
        : `~${eta}s remaining`
      : "";

    return (
      <Card
        sx={{
          mb: 3,
          border: "1px solid rgba(124, 77, 255, 0.25)",
          bgcolor: "rgba(124, 77, 255, 0.02)",
        }}
      >
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
            <SchoolIcon sx={{ color: "primary.main", fontSize: 20 }} />
            <Typography variant="subtitle2" fontWeight={700}>
              Training LoRA Adapter
            </Typography>
            {trainingProgress.step === "training" && trainingProgress.epoch > 0 && (
              <Chip
                label={`Epoch ${trainingProgress.epoch}/${trainingProgress.total_epochs}`}
                size="small"
                sx={{ height: 20, fontSize: 10, bgcolor: "rgba(124,77,255,0.1)" }}
              />
            )}
          </Box>

          <LinearProgress
            variant="determinate"
            value={progress}
            sx={{
              mb: 1,
              height: 6,
              borderRadius: 3,
              bgcolor: "rgba(255,255,255,0.05)",
              "& .MuiLinearProgress-bar": {
                borderRadius: 3,
                background: "linear-gradient(90deg, #7c4dff 0%, #651fff 100%)",
              },
            }}
          />

          <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                {trainingProgress.message}
              </Typography>
              {trainingProgress.train_loss != null && (
                <Typography
                  variant="caption"
                  sx={{ ml: 1.5, fontFamily: "monospace", fontSize: 10, color: "text.secondary" }}
                >
                  Loss: {trainingProgress.train_loss.toFixed(4)}
                </Typography>
              )}
              {etaText && (
                <Typography variant="caption" sx={{ ml: 1.5, color: "text.secondary" }}>
                  {etaText}
                </Typography>
              )}
            </Box>
            <Button
              size="small"
              color="error"
              startIcon={<CancelIcon />}
              onClick={handleCancel}
              disabled={cancelling}
              sx={{ textTransform: "none" }}
            >
              {cancelling ? "Cancelling..." : "Cancel"}
            </Button>
          </Box>
        </CardContent>
      </Card>
    );
  }

  // Failed or cancelled state
  if (isFailed || isCancelled) {
    return (
      <Alert
        severity={isCancelled ? "info" : "error"}
        sx={{ mb: 2 }}
        action={
          <Button
            size="small"
            onClick={() => setConfirming(true)}
            sx={{ textTransform: "none" }}
          >
            Retry
          </Button>
        }
      >
        {trainingProgress?.message || (isCancelled ? "Training was cancelled" : "Training failed")}
      </Alert>
    );
  }

  // Adapter active — show status with option to change
  if (isDone || hasAdapter) {
    const activeAdapter = adapters.find((a) => a.adapter_id === activeAdapterId);
    const adapterLabel = activeAdapter?.name || activeAdapterId?.split(":")[1] || "Active";

    return (
      <Card
        sx={{
          mb: 3,
          border: "1px solid rgba(76,175,80,0.2)",
          bgcolor: "rgba(76,175,80,0.02)",
        }}
      >
        <CardContent sx={{ py: 1.5, "&:last-child": { pb: 1.5 } }}>
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <Chip
                icon={<CheckCircleIcon />}
                label={adapterLabel}
                size="small"
                sx={{
                  bgcolor: "rgba(76,175,80,0.12)",
                  color: "#4caf50",
                  fontWeight: 600,
                  "& .MuiChip-icon": { color: "#4caf50" },
                }}
              />
              <Typography variant="caption" color="text.secondary">
                LoRA-enhanced search active
              </Typography>
            </Box>
            <Box sx={{ display: "flex", gap: 0.5 }}>
              <Button
                size="small"
                startIcon={<SwapHorizIcon />}
                onClick={() => { setShowPicker(true); setAdapters([]); }}
                sx={{ textTransform: "none", fontSize: 12 }}
              >
                Change
              </Button>
              <Button
                size="small"
                color="error"
                onClick={handleRemoveAdapter}
                disabled={applying}
                sx={{ textTransform: "none", fontSize: 12 }}
              >
                Remove
              </Button>
            </Box>
          </Box>

          {/* Adapter picker inline */}
          {showPicker && (
            <>
              <Divider sx={{ my: 1.5, opacity: 0.15 }} />
              <AdapterPicker
                adapters={adapters}
                selectedId={selectedAdapterId}
                activeId={activeAdapterId}
                onSelect={setSelectedAdapterId}
                onApply={handleApplyAdapter}
                onCancel={() => setShowPicker(false)}
                onTrain={() => { setShowPicker(false); setConfirming(true); }}
                applying={applying}
                error={error}
                estimatedMinutes={loraStatus?.estimated_minutes}
              />
            </>
          )}
        </CardContent>
      </Card>
    );
  }

  // Training confirmation
  if (confirming) {
    return (
      <Card
        sx={{
          mb: 3,
          border: "1px solid rgba(124, 77, 255, 0.25)",
          bgcolor: "rgba(124, 77, 255, 0.02)",
        }}
      >
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
            <SchoolIcon sx={{ color: "primary.main", fontSize: 20 }} />
            <Typography variant="subtitle2" fontWeight={700}>
              Train LoRA Adapter for This Project?
            </Typography>
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            This will fine-tune a project-specific query rewriting model to improve
            search quality for this repository.
            {loraStatus?.estimated_minutes != null && (
              <> Estimated time: <strong>~{loraStatus.estimated_minutes} min</strong>.</>
            )}
          </Typography>

          {error && (
            <Alert severity="error" sx={{ mb: 1.5 }}>
              {error}
            </Alert>
          )}

          <Box sx={{ display: "flex", gap: 1 }}>
            <Button
              variant="contained"
              size="small"
              startIcon={<AutoFixHighIcon />}
              onClick={handleStart}
              disabled={starting}
              sx={{
                textTransform: "none",
                background: "linear-gradient(135deg, #7c4dff 0%, #651fff 100%)",
              }}
            >
              {starting ? "Starting..." : "Start Training"}
            </Button>
            <Button
              size="small"
              onClick={() => setConfirming(false)}
              sx={{ textTransform: "none" }}
            >
              Cancel
            </Button>
          </Box>
        </CardContent>
      </Card>
    );
  }

  // Default: show adapter picker + train button
  return (
    <Card
      sx={{
        mb: 3,
        border: "1px solid rgba(124, 77, 255, 0.15)",
        bgcolor: "rgba(124, 77, 255, 0.02)",
      }}
    >
      <CardContent sx={{ py: 1.5, "&:last-child": { pb: 1.5 } }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: showPicker ? 1.5 : 0 }}>
          <AutoFixHighIcon sx={{ color: "primary.main", fontSize: 18 }} />
          <Typography variant="subtitle2" fontWeight={600}>
            LoRA Adapter
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ flex: 1 }}>
            Enhance search with a fine-tuned model
          </Typography>
          {!showPicker && (
            <Button
              size="small"
              onClick={() => setShowPicker(true)}
              sx={{ textTransform: "none", fontSize: 12 }}
            >
              Choose adapter
            </Button>
          )}
        </Box>

        {showPicker && (
          <AdapterPicker
            adapters={adapters}
            selectedId={selectedAdapterId}
            activeId={activeAdapterId}
            onSelect={setSelectedAdapterId}
            onApply={handleApplyAdapter}
            onCancel={() => setShowPicker(false)}
            onTrain={() => { setShowPicker(false); setConfirming(true); }}
            applying={applying}
            error={error}
            estimatedMinutes={loraStatus?.estimated_minutes}
          />
        )}
      </CardContent>
    </Card>
  );
}


// --- Sub-component: Adapter picker ---

interface AdapterPickerProps {
  adapters: LoRAAdapterInfo[];
  selectedId: string;
  activeId: string | null;
  onSelect: (id: string) => void;
  onApply: () => void;
  onCancel: () => void;
  onTrain: () => void;
  applying: boolean;
  error: string | null;
  estimatedMinutes: number | null | undefined;
}

function AdapterPicker({
  adapters,
  selectedId,
  activeId,
  onSelect,
  onApply,
  onCancel,
  onTrain,
  applying,
  error,
  estimatedMinutes,
}: AdapterPickerProps) {
  if (adapters.length === 0) {
    return (
      <Typography variant="caption" color="text.secondary">
        Loading adapters...
      </Typography>
    );
  }

  const selectedAdapter = adapters.find((a) => a.adapter_id === selectedId);
  const isAlreadyActive = selectedId === activeId;

  return (
    <Box>
      <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: "block" }}>
        Select an existing adapter or train a new one for this project:
      </Typography>

      <Select
        value={selectedId}
        onChange={(e) => onSelect(e.target.value)}
        size="small"
        fullWidth
        displayEmpty
        sx={{
          mb: 1,
          fontSize: 13,
          bgcolor: "rgba(255,255,255,0.03)",
          "& .MuiSelect-select": { py: 1 },
        }}
      >
        <MenuItem value="" disabled>
          <em>Choose an adapter...</em>
        </MenuItem>
        {adapters.map((a) => (
          <MenuItem key={a.adapter_id} value={a.adapter_id}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, width: "100%" }}>
              <Chip
                label={a.source === "bundled" ? "Bundled" : "Trained"}
                size="small"
                sx={{
                  height: 18,
                  fontSize: 9,
                  fontWeight: 700,
                  bgcolor: a.source === "bundled"
                    ? "rgba(33,150,243,0.12)"
                    : "rgba(76,175,80,0.12)",
                  color: a.source === "bundled" ? "#2196f3" : "#4caf50",
                }}
              />
              <Typography variant="body2" sx={{ fontSize: 13 }}>
                {a.name}
              </Typography>
              {a.adapter_id === activeId && (
                <Chip
                  label="Active"
                  size="small"
                  sx={{
                    height: 16,
                    fontSize: 8,
                    fontWeight: 700,
                    ml: "auto",
                    bgcolor: "rgba(124,77,255,0.15)",
                    color: "#7c4dff",
                  }}
                />
              )}
            </Box>
          </MenuItem>
        ))}
      </Select>

      {selectedAdapter && (
        <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
          {selectedAdapter.description}
        </Typography>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 1, py: 0 }} variant="outlined">
          <Typography variant="caption">{error}</Typography>
        </Alert>
      )}

      <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
        <Button
          variant="contained"
          size="small"
          onClick={onApply}
          disabled={!selectedId || isAlreadyActive || applying}
          sx={{
            textTransform: "none",
            background: "linear-gradient(135deg, #7c4dff 0%, #651fff 100%)",
          }}
        >
          {applying ? "Applying..." : "Use This Adapter"}
        </Button>

        <Divider orientation="vertical" flexItem sx={{ mx: 0.5, opacity: 0.2 }} />

        <Button
          size="small"
          startIcon={<SchoolIcon sx={{ fontSize: "16px !important" }} />}
          onClick={onTrain}
          sx={{ textTransform: "none", fontSize: 12 }}
        >
          Train for this project
          {estimatedMinutes != null && ` (~${estimatedMinutes} min)`}
        </Button>

        <Box sx={{ flex: 1 }} />

        <Button
          size="small"
          onClick={onCancel}
          sx={{ textTransform: "none", fontSize: 12 }}
        >
          Cancel
        </Button>
      </Box>
    </Box>
  );
}
