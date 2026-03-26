import {
  Box,
  LinearProgress,
  Paper,
  Step,
  StepLabel,
  Stepper,
  Typography,
} from "@mui/material";
import type { IndexingProgress as IProgress, IndexingStep } from "../types";

const STEPS: { key: IndexingStep; label: string }[] = [
  { key: "cloning", label: "Clone" },
  { key: "parsing", label: "Parse" },
  { key: "building_bm25", label: "BM25" },
  { key: "building_callgraph", label: "Call Graph" },
  { key: "saving", label: "Save" },
];

function getActiveStep(step: IndexingStep): number {
  const idx = STEPS.findIndex((s) => s.key === step);
  if (step === "done") return STEPS.length;
  if (step === "failed") return -1;
  return idx >= 0 ? idx : 0;
}

interface Props {
  progress: IProgress | null;
}

export default function IndexingProgress({ progress }: Props) {
  if (!progress) {
    return (
      <Paper sx={{ p: 4, textAlign: "center" }}>
        <Typography color="text.secondary">
          Connecting to indexing service...
        </Typography>
        <LinearProgress sx={{ mt: 2 }} />
      </Paper>
    );
  }

  const activeStep = getActiveStep(progress.step);
  const isFailed = progress.step === "failed";
  const isDone = progress.step === "done";

  return (
    <Paper sx={{ p: 4 }}>
      <Typography variant="h6" sx={{ mb: 3 }}>
        {isDone
          ? "Indexing Complete!"
          : isFailed
          ? "Indexing Failed"
          : "Indexing Repository..."}
      </Typography>

      <Stepper activeStep={activeStep} alternativeLabel sx={{ mb: 3 }}>
        {STEPS.map((step) => (
          <Step key={step.key}>
            <StepLabel
              error={isFailed && step.key === progress.step}
            >
              {step.label}
            </StepLabel>
          </Step>
        ))}
      </Stepper>

      {!isDone && !isFailed && (
        <>
          <LinearProgress
            variant="determinate"
            value={progress.progress * 100}
            sx={{
              height: 8,
              borderRadius: 4,
              mb: 2,
              "& .MuiLinearProgress-bar": {
                background:
                  "linear-gradient(90deg, #7c4dff 0%, #00e5ff 100%)",
              },
            }}
          />
          <Typography variant="body2" color="text.secondary">
            {progress.message}
          </Typography>
          {progress.files_total > 0 && (
            <Typography variant="caption" color="text.secondary">
              {progress.files_processed} / {progress.files_total} files
            </Typography>
          )}
        </>
      )}

      {isFailed && (
        <Typography color="error" sx={{ mt: 1 }}>
          {progress.message}
        </Typography>
      )}
    </Paper>
  );
}
