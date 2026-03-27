import {
  Card,
  CardActionArea,
  CardContent,
  Box,
  Typography,
  Chip,
} from "@mui/material";
import AutoFixHighIcon from "@mui/icons-material/AutoFixHigh";
import FolderIcon from "@mui/icons-material/Folder";
import StarIcon from "@mui/icons-material/Star";
import type { RepoInfo } from "../types";

interface Props {
  repo: RepoInfo;
  onClick: () => void;
}

export default function RepoCard({ repo, onClick }: Props) {
  return (
    <Card
      sx={{
        transition: "all 0.2s ease",
        "&:hover": {
          borderColor: "primary.main",
          transform: "translateY(-2px)",
          boxShadow: "0 8px 24px rgba(124, 77, 255, 0.15)",
        },
      }}
    >
      <CardActionArea onClick={onClick} sx={{ p: 0.5 }}>
        <CardContent>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 1 }}>
            <FolderIcon sx={{ color: "primary.light" }} />
            <Typography variant="h6" sx={{ fontSize: "1rem" }}>
              {repo.full_name}
            </Typography>
          </Box>
          {repo.description && (
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{ mb: 1.5 }}
              noWrap
            >
              {repo.description}
            </Typography>
          )}
          <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
            <Chip
              icon={<StarIcon sx={{ fontSize: 14 }} />}
              label={repo.stars.toLocaleString()}
              size="small"
              variant="outlined"
            />
            <Chip
              label={`${repo.chunk_count} chunks`}
              size="small"
              color="primary"
              variant="outlined"
            />
            {repo.has_lora_adapter && (
              <Chip
                icon={<AutoFixHighIcon sx={{ fontSize: "14px !important" }} />}
                label="LoRA"
                size="small"
                sx={{
                  height: 22,
                  fontSize: 10,
                  fontWeight: 700,
                  bgcolor: "rgba(124,77,255,0.12)",
                  color: "#7c4dff",
                  "& .MuiChip-icon": { color: "#7c4dff" },
                }}
              />
            )}
            {repo.indexed_at && (
              <Typography variant="caption" color="text.secondary">
                Indexed {new Date(repo.indexed_at).toLocaleDateString()}
              </Typography>
            )}
          </Box>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}
