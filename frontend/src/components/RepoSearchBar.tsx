import {
  Autocomplete,
  Avatar,
  Box,
  CircularProgress,
  TextField,
  Typography,
} from "@mui/material";
import StarIcon from "@mui/icons-material/Star";
import type { GitHubRepo } from "../types";
import { useRepoSearch } from "../hooks/useRepoSearch";

interface Props {
  onSelect: (repo: GitHubRepo) => void;
}

export default function RepoSearchBar({ onSelect }: Props) {
  const { query, setQuery, results, loading } = useRepoSearch();

  return (
    <Autocomplete
      freeSolo
      options={results}
      getOptionLabel={(option) =>
        typeof option === "string" ? option : option.full_name
      }
      filterOptions={(x) => x}
      inputValue={query}
      onInputChange={(_, value) => setQuery(value)}
      onChange={(_, value) => {
        if (value && typeof value !== "string") {
          onSelect(value);
        }
      }}
      loading={loading}
      renderInput={(params) => (
        <TextField
          {...params}
          placeholder="Search Java repositories on GitHub..."
          variant="outlined"
          sx={{
            "& .MuiOutlinedInput-root": {
              borderRadius: 3,
              fontSize: "1.1rem",
              backgroundColor: "rgba(255,255,255,0.03)",
              "&:hover .MuiOutlinedInput-notchedOutline": {
                borderColor: "primary.main",
              },
            },
          }}
          slotProps={{
            input: {
              ...params.InputProps,
              endAdornment: (
                <>
                  {loading && <CircularProgress size={20} />}
                  {params.InputProps.endAdornment}
                </>
              ),
            },
          }}
        />
      )}
      renderOption={(props, option) => {
        const { key, ...rest } = props;
        return (
          <Box
            component="li"
            key={key}
            {...rest}
            sx={{ display: "flex", gap: 2, py: 1.5 }}
          >
            <Avatar
              src={option.owner_avatar}
              sx={{ width: 32, height: 32 }}
            />
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography variant="body1" fontWeight={600}>
                {option.full_name}
              </Typography>
              {option.description && (
                <Typography
                  variant="body2"
                  color="text.secondary"
                  noWrap
                >
                  {option.description}
                </Typography>
              )}
            </Box>
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                gap: 0.5,
                color: "text.secondary",
              }}
            >
              <StarIcon sx={{ fontSize: 16 }} />
              <Typography variant="body2">
                {option.stars.toLocaleString()}
              </Typography>
            </Box>
          </Box>
        );
      }}
    />
  );
}
