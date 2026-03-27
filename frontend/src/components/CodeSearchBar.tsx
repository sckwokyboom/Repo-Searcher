import { useState } from "react";
import {
  Box,
  IconButton,
  InputAdornment,
  TextField,
} from "@mui/material";
import SearchIcon from "@mui/icons-material/Search";

interface Props {
  onSearch: (query: string) => void;
  loading?: boolean;
}

export default function CodeSearchBar({ onSearch, loading }: Props) {
  const [query, setQuery] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) onSearch(query.trim());
  };

  return (
    <Box component="form" onSubmit={handleSubmit}>
      <TextField
        fullWidth
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search code with natural language, e.g. 'how are new pet owners registered'"
        variant="outlined"
        disabled={loading}
        sx={{
          "& .MuiOutlinedInput-root": {
            borderRadius: 3,
            fontSize: "1.05rem",
            backgroundColor: "rgba(255,255,255,0.03)",
            "&:hover .MuiOutlinedInput-notchedOutline": {
              borderColor: "primary.main",
            },
          },
        }}
        slotProps={{
          input: {
            endAdornment: (
              <InputAdornment position="end">
                <IconButton
                  type="submit"
                  disabled={!query.trim() || loading}
                  color="primary"
                >
                  <SearchIcon />
                </IconButton>
              </InputAdornment>
            ),
          },
        }}
      />
    </Box>
  );
}
