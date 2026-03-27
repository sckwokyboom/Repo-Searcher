import { useState, useCallback } from "react";
import type { SearchResponse } from "../types";
import { executeSearch } from "../services/searchService";

export function useCodeSearch(repoId: string) {
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const search = useCallback(
    async (query: string) => {
      if (!query.trim()) return;
      setLoading(true);
      setError(null);
      setResponse(null);
      try {
        const result = await executeSearch(repoId, query);
        setResponse(result);
      } catch (err: any) {
        setError(err?.response?.data?.detail || "Search failed");
        setResponse(null);
      } finally {
        setLoading(false);
      }
    },
    [repoId]
  );

  return { response, loading, error, search };
}
