import { useState, useEffect, useRef } from "react";
import type { GitHubRepo } from "../types";
import { searchRepos } from "../services/repoService";

export function useRepoSearch(debounceMs: number = 300) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<GitHubRepo[]>([]);
  const [loading, setLoading] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    if (!query || query.length < 2) {
      setResults([]);
      return;
    }

    if (timerRef.current) clearTimeout(timerRef.current);

    timerRef.current = setTimeout(async () => {
      setLoading(true);
      try {
        const repos = await searchRepos(query);
        setResults(repos);
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, debounceMs);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [query, debounceMs]);

  return { query, setQuery, results, loading };
}
