import api from "./api";
import type { SearchResponse } from "../types";

export async function executeSearch(
  repoId: string,
  query: string,
  topK: number = 5
): Promise<SearchResponse> {
  const { data } = await api.post(`/repos/${repoId}/search`, {
    query,
    top_k: topK,
  });
  return data;
}
