import api from "./api";
import type { GitHubRepo, RepoInfo } from "../types";

export async function searchRepos(query: string): Promise<GitHubRepo[]> {
  const { data } = await api.get("/repos/search", { params: { q: query } });
  return data;
}

export async function getIndexedRepos(): Promise<RepoInfo[]> {
  const { data } = await api.get("/repos/indexed");
  return data;
}

export async function indexRepo(repo: RepoInfo): Promise<void> {
  await api.post("/repos/index", repo);
}

export async function getRepoStatus(
  repoId: string
): Promise<{ repo_id: string; status: string; repo_info: RepoInfo | null }> {
  const { data } = await api.get(`/repos/${repoId}/status`);
  return data;
}

export async function deleteRepoIndex(repoId: string): Promise<void> {
  await api.delete(`/repos/${repoId}`);
}
