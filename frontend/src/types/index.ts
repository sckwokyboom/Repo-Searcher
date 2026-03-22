export type IndexingStep =
  | "cloning"
  | "parsing"
  | "building_bm25"
  | "building_vectors"
  | "building_callgraph"
  | "saving"
  | "done"
  | "failed";

export interface RepoInfo {
  repo_id: string;
  owner: string;
  name: string;
  full_name: string;
  description: string | null;
  stars: number;
  url: string;
  language: string | null;
  indexed_at: string | null;
  chunk_count: number;
}

export interface IndexingProgress {
  repo_id: string;
  step: IndexingStep;
  progress: number;
  message: string;
  files_processed: number;
  files_total: number;
}

export interface CodeChunk {
  chunk_id: string;
  chunk_type: "method" | "class";
  file_path: string;
  class_name: string | null;
  method_name: string | null;
  signature: string;
  javadoc: string | null;
  body: string;
  start_line: number;
  end_line: number;
}

export interface SearchResult {
  chunk: CodeChunk;
  score: number;
  bm25_rank: number | null;
  vector_rank: number | null;
  rrf_score: number | null;
  callers: string[];
  callees: string[];
}

export interface SearchResponse {
  query: string;
  expanded_keywords: string[];
  results: SearchResult[];
  search_time_ms: number;
}

export interface CallGraphData {
  nodes: { id: string; label: string; file_path: string; is_result: boolean }[];
  edges: { source: string; target: string }[];
}

export interface GitHubRepo {
  full_name: string;
  description: string | null;
  stars: number;
  url: string;
  owner: string;
  owner_avatar: string;
  language: string | null;
}
