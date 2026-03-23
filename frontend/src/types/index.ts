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
  source: "search" | "graph_mcts";
  discovered_via: string | null;
  relation: string | null;
}

// --- MCTS Trace types ---

export interface MCTSHit {
  chunk_id: string;
  name: string;
  file_path: string;
  chunk_type: string;
  signature: string;
  bm25_score: number;
  semantic_score: number;
  is_new: boolean;
}

export interface MCTSRewardComponents {
  bm25: number;
  semantic: number;
  llm: number;
}

export interface MCTSNode {
  id: number;
  query: string;
  parent_id: number | null;
  children_ids: number[];
  visits: number;
  avg_reward: number;
  is_best: boolean;
  top_hits: MCTSHit[];
  reward_components: MCTSRewardComponents;
}

export interface MCTSTrace {
  nodes: MCTSNode[];
  iterations: number;
  best_path: number[];
  best_query: string;
  original_query: string;
}

// --- Graph MCTS types ---

export interface GraphMCTSNode {
  chunk_id: string;
  name: string;
  file_path: string;
  visits: number;
  avg_reward: number;
  discovered_via: string;
  relation: string;
}

export interface GraphMCTSTrace {
  explored_nodes: GraphMCTSNode[];
  total_nodes_visited: number;
  discoveries_count: number;
}

export interface SearchResponse {
  query: string;
  expanded_keywords: string[];
  results: SearchResult[];
  search_time_ms: number;
  mcts_trace: MCTSTrace | null;
  graph_mcts_trace: GraphMCTSTrace | null;
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
