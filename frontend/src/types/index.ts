export type IndexingStep =
  | "cloning"
  | "parsing"
  | "building_bm25"
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
  has_lora_adapter: boolean;
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
  callers: string[];
  callees: string[];
}

export interface RewriteDetails {
  intent: string | null;
  search_scope: string | null;
  keywords: string[];
  project_terms: string[];
  method_hints: string[];
  api_hints: string[];
  search_queries: string[];
}

export interface SearchResponse {
  query: string;
  rewritten_query: string | null;
  expanded_keywords: string[];
  rewrite_details: RewriteDetails | null;
  results: SearchResult[];
  search_time_ms: number;
  lora_active: boolean;
}

// --- LoRA Training types ---

export type LoRATrainingStep =
  | "preparing_data"
  | "training"
  | "saving"
  | "done"
  | "failed"
  | "cancelled";

export interface LoRATrainingProgress {
  repo_id: string;
  step: LoRATrainingStep;
  progress: number;
  message: string;
  epoch: number;
  total_epochs: number;
  train_loss: number | null;
  eval_loss: number | null;
  estimated_time_remaining_sec: number | null;
}

export interface LoRAStatus {
  repo_id: string;
  has_adapter: boolean;
  active_adapter_id: string | null;
  is_training: boolean;
  estimated_minutes: number | null;
}

export interface LoRAAdapterInfo {
  adapter_id: string;
  name: string;
  description: string;
  source: "bundled" | "trained";
  trained_for_repo: string | null;
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

export interface MCTSRewardComponents {
  bm25: number;
  semantic: number;
  llm: number;
}

export interface MCTSHit {
  chunk_id: string;
  chunk_type: string;
  name: string;
  file_path: string;
  signature: string;
  bm25_score: number;
  semantic_score: number;
  is_new: boolean;
}

export interface MCTSNode {
  id: number;
  parent_id: number | null;
  query: string;
  visits: number;
  avg_reward: number;
  is_best: boolean;
  children_ids: number[];
  top_hits: MCTSHit[];
  reward_components: MCTSRewardComponents | null;
}

export interface MCTSTrace {
  trace_id: string;
  repo_id: string;
  original_query: string;
  best_query: string;
  iterations: number;
  nodes: MCTSNode[];
}
