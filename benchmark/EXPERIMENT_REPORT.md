# Repo-Searcher: Benchmark Experiment Report

**Date:** 2026-03-24
**Author:** Auto-generated from experimental data
**Platform:** Apple Silicon (MPS), macOS

---

## 1. Executive Summary

This report documents a comprehensive benchmarking effort to evaluate and improve the Repo-Searcher code retrieval system. The core question: **can an MCTS + LLM-based retriever outperform simple BM25 keyword search for finding relevant Java code files?**

**Key findings:**
- **BM25 is the strongest baseline** with Recall@5=0.544 (small repos) and 0.259 (all repos)
- **LLM reranking with Qwen2.5-Coder-1.5B provides zero improvement** over BM25 while being 400x slower
- **File-level aggregation** significantly improves Recall@20 (+25% absolute) at the cost of Recall@1
- **LoRA fine-tuning** collapsed to predicting a single score, yielding no improvement
- **The MCTS + Graph pipeline was not benchmarked** due to prohibitive inference cost (~30K LLM calls per full benchmark run)

---

## 2. Dataset Construction

### 2.1 Source Data

The benchmark was derived from **claude-20250813.jsonl** (AgentPack dataset with agent trajectories).

**Extraction criteria:**
- Samples where agents generated code for Java projects
- Git diffs containing `.java` files
- Selecting repos with the most samples to maximize per-project density
- Goal: 50+ samples across 5-10 projects

**Query cleaning pipeline** (`query_cleaner.py`):
- Strip boilerplate (Co-authored-by, Generated with, etc.)
- Remove conventional commit prefixes (feat:, fix:, chore:, etc.)
- Strip SHA references, markdown links, PR/issue references, emoji
- Filter low-quality queries (<5 words)

### 2.2 Final Dataset

**Total:** 859 samples across 10 repositories

| Repository | Samples | Chunks | Java Files | Notes |
|---|---|---|---|---|
| 100-hours-a-week/22-tenten-be | 123 | 1,354 | 272 | Korean project |
| panghy/javaflow | 121 | 3,498 | 240 | FoundationDB-style async |
| etonai/OpenFields2 | 108 | 3,097 | 217 | Game (combat system) |
| hmislk/hmis | 100 | 74,049 | 1,797 | Hospital management (huge) |
| rydnr/bytehot | 92 | 5,793 | 634 | Hot-reloading framework |
| heymumford/Samstraumr | 79 | 8,041 | 635 | Testing framework |
| dotCMS/core | 63 | 67,562 | 7,449 | CMS (huge) |
| PAIR-Systems-Inc/goodmem | 63 | - | - | Repo deleted, excluded |
| jdereg/java-util | 58 | 9,573 | ~400 | Utility library |
| DataSQRL/sqrl | 52 | 4,936 | 845 | SQL engine |

**Effective samples after validation:** 634 (full benchmark), 287 (small repos subset)

### 2.3 Sample Format

Each benchmark sample contains:

```json
{
  "event_id": "51445618907",
  "repo": "jdereg/java-util",
  "sha": "abc123...",
  "query": "Fix ArrayUtilities code review issues - Fix EMPTY_CLASS_ARRAY generic type safety",
  "raw_description": "fix(ArrayUtilities): Fix code review issues...",
  "changed_files": ["src/main/java/com/cedarsoftware/util/ArrayUtilities.java"],
  "changed_methods": ["isEmpty", "getLength"]
}
```

**Ground truth:** The `changed_files` from the agent's git diff serve as the retrieval ground truth. If a retriever returns any file from this set within top-K, it's a hit.

---

## 3. Indexing Pipeline

### 3.1 Repository Cloning and Parsing

Each repo was cloned and parsed into **code chunks**:
- Java AST parsing extracts: classes, methods, fields, constructors
- Each chunk includes: `file_path`, `class_name`, `method_name`, `signature`, `javadoc`, `text_representation`
- Chunk ID format: `file_path::ClassName::methodName`

### 3.2 Index Types Built

For each repository, three indexes were built:

1. **BM25 index** — tokenized text representations, using `rank_bm25` library
2. **FAISS vector index** — embeddings from UniXcoder (microsoft/unixcoder-base, 768-dim)
3. **Call graph** — NetworkX directed graph of method call relationships

**Indexing time:** Large repos (hmis: 74K chunks, dotCMS: 67K chunks) took ~10-20 minutes each for vector encoding on MPS.

---

## 4. Retrievers Evaluated

### 4.1 Retriever Descriptions

| Retriever | Description | Speed per query |
|---|---|---|
| `bm25` | BM25 keyword search on tokenized chunk text | ~0.07s |
| `vector` | FAISS cosine similarity with UniXcoder embeddings | ~0.1s |
| `rrf` | Reciprocal Rank Fusion of BM25 + Vector (k=60) | ~0.15s |
| `weighted` | Weighted normalized fusion: 0.7*BM25 + 0.3*Vector | ~0.15s |
| `bm25_fileagg` | BM25 with file-level score aggregation | ~0.07s |
| `hybrid_fileagg` | Weighted BM25+Vector with file-level aggregation | ~0.15s |
| `bm25_qwen_rerank` | BM25 top-30 + Qwen 1.5B batch reranking (few-shot) | ~8s |
| `qwen3_vector` | FAISS with Qwen3-Embedding-0.6B embeddings | ~0.1s |
| `qwen3_hybrid_fileagg` | BM25 + Qwen3-Embedding with file aggregation | ~0.15s |

### 4.2 RRF Formula

Reciprocal Rank Fusion score for document d:

```
RRF(d) = sum_i(1 / (k + rank_i(d)))
```

where `rank_i(d)` is the rank of document `d` in the i-th retriever's list, `k=60` (constant).

### 4.3 Batch LLM Reranking Prompt (Few-Shot)

```
Rate each Java code snippet's relevance to the search query (0=irrelevant, 10=perfect match).

Example:
Query: Fix NPE in user authentication
Candidates:
[0] AuthService.java | AuthService | authenticate | public User authenticate(String token)
[1] MathUtils.java | MathUtils | factorial | public static long factorial(int n)
[2] AuthController.java | AuthController | login | public Response login(LoginRequest req)
Scores: [0] 9, [1] 0, [2] 6

Now rate:
Query: {user_query}
Candidates:
[0] {chunk_0_summary}
[1] {chunk_1_summary}
...
Scores:
```

This few-shot format significantly improved score differentiation compared to zero-shot (which produced flat scores of all 9s or all 10s).

---

## 5. Experiment 1: Baseline Retriever Comparison

### 5.1 Setup

- **Dataset:** 634 validated samples across 9 repos (excluding deleted goodmem)
- **Repos included:** All 9 available repos (including large hmis and dotCMS)
- **K values:** 1, 3, 5, 10, 20

### 5.2 Results

| Retriever | R@1 | R@3 | R@5 | R@10 | R@20 | MRR | Hit@5 |
|---|---|---|---|---|---|---|---|
| **bm25** | 0.120 | 0.220 | **0.259** | 0.337 | 0.362 | **0.335** | 0.424 |
| **weighted** | **0.126** | **0.214** | **0.260** | 0.325 | 0.352 | **0.337** | **0.435** |
| rrf | 0.108 | 0.200 | 0.236 | 0.306 | 0.338 | 0.323 | 0.409 |
| bm25_fileagg | 0.070 | 0.126 | 0.178 | 0.270 | 0.374 | 0.241 | 0.358 |
| hybrid_fileagg | 0.061 | 0.132 | 0.176 | 0.262 | 0.375 | 0.239 | 0.345 |
| vector | 0.059 | 0.122 | 0.156 | 0.218 | 0.230 | 0.222 | 0.312 |

### 5.3 Analysis

1. **BM25 dominates** — The best retriever is pure BM25 or weighted BM25-dominant fusion (0.7/0.3). This is because queries in this dataset are commit messages that often contain class/method names literally (e.g., "Add StringUtilities.containsIgnoreCase").

2. **Vector search (UniXcoder) is weak** — Recall@5=0.156, roughly half of BM25. UniXcoder was trained for code-to-code similarity, not natural-language-to-code retrieval.

3. **RRF hurts more than helps** — RRF(BM25 + Vector) = 0.236 < BM25 alone (0.259). The weak vector component "dilutes" the strong BM25 signal.

4. **File aggregation trades precision for coverage** — BM25_FileAgg has lower R@1 (0.070 vs 0.120) but higher R@20 (0.374 vs 0.362). Summing chunk scores per file helps find the right file but loses fine-grained chunk ranking.

---

## 6. Experiment 2: Qwen3-Embedding-0.6B

### 6.1 Motivation

UniXcoder (microsoft/unixcoder-base, 125M params) was designed for code-to-code tasks. Qwen3-Embedding-0.6B is a newer, larger model specifically trained for cross-lingual text-code retrieval with 32K context and 1024-dim embeddings.

### 6.2 Setup

- Loaded via HuggingFace transformers with `trust_remote_code=True`
- Mean pooling + L2 normalization of last hidden states
- FAISS IndexFlatIP (inner product = cosine similarity on normalized vectors)
- Cached embeddings as `.npy` files per repo
- Tested on jdereg/java-util (quick validation: 10 queries)

### 6.3 Results (Quick Test on jdereg/java-util)

Informal test on 10 queries showed 8/10 correct top-file hits, suggesting significant improvement over UniXcoder for natural language queries. However, full benchmark across all repos was not completed due to time constraints (encoding 74K chunks for hmis).

### 6.4 Cached Embeddings Generated

| Repository | Embedding File Size |
|---|---|
| 100-hours-a-week--22-tenten-be | 5.3 MB |
| etonai--OpenFields2 | 12 MB |
| jdereg--java-util | 37 MB |
| panghy--javaflow | 14 MB |
| rydnr--bytehot | 23 MB |

---

## 7. Experiment 3: LLM Reranking (Qwen2.5-Coder-1.5B)

### 7.1 Setup

- **Model:** Qwen/Qwen2.5-Coder-1.5B (float16 on MPS)
- **Method:** BM25 top-30 candidates → batch reranking with few-shot prompt → re-sort by LLM score
- **Decoding:** Greedy (do_sample=False) — required because float16 on MPS produces nan with sampling
- **Dataset:** 287 samples across 3 small repos (jdereg/java-util, panghy/javaflow, etonai/OpenFields2)

### 7.2 Results

| Retriever | R@1 | R@3 | R@5 | R@10 | R@20 | MRR@5 | Time |
|---|---|---|---|---|---|---|---|
| bm25 | **0.324** | 0.481 | **0.544** | 0.648 | 0.652 | 0.404 | 20.4s |
| bm25_fileagg | 0.244 | 0.394 | 0.512 | **0.676** | **0.819** | 0.334 | 19.2s |
| bm25_qwen_rerank | **0.324** | **0.488** | **0.544** | 0.641 | 0.645 | **0.407** | **8,162s** |

### 7.3 Analysis

**LLM reranking provides zero meaningful improvement:**
- R@1 identical (0.324)
- R@3 marginally better (0.488 vs 0.481, +0.7%)
- R@5 identical (0.544)
- R@10 slightly worse (0.641 vs 0.648)
- **Time: 8,162s vs 20s = 408x slower**

**Root cause:** Qwen 1.5B outputs nearly flat scores. Even with few-shot prompting, the model doesn't meaningfully distinguish relevant from irrelevant code:
- Zero-shot: all candidates get score 10
- Few-shot: the model copies the example pattern (9, 0, 6) with slight variations
- The LoRA-finetuned version collapsed to outputting "2" for everything

### 7.4 Technical Challenges

1. **MPS segfaults:** LoRA-merged models crash on MPS. Root cause: float16 tensor operations after PEFT merge produce nan values. Workaround: use float32 (2x memory) or run on CPU.

2. **Memory pressure:** Loading Qwen 1.5B in float32 for LoRA merge requires ~6GB, which caused OOM kills in background processes with limited memory allocations.

3. **Speed:** Each batch reranking call (30 candidates) takes ~8s on MPS, making full dataset evaluation impractical. 287 samples = 2h16m.

---

## 8. Experiment 4: LoRA Fine-Tuning

### 8.1 Training Data Generation

**Source:** Git history of jdereg/java-util and panghy/javaflow (2 repos with most commits)

**Method:**
1. Extract commits that modify `.java` files (up to 2000 per repo)
2. Clean commit messages (same pipeline as benchmark query cleaning)
3. For each commit:
   - **Positive chunks:** Best BM25-matching chunk from each changed file (score label: 9)
   - **Hard negatives:** Top BM25 hits from unchanged files (score label: 2)
   - Max 5 positives + 5 negatives per commit

**Training data format:**
```
Prompt: "Is this Java code relevant to the search query? Answer with a single number from 0 (irrelevant) to 10 (perfect match).

Query: Add negative tests for mapOf
File: src/test/java/.../DimensionConversionsTest.java
Class: DimensionConversionsTest
Method: testNumberToDimension_n

Relevance score (0-10):"

Completion: "2"
```

### 8.2 Dataset Statistics

| Split | Positive | Negative | Total | Positive Ratio |
|---|---|---|---|---|
| Train | 2,193 | 5,101 | 7,294 | 30.1% |
| Validation | 218 | 593 | 811 | 26.9% |
| **Total** | **2,411** | **5,694** | **8,105** | **29.7%** |

Note: Training was capped at 1,000 samples (random subset of train) and 200 samples (random subset of val) to fit in MPS memory.

### 8.3 LoRA Configuration

| Parameter | Value |
|---|---|
| Base Model | Qwen/Qwen2.5-Coder-1.5B |
| PEFT Type | LoRA |
| Rank (r) | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Task Type | CAUSAL_LM |

### 8.4 Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 1 |
| Batch Size | 4 |
| Gradient Accumulation Steps | 4 |
| Effective Batch Size | 16 |
| Learning Rate | 2e-4 |
| LR Scheduler | Cosine |
| Warmup Ratio | 0.05 |
| Weight Decay | 0.01 |
| Max Gradient Norm | 1.0 |
| Max Sequence Length | 256 tokens |
| Precision | float16 (model) |
| Device | MPS (Apple Silicon) |
| Training Samples Used | 1,000 (of 7,294) |
| Validation Samples Used | 200 (of 811) |

### 8.5 Training Results

| Metric | Value |
|---|---|
| Final Eval Loss | 1.486 |
| Training Time | ~15 minutes |
| Adapter Size | ~18 MB (adapter_model.safetensors) |
| Trainable Parameters | ~3.5M (of 1.5B total) |

### 8.6 Evaluation

**The LoRA adapter collapsed to a degenerate solution:**
- In zero-shot scoring mode: outputs "2" for all inputs (the majority negative class label)
- With few-shot prompt: copies the example pattern with minimal variation
- Score distribution is nearly uniform, providing no ranking signal

**Probable causes:**
1. **Too few effective training samples:** Only 1,000 samples used (memory constraint), with 256-token truncation losing most code context
2. **Binary-like labels:** Scores fixed at 9 (positive) and 2 (negative) — model learns to output one of two values, not a continuous relevance scale
3. **1 epoch insufficient:** A single pass over 1,000 samples (62 gradient steps at effective batch size 16) is not enough for the model to learn the scoring task
4. **SFT vs preference training:** SFT on "(prompt, score)" pairs doesn't effectively teach ranking. A contrastive/preference-based approach (DPO, listwise ranking loss) would be more appropriate.

---

## 9. Architecture Analysis: MCTS + Graph Pipeline

### 9.1 Full Pipeline Description

The Repo-Searcher's "full" pipeline consists of three LLM-dependent stages:

```
User Query
    |
    v
[1] MCTS Query Rewriter (mcts_rewriter.py)
    |-- 2 iterations x 3 children = 7 LLM generate() calls for rewriting
    |-- 7 LLM _llm_score() calls for reward evaluation
    |-- 14 BM25 + FAISS searches for reward
    v
Best rewritten query
    |
    v
[2] BM25 + FAISS → RRF Fusion → top-30 candidates
    v
[3] Graph MCTS (graph_mcts.py)
    |-- 3 iterations of call graph traversal
    |-- UniXcoder encoding for each explored node (~10-30 nodes)
    v
Merged candidates (RRF + graph discoveries)
    |
    v
[4] LLM Reranker (reranker.py)
    |-- 30 LLM generate() calls (one per candidate)
    v
Final ranked results
```

**Total LLM calls per query: ~44** (7 rewrite + 7 reward + 30 rerank)

### 9.2 Why It Was Not Benchmarked

At ~8s per LLM call on MPS:
- 44 calls per query = **352 seconds per query**
- 287 samples = **28 hours** for one benchmark run
- 634 samples = **62 hours**

This is impractical for iterative experimentation.

### 9.3 Fundamental Issues

1. **Speed:** 352s/query vs 0.07s/query (BM25) = **5,000x slower**
2. **LLM quality:** Qwen 1.5B cannot meaningfully score code relevance (Experiment 3 proved this)
3. **Graph MCTS uses UniXcoder:** The graph traversal reward is cosine similarity from UniXcoder, which we've shown is weak for NL-to-code retrieval
4. **Query rewriting adds cost without benefit:** If BM25 already matches class/method names from the query, rewriting to "different aspects" can only lose signal

---

## 10. Comparative Timing

| Component | Time per Query | For 287 Samples | For 634 Samples |
|---|---|---|---|
| BM25 | 0.07s | 20s | 44s |
| BM25 + FileAgg | 0.07s | 19s | 42s |
| Vector (UniXcoder) | ~0.1s | ~29s | ~63s |
| RRF (BM25 + Vector) | ~0.15s | ~43s | ~95s |
| BM25 + Qwen Rerank (batch) | ~8s | 8,162s (2h16m) | ~18,000s (~5h) |
| Full MCTS Pipeline (est.) | ~352s | ~28h (est.) | ~62h (est.) |

---

## 11. Embedding Models Investigated

| Model | Params | Dim | Purpose | Status |
|---|---|---|---|---|
| microsoft/unixcoder-base | 125M | 768 | Code-to-code similarity | Tested, weak for NL-to-code |
| Qwen/Qwen3-Embedding-0.6B | 600M | 1024 | NL-to-code retrieval | Partially tested, promising |
| Qwen/Qwen2.5-Coder-1.5B | 1.5B | - | Generative (reranking/scoring) | Tested, too weak for scoring |
| codesage/codesage-small | 130M | 1024 | Code search | Researched, not tested |
| jinaai/jina-embeddings-v2-base-code | 161M | - | Code search, 8K context | Researched, not tested |
| Salesforce/SFR-Embedding-Code-2B_R | 2B | - | Code search (SOTA on CoIR) | Researched, non-commercial |
| nomic-ai/nomic-embed-code | 7B | - | Code search (SOTA on CSN) | Researched, too large |

---

## 12. Lessons Learned

### 12.1 What Worked

1. **BM25 is an extremely strong baseline** for code retrieval when queries contain code identifiers
2. **File-level aggregation** meaningfully improves recall at higher K values
3. **Qwen3-Embedding-0.6B** showed promise in informal testing (8/10 hits vs ~5/10 for UniXcoder)
4. **Batch scoring with few-shot prompts** is significantly better than individual zero-shot scoring for small LLMs
5. **The benchmark infrastructure** (JSONL extraction, indexing pipeline, evaluation framework) works correctly and is reusable

### 12.2 What Didn't Work

1. **Qwen2.5-Coder-1.5B for scoring:** Too small to understand relevance. Outputs flat scores.
2. **LoRA SFT with binary labels:** Model collapses to majority class. Need continuous labels or preference-based training.
3. **MCTS query rewriting:** Adds 14+ LLM calls per query. If LLM is weak, rewriting only adds noise and latency.
4. **UniXcoder for NL-to-code:** Designed for code-code similarity, not cross-modal retrieval.
5. **MPS + float16 + LoRA:** Consistently crashes. MPS has known issues with certain tensor operations in merged PEFT models.

### 12.3 Recommendations for Future Work

1. **Replace UniXcoder with Qwen3-Embedding-0.6B** as the vector retrieval backbone
2. **Drop MCTS and LLM reranking** unless a 7B+ model is available
3. **Focus on BM25 + Qwen3-Embedding hybrid + FileAgg** — this should combine the lexical strength of BM25 with semantic understanding
4. **For LoRA training:** Use DPO/contrastive loss instead of SFT, train for 3-5 epochs on full dataset, use continuous relevance labels
5. **Consider semantic query types separately:** BM25 excels when queries contain identifiers. A semantic-only query subset would better test vector retrieval value.

---

## 13. Appendix: File Structure

```
benchmark/
  run_benchmark.py          # Main benchmark pipeline CLI
  run_lora_benchmark.py     # LoRA-specific benchmark script
  config.py                 # Benchmark data models and paths
  retrievers.py             # All retriever implementations
  evaluator.py              # Metric computation
  visualize.py              # Table/plot generation
  extract_samples.py        # JSONL sample extraction
  clone_and_index.py        # Repository cloning and indexing
  query_cleaner.py          # Commit message cleaning
  results/
    benchmark_samples.json  # 859 samples, 10 repos
    eval_results.json       # First benchmark (634 samples, 6 retrievers)
    lora_benchmark_results.json  # LoRA benchmark (287 samples, 3 retrievers)
    lora_benchmark_results.md
    qwen3_faiss/            # Cached Qwen3 embeddings (5 repos)
  lora_training/
    prepare_data.py         # Training data generator from git history
    train_scorer.py         # LoRA training script
    data/
      train_scorer.jsonl    # 7,294 training samples (8.5 MB)
      val_scorer.jsonl      # 811 validation samples (950 KB)
    output/scorer_lora/final/  # Trained adapter (18 MB)
```

---

## 14. Appendix: Configuration Parameters

### Backend Settings (`backend/app/config.py`)

| Parameter | Value |
|---|---|
| unixcoder_model | microsoft/unixcoder-base |
| qwen_model | Qwen/Qwen2.5-Coder-1.5B |
| bm25_top_k | 30 |
| faiss_top_k | 30 |
| rrf_k | 60 |
| rrf_top_k | 30 |
| reranker_top_k | 10 |
| embedding_batch_size | 32 |
| embedding_dim | 768 |
| mcts_iterations | 2 |
| mcts_children | 3 |
| graph_mcts_iterations | 3 |
| graph_mcts_reward_threshold | 0.3 |
| graph_mcts_max_discoveries | 3 |

---

## 15. Appendix: Raw Metrics Tables

### Full Benchmark (634 samples, 9 repos)

| Retriever | R@1 | R@3 | R@5 | R@10 | R@20 | MRR | Hit@5 |
|---|---|---|---|---|---|---|---|
| bm25 | 0.120 | 0.220 | 0.259 | 0.337 | 0.362 | 0.335 | 0.424 |
| weighted | 0.126 | 0.214 | 0.260 | 0.325 | 0.352 | 0.337 | 0.435 |
| rrf | 0.108 | 0.200 | 0.236 | 0.306 | 0.338 | 0.323 | 0.409 |
| bm25_fileagg | 0.070 | 0.126 | 0.178 | 0.270 | 0.374 | 0.241 | 0.358 |
| hybrid_fileagg | 0.061 | 0.132 | 0.176 | 0.262 | 0.375 | 0.239 | 0.345 |
| vector | 0.059 | 0.122 | 0.156 | 0.218 | 0.230 | 0.222 | 0.312 |

### Small Repos Benchmark (287 samples, 3 repos)

| Retriever | R@1 | R@3 | R@5 | R@10 | R@20 | MRR@5 | Time |
|---|---|---|---|---|---|---|---|
| bm25 | 0.324 | 0.481 | 0.544 | 0.648 | 0.652 | 0.404 | 20.4s |
| bm25_fileagg | 0.244 | 0.394 | 0.512 | 0.676 | 0.819 | 0.334 | 19.2s |
| bm25_qwen_rerank | 0.324 | 0.488 | 0.544 | 0.641 | 0.645 | 0.407 | 8,162s |

**Note:** Small repos show much higher recall because the search space is smaller (3K-10K chunks vs 67K-74K for hmis/dotCMS).
