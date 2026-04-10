from pathlib import Path

from pydantic import BaseModel

BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
REPOS_DIR = RESULTS_DIR / "repos"

SAMPLES_PATH = RESULTS_DIR / "benchmark_samples.json"
RAW_RESULTS_PATH = RESULTS_DIR / "raw_results.json"
EVAL_RESULTS_PATH = RESULTS_DIR / "eval_results.json"

DEFAULT_TOP_K_VALUES = [1, 3, 5, 10, 20]
DEFAULT_MAX_REPOS = 10
DEFAULT_MIN_SAMPLES = 50


class BenchmarkSample(BaseModel):
    event_id: str
    repo: str
    sha: str
    query: str
    raw_description: str
    changed_files: list[str]
    changed_methods: list[str]
    low_quality: bool = False


class BenchmarkDataset(BaseModel):
    repos: dict[str, list[BenchmarkSample]]
    total_samples: int = 0
    total_repos: int = 0


class RetrievalResult(BaseModel):
    sample_id: str
    retriever: str
    retrieved_files: list[str]
    retrieved_methods: list[str]
    scores: list[float]
    top_k: int


class SampleMetrics(BaseModel):
    sample_id: str
    retriever: str
    repo: str
    recall_at_k: dict[int, float] = {}
    precision_at_k: dict[int, float] = {}
    mrr: float = 0.0
    hit_at_k: dict[int, float] = {}
    method_recall_at_k: dict[int, float] = {}
    method_hit_at_k: dict[int, float] = {}


class AggregatedMetrics(BaseModel):
    retriever: str
    recall_at_k: dict[int, float] = {}
    precision_at_k: dict[int, float] = {}
    mrr: float = 0.0
    hit_at_k: dict[int, float] = {}
    method_recall_at_k: dict[int, float] = {}
    method_hit_at_k: dict[int, float] = {}
    num_samples: int = 0


class EvalResults(BaseModel):
    per_sample: list[SampleMetrics] = []
    per_retriever: list[AggregatedMetrics] = []
    per_repo: dict[str, list[AggregatedMetrics]] = {}
