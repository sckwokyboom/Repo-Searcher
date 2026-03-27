from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    github_token: str = ""
    data_dir: Path = Path(__file__).parent.parent / "data"
    repos_dir: Path = Path(__file__).parent.parent / "data" / "repos"
    indexes_dir: Path = Path(__file__).parent.parent / "data" / "indexes"

    unixcoder_model: str = "microsoft/unixcoder-base"
    qwen_model: str = "Qwen/Qwen2.5-Coder-1.5B"

    bm25_top_k: int = 30
    faiss_top_k: int = 30
    rrf_k: int = 60
    rrf_top_k: int = 30
    reranker_top_k: int = 10

    embedding_batch_size: int = 32
    embedding_dim: int = 768

    mcts_iterations: int = 2
    mcts_children: int = 3

    graph_mcts_iterations: int = 3
    graph_mcts_reward_threshold: float = 0.3
    graph_mcts_max_discoveries: int = 3

    frontend_url: str = "http://localhost:7860"
    dist_path: Path = Path(__file__).parent.parent.parent / "frontend" / "dist"

    model_config = {"env_prefix": "CODEGRAPH_", "env_file": ".env"}


settings = Settings()

settings.repos_dir.mkdir(parents=True, exist_ok=True)
settings.indexes_dir.mkdir(parents=True, exist_ok=True)
