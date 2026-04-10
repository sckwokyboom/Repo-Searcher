from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    github_token: str = ""
    data_dir: Path = Path(__file__).parent.parent / "data"
    repos_dir: Path = Path(__file__).parent.parent / "data" / "repos"
    indexes_dir: Path = Path(__file__).parent.parent / "data" / "indexes"

    qwen_model: str = "Qwen/Qwen2.5-Coder-1.5B"

    bm25_top_k: int = 30

    frontend_url: str = "http://localhost:7860"
    dist_path: Path = Path(__file__).parent.parent.parent / "frontend" / "dist"

    # LoRA settings
    lora_adapters_dir: Path = Path(__file__).parent.parent / "data" / "lora_adapters"
    default_lora_repo_id: str = "jdereg__java-util"
    default_lora_adapter_path: Path = (
        Path(__file__).parent.parent.parent
        / "benchmark"
        / "lora_training"
        / "output"
        / "rewriter_lora_v2"
        / "final"
    )
    lora_epochs: int = 3
    lora_batch_size: int = 2
    lora_gradient_accumulation: int = 8
    lora_lr: float = 1e-4

    model_config = {"env_prefix": "CODEGRAPH_", "env_file": ".env"}


settings = Settings()

settings.repos_dir.mkdir(parents=True, exist_ok=True)
settings.indexes_dir.mkdir(parents=True, exist_ok=True)
settings.lora_adapters_dir.mkdir(parents=True, exist_ok=True)
