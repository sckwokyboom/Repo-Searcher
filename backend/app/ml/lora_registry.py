import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

_BUNDLED_ADAPTERS_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "benchmark"
    / "lora_training"
    / "output"
)

_BUNDLED_DESCRIPTIONS = {
    "rewriter_lora_v2": "Query Rewriter v2 (trained on jdereg/java-util)",
    "scorer_lora": "Relevance Scorer (trained on jdereg/java-util)",
}


class AdapterSource(StrEnum):
    BUNDLED = "bundled"
    TRAINED = "trained"


@dataclass
class AdapterInfo:
    adapter_id: str
    name: str
    description: str
    path: str
    source: AdapterSource
    trained_for_repo: str | None = None

    def to_dict(self) -> dict:
        return {
            "adapter_id": self.adapter_id,
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "source": self.source,
            "trained_for_repo": self.trained_for_repo,
        }


def _assignments_path() -> Path:
    return settings.lora_adapters_dir / "assignments.json"


def _load_assignments() -> dict[str, str]:
    """Load repo_id -> adapter_id mapping."""
    path = _assignments_path()
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_assignments(assignments: dict[str, str]):
    path = _assignments_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(assignments, f, indent=2)


def list_adapters() -> list[AdapterInfo]:
    adapters: list[AdapterInfo] = []
    seen_ids: set[str] = set()

    if _BUNDLED_ADAPTERS_DIR.is_dir():
        for subdir in sorted(_BUNDLED_ADAPTERS_DIR.iterdir()):
            if not subdir.is_dir():
                continue
            final = subdir / "final"
            if not (final / "adapter_config.json").exists():
                continue
            adapter_id = f"bundled:{subdir.name}"
            desc = _BUNDLED_DESCRIPTIONS.get(
                subdir.name,
                f"Bundled adapter: {subdir.name}",
            )
            adapters.append(
                AdapterInfo(
                    adapter_id=adapter_id,
                    name=subdir.name,
                    description=desc,
                    path=str(final),
                    source=AdapterSource.BUNDLED,
                )
            )
            seen_ids.add(adapter_id)

    if settings.lora_adapters_dir.is_dir():
        for subdir in sorted(settings.lora_adapters_dir.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            final = subdir / "final"
            if not (final / "adapter_config.json").exists():
                continue
            adapter_id = f"trained:{subdir.name}"
            if adapter_id in seen_ids:
                continue
            repo_name = subdir.name.replace("__", "/")
            adapters.append(
                AdapterInfo(
                    adapter_id=adapter_id,
                    name=f"Trained for {repo_name}",
                    description=f"Project-specific adapter trained on {repo_name}",
                    path=str(final),
                    source=AdapterSource.TRAINED,
                    trained_for_repo=subdir.name,
                )
            )
            seen_ids.add(adapter_id)

    return adapters


def _resolve_adapter_path(adapter_id: str) -> Path | None:
    for adapter in list_adapters():
        if adapter.adapter_id == adapter_id:
            return Path(adapter.path)
    return None


def assign_adapter(repo_id: str, adapter_id: str) -> bool:
    path = _resolve_adapter_path(adapter_id)
    if path is None or not path.exists():
        return False
    assignments = _load_assignments()
    assignments[repo_id] = adapter_id
    _save_assignments(assignments)
    return True


def unassign_adapter(repo_id: str):
    assignments = _load_assignments()
    assignments.pop(repo_id, None)
    _save_assignments(assignments)


def get_adapter_path(repo_id: str) -> Path | None:
    assignments = _load_assignments()
    if repo_id in assignments:
        path = _resolve_adapter_path(assignments[repo_id])
        if path and path.exists():
            return path

    adapter_dir = settings.lora_adapters_dir / repo_id / "final"
    if (adapter_dir / "adapter_config.json").exists():
        return adapter_dir

    if repo_id == settings.default_lora_repo_id:
        default = settings.default_lora_adapter_path
        if (default / "adapter_config.json").exists():
            return default

    return None


def get_active_adapter_id(repo_id: str) -> str | None:
    assignments = _load_assignments()
    if repo_id in assignments:
        adapter_id = assignments[repo_id]
        if _resolve_adapter_path(adapter_id):
            return adapter_id

    adapter_dir = settings.lora_adapters_dir / repo_id / "final"
    if (adapter_dir / "adapter_config.json").exists():
        return f"trained:{repo_id}"

    if repo_id == settings.default_lora_repo_id:
        default = settings.default_lora_adapter_path
        if (default / "adapter_config.json").exists():
            return "bundled:rewriter_lora_v2"

    return None


def has_adapter(repo_id: str) -> bool:
    return get_adapter_path(repo_id) is not None
