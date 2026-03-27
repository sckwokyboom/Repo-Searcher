"""
LoRA adapter registry — discovers available adapters and tracks
which adapter is assigned to which repo.

Adapters are discovered from:
  1. benchmark/lora_training/output/  (bundled pre-trained adapters)
  2. settings.lora_adapters_dir/      (user-trained per-repo adapters)

Manual assignments (using a bundled adapter for a different repo)
are stored in settings.lora_adapters_dir / "assignments.json".
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

# Well-known bundled adapters with human-readable descriptions
_BUNDLED_ADAPTERS_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "benchmark" / "lora_training" / "output"
)

_BUNDLED_DESCRIPTIONS = {
    "rewriter_lora_v2": "Query Rewriter v2 (trained on jdereg/java-util)",
    "scorer_lora": "Relevance Scorer (trained on jdereg/java-util)",
}


@dataclass
class AdapterInfo:
    """Metadata about a discovered LoRA adapter."""
    adapter_id: str
    name: str
    description: str
    path: str
    source: str  # "bundled" | "trained"
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
    """Discover all available LoRA adapters."""
    adapters: list[AdapterInfo] = []
    seen_ids: set[str] = set()

    # 1. Bundled adapters from benchmark/lora_training/output/
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
            adapters.append(AdapterInfo(
                adapter_id=adapter_id,
                name=subdir.name,
                description=desc,
                path=str(final),
                source="bundled",
            ))
            seen_ids.add(adapter_id)

    # 2. User-trained adapters from lora_adapters_dir/
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
            adapters.append(AdapterInfo(
                adapter_id=adapter_id,
                name=f"Trained for {repo_name}",
                description=f"Project-specific adapter trained on {repo_name}",
                path=str(final),
                source="trained",
                trained_for_repo=subdir.name,
            ))
            seen_ids.add(adapter_id)

    return adapters


def _resolve_adapter_path(adapter_id: str) -> Path | None:
    """Resolve an adapter_id to its filesystem path."""
    for adapter in list_adapters():
        if adapter.adapter_id == adapter_id:
            return Path(adapter.path)
    return None


def assign_adapter(repo_id: str, adapter_id: str) -> bool:
    """Assign an adapter to a repo. Returns True if successful."""
    path = _resolve_adapter_path(adapter_id)
    if path is None or not path.exists():
        return False
    assignments = _load_assignments()
    assignments[repo_id] = adapter_id
    _save_assignments(assignments)
    return True


def unassign_adapter(repo_id: str):
    """Remove adapter assignment for a repo."""
    assignments = _load_assignments()
    assignments.pop(repo_id, None)
    _save_assignments(assignments)


def get_adapter_path(repo_id: str) -> Path | None:
    """Return the filesystem path to the active LoRA adapter for a repo."""
    # 1. Check manual assignment
    assignments = _load_assignments()
    if repo_id in assignments:
        path = _resolve_adapter_path(assignments[repo_id])
        if path and path.exists():
            return path

    # 2. Check repo-specific trained adapter
    adapter_dir = settings.lora_adapters_dir / repo_id / "final"
    if (adapter_dir / "adapter_config.json").exists():
        return adapter_dir

    # 3. Fall back to default adapter for the bundled repo
    if repo_id == settings.default_lora_repo_id:
        default = settings.default_lora_adapter_path
        if (default / "adapter_config.json").exists():
            return default

    return None


def get_active_adapter_id(repo_id: str) -> str | None:
    """Return the adapter_id currently assigned/active for a repo."""
    # 1. Manual assignment
    assignments = _load_assignments()
    if repo_id in assignments:
        adapter_id = assignments[repo_id]
        if _resolve_adapter_path(adapter_id):
            return adapter_id

    # 2. Trained adapter
    adapter_dir = settings.lora_adapters_dir / repo_id / "final"
    if (adapter_dir / "adapter_config.json").exists():
        return f"trained:{repo_id}"

    # 3. Default
    if repo_id == settings.default_lora_repo_id:
        default = settings.default_lora_adapter_path
        if (default / "adapter_config.json").exists():
            return "bundled:rewriter_lora_v2"

    return None


def has_adapter(repo_id: str) -> bool:
    """Check if any LoRA adapter is active for the given repo."""
    return get_adapter_path(repo_id) is not None
