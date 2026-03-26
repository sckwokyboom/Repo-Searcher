import asyncio
import shutil
from pathlib import Path

import pygit2

from app.config import settings


async def clone_repo(repo_url: str, repo_id: str) -> Path:
    target = settings.repos_dir / repo_id
    if target.exists():
        shutil.rmtree(target)

    clone_url = repo_url
    if not clone_url.endswith(".git"):
        clone_url = clone_url + ".git"
    elif not clone_url.startswith("http"):
        clone_url = f"https://github.com/{clone_url}"

    repo = await asyncio.to_thread(
        pygit2.clone_repository, clone_url, str(target), depth=1
    )

    return Path(repo.path)
