import asyncio
import shutil
from pathlib import Path

from app.config import settings


async def clone_repo(repo_url: str, repo_id: str) -> Path:
    target = settings.repos_dir / repo_id
    if target.exists():
        shutil.rmtree(target)

    clone_url = repo_url
    if not clone_url.endswith(".git"):
        clone_url = clone_url + ".git"
    if clone_url.startswith("https://github.com/"):
        pass
    elif not clone_url.startswith("http"):
        clone_url = f"https://github.com/{clone_url}"

    process = await asyncio.create_subprocess_exec(
        "git", "clone", "--depth", "1", clone_url, str(target),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"git clone failed: {stderr.decode()}")

    return target
