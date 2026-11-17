"""Record where a number came from, so every saved result can be traced to code."""

import subprocess
from importlib.metadata import version
from pathlib import Path


def git_commit(root: Path) -> str:
    """Current commit hash, with "+dirty" appended if tracked files have uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if dirty else "")


def versions(*packages: str) -> dict[str, str]:
    """Installed versions of the named packages."""
    return {p: version(p) for p in packages}
