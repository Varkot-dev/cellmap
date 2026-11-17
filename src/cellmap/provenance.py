"""Record where a number came from, so every saved result can be traced to code."""

import subprocess
from importlib.metadata import version
from pathlib import Path


def git_commit(root: Path) -> str:
    """Current commit hash, with "+dirty" if tracked files outside `results/` have changes.

    Results files are excluded because scripts rewrite them as they run: they say nothing about
    whether the code that produced them was committed.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"],
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
