"""Record where a number came from, so every saved result can be traced to code."""

import subprocess
from importlib.metadata import version
from pathlib import Path


def git_commit(root: Path) -> str:
    """Current commit hash, with "+dirty" if files outside `results/` differ from it.

    Results files are excluded because scripts rewrite them as they run: they say nothing about
    whether the code that produced them was committed. Untracked files count as dirty (ignored
    ones such as data/ do not): a new script run before it is committed would otherwise be
    stamped with a commit that does not contain it, which an independent review found had
    happened to results/leakage/splits.csv.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal", "--", ".", ":!results"],
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
