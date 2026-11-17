"""Load YAML config files."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file into a plain dictionary."""
    with open(path) as f:
        return yaml.safe_load(f)
