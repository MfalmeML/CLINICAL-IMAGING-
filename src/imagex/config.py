"""Configuration loading. Single source of truth for pipeline parameters."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file. Raises on missing file or malformed YAML."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such config file: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data
