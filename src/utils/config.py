from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_namespace(value: Any) -> Any:
    """Recursively convert dictionaries into SimpleNamespace objects."""
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def load_config(path: str | Path, as_namespace: bool = False) -> dict[str, Any] | SimpleNamespace:
    """Read a YAML config file.

    Args:
        path: YAML file path.
        as_namespace: When True, nested dictionaries can be accessed as attributes.

    Returns:
        A plain dict by default, or a SimpleNamespace tree.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    return _to_namespace(config) if as_namespace else config
