"""Configuration loading and small runtime helpers."""

from __future__ import annotations

import copy
import datetime as dt
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOGGER = logging.getLogger(__name__)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""

    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load a YAML config and annotate it with its project root."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Configuration must be a mapping: {config_path}")
    loaded["_config_path"] = str(config_path)
    root = config_path.parent.parent if config_path.parent.name == "config" else Path.cwd()
    loaded["_project_root"] = str(root)
    return loaded


def project_root(config: dict[str, Any]) -> Path:
    return Path(config.get("_project_root", Path.cwd()))


def config_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a path from the ``paths`` section relative to the project root."""

    value = config.get("paths", {}).get(key)
    if value is None:
        raise KeyError(f"Missing paths.{key} in configuration")
    path = Path(value)
    return path if path.is_absolute() else project_root(config) / path


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_date(value: Any, *, default: dt.date | None = None) -> dt.date:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        if default is None:
            raise ValueError("A date is required")
        return default
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    parsed = dt.date.fromisoformat(str(value)[:10])
    return parsed


def runtime_as_of(config: dict[str, Any], explicit: str | None = None) -> dt.date:
    value = explicit if explicit is not None else config.get("data", {}).get("as_of")
    return parse_date(value, default=dt.datetime.now(dt.timezone.utc).date())


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch when torch is installed."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass
