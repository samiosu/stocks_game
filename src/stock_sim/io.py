"""Small storage helpers with explicit Parquet errors."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ensure_parent


def write_parquet(frame: pd.DataFrame, path: str | Path) -> None:
    path = ensure_parent(path)
    try:
        frame.to_parquet(path, index=False)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("pyarrow or fastparquet is required for Parquet storage") from exc


def read_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_parquet(path)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("pyarrow or fastparquet is required for Parquet storage") from exc

