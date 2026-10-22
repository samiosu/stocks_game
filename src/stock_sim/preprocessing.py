"""Leakage-safe scaling and chronological split utilities."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_parent


class FeatureScaler:
    """A small serializable standard scaler with explicit feature names."""

    def __init__(self, columns: Iterable[str] | None = None) -> None:
        self.columns = list(columns or [])
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.mean_ is not None and self.scale_ is not None

    def fit(self, frame: pd.DataFrame | np.ndarray, columns: Iterable[str] | None = None) -> FeatureScaler:
        if isinstance(frame, pd.DataFrame):
            if columns is not None:
                self.columns = list(columns)
            elif not self.columns:
                self.columns = list(frame.columns)
            values = frame[self.columns].to_numpy(dtype=np.float64)
        else:
            values = np.asarray(frame, dtype=np.float64)
            if not self.columns:
                self.columns = [f"feature_{i}" for i in range(values.shape[-1])]
        if values.ndim != 2:
            raise ValueError("Scaler fit expects a 2D feature matrix")
        if not np.isfinite(values).all():
            raise ValueError("Cannot fit scaler on NaN or infinite values")
        self.mean_ = values.mean(axis=0)
        scale = values.std(axis=0)
        self.scale_ = np.where(scale < 1e-12, 1.0, scale)
        return self

    def _values(self, frame: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, bool]:
        if not self.fitted:
            raise RuntimeError("FeatureScaler has not been fit")
        if isinstance(frame, pd.DataFrame):
            return frame[self.columns].to_numpy(dtype=np.float64), True
        return np.asarray(frame, dtype=np.float64), False

    def transform(self, frame: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        values, is_frame = self._values(frame)
        transformed = (values - self.mean_) / self.scale_
        if not np.isfinite(transformed).all():
            raise ValueError("Feature transform produced NaN or infinite values")
        if is_frame:
            untouched = frame.drop(columns=self.columns).copy()
            scaled = pd.DataFrame(transformed, index=frame.index, columns=self.columns)
            return pd.concat([untouched, scaled], axis=1).loc[:, frame.columns]
        return transformed

    def inverse_transform(self, frame: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        values, is_frame = self._values(frame)
        restored = values * self.scale_ + self.mean_
        if is_frame:
            untouched = frame.drop(columns=self.columns).copy()
            restored_frame = pd.DataFrame(restored, index=frame.index, columns=self.columns)
            return pd.concat([untouched, restored_frame], axis=1).loc[:, frame.columns]
        return restored

    def save(self, path: str | Path) -> None:
        if not self.fitted:
            raise RuntimeError("FeatureScaler has not been fit")
        metadata = {
            "columns": self.columns,
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }
        self.from_metadata(metadata)
        payload = {"schema_version": 1, **metadata}
        output = ensure_parent(path)
        output.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def from_metadata(cls, metadata: Any) -> FeatureScaler:
        if not isinstance(metadata, dict):
            raise ValueError("Scaler metadata must be an object")
        columns = metadata.get("columns")
        if (
            not isinstance(columns, list)
            or not columns
            or any(not isinstance(column, str) or not column for column in columns)
            or len(set(columns)) != len(columns)
        ):
            raise ValueError("Scaler columns must be unique non-empty strings")
        arrays = []
        for name in ("mean", "scale"):
            values = metadata.get(name)
            if (
                not isinstance(values, list)
                or len(values) != len(columns)
                or any(type(value) not in (int, float) for value in values)
            ):
                raise ValueError(f"Invalid scaler {name}")
            try:
                array = np.asarray(values, dtype=np.float64)
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid scaler {name}") from exc
            if not np.isfinite(array).all() or (name == "scale" and (array <= 0).any()):
                raise ValueError(f"Invalid scaler {name}")
            arrays.append(array)
        scaler = cls(columns)
        scaler.mean_, scaler.scale_ = arrays
        return scaler

    @classmethod
    def load(cls, path: str | Path) -> FeatureScaler:
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Scaler must be JSON. To replace legacy pickle scalers, run "
                "python -m stock_sim.migrate_scalers --config config/config.yaml"
            ) from exc
        if not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
            raise ValueError("Unsupported scaler JSON schema_version")
        return cls.from_metadata(metadata)


def chronological_masks(
    dates: pd.Series | Iterable[dt.date | str],
    *,
    train_end: str = "2021-12-31",
    validation_end: str = "2023-12-31",
) -> dict[str, np.ndarray]:
    values = pd.to_datetime(pd.Series(dates)).dt.normalize()
    train_limit = pd.Timestamp(train_end)
    validation_limit = pd.Timestamp(validation_end)
    return {
        "train": (values <= train_limit).to_numpy(),
        "validation": ((values > train_limit) & (values <= validation_limit)).to_numpy(),
        "test": (values > validation_limit).to_numpy(),
    }


def fit_training_scaler(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    train_end: str = "2021-12-31",
    output_path: str | Path | None = None,
) -> FeatureScaler:
    masks = chronological_masks(frame["date"], train_end=train_end, validation_end=train_end)
    train = frame.loc[masks["train"], feature_columns]
    if train.empty:
        raise ValueError("No training rows available for scaler fit")
    scaler = FeatureScaler(feature_columns).fit(train)
    if output_path is not None:
        scaler.save(output_path)
    return scaler


def assert_finite(frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    columns = columns or list(frame.columns)
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))[0]
        raise ValueError(f"NaN or infinite value at row={bad[0]}, column={columns[bad[1]]}")
