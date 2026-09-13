"""Chronological 60-day windows for probabilistic sector training."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .preprocessing import FeatureScaler, assert_finite, chronological_masks

LOGGER = logging.getLogger(__name__)


@dataclass
class WindowArrays:
    features: np.ndarray
    target_returns: np.ndarray
    target_log_volatility: np.ndarray
    target_dates: np.ndarray


class MarketWindowDataset:
    """A torch-compatible dataset without importing torch at module import time."""

    def __init__(self, arrays: WindowArrays) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays.target_dates)

    def __getitem__(self, index: int):
        try:
            import torch

            return (
                torch.from_numpy(self.arrays.features[index]).float(),
                torch.from_numpy(self.arrays.target_returns[index]).float(),
                torch.from_numpy(self.arrays.target_log_volatility[index]).float(),
            )
        except ImportError:  # pragma: no cover - torch is a project dependency
            return (
                self.arrays.features[index],
                self.arrays.target_returns[index],
                self.arrays.target_log_volatility[index],
            )


@dataclass
class OHLCVWindowArrays:
    features: np.ndarray
    targets: np.ndarray
    target_dates: np.ndarray


class OHLCVWindowDataset:
    """A torch-compatible dataset for next-bar OHLCV prediction."""

    def __init__(self, arrays: OHLCVWindowArrays, *, sector_count: int, field_count: int) -> None:
        self.arrays = arrays
        self.sector_count = int(sector_count)
        self.field_count = int(field_count)

    def __len__(self) -> int:
        return len(self.arrays.target_dates)

    def __getitem__(self, index: int):
        target = self.arrays.targets[index].reshape(self.sector_count, self.field_count)
        try:
            import torch

            return (
                torch.from_numpy(self.arrays.features[index]).float(),
                torch.from_numpy(target).float(),
            )
        except ImportError:  # pragma: no cover - torch is a project dependency
            return self.arrays.features[index], target


def _target_volatility(
    frame: pd.DataFrame,
    target_index: int,
    volatility_columns: list[str],
    future_returns: np.ndarray,
) -> np.ndarray:
    values = frame.iloc[target_index][volatility_columns].to_numpy(dtype=np.float64)
    values = np.where(np.isfinite(values) & (values > 1e-8), values, np.nan)
    fallback = float(np.std(future_returns, axis=0, ddof=0)) if future_returns.shape[0] > 1 else np.nan
    # The sector vol columns correspond to each sector. For a missing initial
    # value, use a small floor rather than looking into validation/test rows.
    values = np.where(np.isfinite(values), values, max(fallback, 1e-6) if np.isfinite(fallback) else 1e-6)
    return np.log(np.maximum(values, 1e-6))


def make_window_arrays(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    return_columns: list[str],
    volatility_columns: list[str],
    sequence_length: int = 60,
    horizon: int = 1,
    scaler: FeatureScaler | None = None,
) -> WindowArrays:
    """Build windows whose target date is strictly after the input window."""

    if horizon not in {1, 5}:
        raise ValueError("Only horizon=1 and horizon=5 are currently supported")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    work = frame.copy().sort_values("date").reset_index(drop=True)
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    assert_finite(work, feature_columns + return_columns + volatility_columns)
    if scaler is None:
        scaled_values = work[feature_columns].to_numpy(dtype=np.float64)
    else:
        scaled_values = scaler.transform(work[feature_columns])
        scaled_values = np.asarray(scaled_values[feature_columns], dtype=np.float64)
    raw_returns = work[return_columns].to_numpy(dtype=np.float64)
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_vols: list[np.ndarray] = []
    dates: list[np.datetime64] = []
    last_input_end = len(work) - horizon - 1
    for input_end in range(sequence_length - 1, last_input_end + 1):
        future_slice = raw_returns[input_end + 1 : input_end + horizon + 1]
        if future_slice.shape[0] != horizon:
            continue
        target_index = input_end + horizon
        features.append(scaled_values[input_end - sequence_length + 1 : input_end + 1])
        targets.append(future_slice.sum(axis=0))
        target_vols.append(_target_volatility(work, target_index, volatility_columns, future_slice))
        dates.append(work.iloc[target_index]["date"].to_datetime64())
    if not features:
        return WindowArrays(
            np.empty((0, sequence_length, len(feature_columns)), dtype=np.float32),
            np.empty((0, len(return_columns)), dtype=np.float32),
            np.empty((0, len(volatility_columns)), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
        )
    result = WindowArrays(
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(target_vols, dtype=np.float32),
        np.asarray(dates, dtype="datetime64[ns]"),
    )
    if not np.isfinite(result.features).all() or not np.isfinite(result.target_returns).all() or not np.isfinite(result.target_log_volatility).all():
        raise ValueError("Window arrays contain NaN or infinite values")
    return result


def make_split_datasets(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    return_columns: list[str],
    volatility_columns: list[str],
    scaler: FeatureScaler,
    sequence_length: int = 60,
    horizon: int = 1,
    train_end: str = "2021-12-31",
    validation_end: str = "2023-12-31",
) -> dict[str, MarketWindowDataset]:
    arrays = make_window_arrays(
        frame,
        feature_columns=feature_columns,
        return_columns=return_columns,
        volatility_columns=volatility_columns,
        sequence_length=sequence_length,
        horizon=horizon,
        scaler=scaler,
    )
    masks = chronological_masks(arrays.target_dates, train_end=train_end, validation_end=validation_end)
    result: dict[str, MarketWindowDataset] = {}
    for name, mask in masks.items():
        split = WindowArrays(
            arrays.features[mask],
            arrays.target_returns[mask],
            arrays.target_log_volatility[mask],
            arrays.target_dates[mask],
        )
        result[name] = MarketWindowDataset(split)
        LOGGER.info("%s windows: %s", name, len(result[name]))
    return result


def make_ohlcv_window_arrays(
    frame: pd.DataFrame,
    *,
    ohlcv_columns: list[str],
    sequence_length: int = 60,
    horizon: int = 1,
    scaler: FeatureScaler | None = None,
) -> OHLCVWindowArrays:
    """Build fixed-history windows and one future standardized OHLCV target."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    work = frame.copy().sort_values("date").reset_index(drop=True)
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    assert_finite(work, ohlcv_columns)
    if scaler is None:
        scaled_values = work[ohlcv_columns].to_numpy(dtype=np.float64)
    else:
        scaled_values = np.asarray(scaler.transform(work[ohlcv_columns]), dtype=np.float64)
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    dates: list[np.datetime64] = []
    last_input_end = len(work) - horizon - 1
    for input_end in range(sequence_length - 1, last_input_end + 1):
        target_index = input_end + horizon
        features.append(scaled_values[input_end - sequence_length + 1 : input_end + 1])
        targets.append(scaled_values[target_index])
        dates.append(work.iloc[target_index]["date"].to_datetime64())
    if not features:
        return OHLCVWindowArrays(
            np.empty((0, sequence_length, len(ohlcv_columns)), dtype=np.float32),
            np.empty((0, len(ohlcv_columns)), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
        )
    result = OHLCVWindowArrays(
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(dates, dtype="datetime64[ns]"),
    )
    if not np.isfinite(result.features).all() or not np.isfinite(result.targets).all():
        raise ValueError("OHLCV window arrays contain NaN or infinite values")
    return result


def make_ohlcv_split_datasets(
    frame: pd.DataFrame,
    *,
    ohlcv_columns: list[str],
    scaler: FeatureScaler,
    sector_count: int,
    field_count: int,
    sequence_length: int = 60,
    horizon: int = 1,
    train_end: str = "2021-12-31",
    validation_end: str = "2023-12-31",
) -> dict[str, OHLCVWindowDataset]:
    arrays = make_ohlcv_window_arrays(
        frame,
        ohlcv_columns=ohlcv_columns,
        sequence_length=sequence_length,
        horizon=horizon,
        scaler=scaler,
    )
    masks = chronological_masks(arrays.target_dates, train_end=train_end, validation_end=validation_end)
    result: dict[str, OHLCVWindowDataset] = {}
    for name, mask in masks.items():
        split = OHLCVWindowArrays(
            arrays.features[mask],
            arrays.targets[mask],
            arrays.target_dates[mask],
        )
        result[name] = OHLCVWindowDataset(split, sector_count=sector_count, field_count=field_count)
        LOGGER.info("%s OHLCV windows: %s", name, len(result[name]))
    return result
