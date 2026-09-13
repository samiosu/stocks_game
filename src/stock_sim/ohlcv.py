"""Relative OHLCV representations used by the autoregressive generator."""

from __future__ import annotations

import numpy as np

from .constants import OHLCV_FIELDS, SECTOR_IDS

RELATIVE_OHLCV_FIELDS = (
    "gap_log_return",
    "body_log_return",
    "upper_wick_log_range",
    "lower_wick_log_range",
    "log_volume_ratio",
)


def relative_column_names() -> list[str]:
    return [
        f"{sector_id}__{field}"
        for sector_id in SECTOR_IDS
        for field in RELATIVE_OHLCV_FIELDS
    ]


def _bars(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim < 2 or result.shape[-2:] != (len(SECTOR_IDS), len(OHLCV_FIELDS)):
        raise ValueError(f"{name} must end with shape [11, 5], got {result.shape}")
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError(f"{name} must contain finite positive OHLCV values")
    return result


def _validate_volume_reference(volume_reference: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    reference = np.asarray(volume_reference, dtype=np.float64)
    if reference.shape != shape:
        raise ValueError(f"volume_reference must have shape {shape}, got {reference.shape}")
    if not np.isfinite(reference).all() or (reference <= 0).any():
        raise ValueError("volume_reference must contain finite positive values")
    return reference


def geometric_volume_reference(bars: np.ndarray, lookback: int = 20) -> np.ndarray:
    """Return each sector's trailing geometric-mean volume."""

    values = _bars(bars, "bars")
    if values.ndim != 3:
        raise ValueError(f"bars must have shape [time, 11, 5], got {values.shape}")
    if lookback < 1 or lookback > values.shape[0]:
        raise ValueError("lookback must be between one and the number of bars")
    return np.exp(np.mean(np.log(values[-lookback:, :, 4]), axis=0))


def bars_to_relative(
    current: np.ndarray,
    previous: np.ndarray,
    *,
    volume_reference: np.ndarray | None = None,
) -> np.ndarray:
    """Convert raw OHLCV bars to gap/body/wick/volume-ratio values."""

    current_values = _bars(current, "current")
    previous_values = _bars(previous, "previous")
    if current_values.shape != previous_values.shape:
        raise ValueError("current and previous bars must have the same shape")
    open_values = current_values[..., 0]
    high_values = current_values[..., 1]
    low_values = current_values[..., 2]
    close_values = current_values[..., 3]
    previous_close = previous_values[..., 3]
    if volume_reference is None:
        volume_reference_values = previous_values[..., 4]
    else:
        volume_reference_values = _validate_volume_reference(volume_reference, current_values.shape[:-1])
    body_high = np.maximum(open_values, close_values)
    body_low = np.minimum(open_values, close_values)
    return np.stack(
        [
            np.log(open_values / previous_close),
            np.log(close_values / open_values),
            np.log(high_values / body_high),
            np.log(body_low / low_values),
            np.log(current_values[..., 4] / volume_reference_values),
        ],
        axis=-1,
    )


def _relative_limits(relative_clip: np.ndarray | list[float] | tuple[float, ...] | None) -> np.ndarray:
    if relative_clip is None:
        return np.asarray([0.12, 0.12, 0.08, 0.08, 1.0], dtype=np.float64)
    values = np.asarray(relative_clip, dtype=np.float64)
    if values.shape != (len(RELATIVE_OHLCV_FIELDS),) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("relative_clip must contain five positive finite limits")
    return values


def relative_to_bars(
    relative: np.ndarray,
    previous: np.ndarray,
    *,
    volume_reference: np.ndarray | None = None,
    relative_clip: np.ndarray | list[float] | tuple[float, ...] | None = None,
) -> np.ndarray:
    """Reconstruct raw OHLCV from relative predictions and recent volume."""

    relative_values = np.asarray(relative, dtype=np.float64)
    if relative_values.ndim < 2 or relative_values.shape[-2:] != (len(SECTOR_IDS), len(RELATIVE_OHLCV_FIELDS)):
        raise ValueError(f"relative must end with shape [11, 5], got {relative_values.shape}")
    if not np.isfinite(relative_values).all():
        raise FloatingPointError("Relative OHLCV prediction contains NaN or infinite values")
    previous_values = _bars(previous, "previous")
    if relative_values.shape[:-1] != previous_values.shape[:-1]:
        raise ValueError("relative and previous bars must have compatible shapes")
    if volume_reference is None:
        volume_reference_values = previous_values[..., 4]
    else:
        volume_reference_values = _validate_volume_reference(volume_reference, relative_values.shape[:-1])
    limits = _relative_limits(relative_clip)
    bounded = np.clip(relative_values, -limits, limits)
    bounded[..., 2:4] = np.maximum(bounded[..., 2:4], 0.0)
    previous_close = previous_values[..., 3]
    open_values = previous_close * np.exp(bounded[..., 0])
    close_values = open_values * np.exp(bounded[..., 1])
    body_high = np.maximum(open_values, close_values)
    body_low = np.minimum(open_values, close_values)
    high_values = body_high * np.exp(bounded[..., 2])
    low_values = body_low * np.exp(-bounded[..., 3])
    volume_values = volume_reference_values * np.exp(bounded[..., 4])
    result = np.stack(
        [open_values, high_values, low_values, close_values, volume_values],
        axis=-1,
    )
    if not np.isfinite(result).all() or (result <= 0).any():
        raise FloatingPointError("Relative OHLCV reconstruction produced invalid values")
    if not (result[..., 1] >= np.maximum(result[..., 0], result[..., 3])).all():
        raise AssertionError("High must contain the candle body")
    if not (result[..., 2] <= np.minimum(result[..., 0], result[..., 3])).all():
        raise AssertionError("Low must contain the candle body")
    return result
