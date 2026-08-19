"""Generate stochastic sector prices from a trained GRU."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import config_path, ensure_parent, load_config, seed_everything, setup_logging
from .constants import SECTOR_DEFINITIONS, SECTOR_IDS
from .io import read_parquet, write_parquet
from .model import MarketGRU, require_torch, sample_returns, sigma_from_log_sigma
from .preprocessing import FeatureScaler

LOGGER = logging.getLogger(__name__)

OHLC_FIELDS = ("open", "high", "low", "close")


def prices_from_returns(
    returns: np.ndarray | Iterable[Iterable[float]],
    *,
    initial_price: float | Iterable[float] = 100.0,
) -> np.ndarray:
    """Convert log returns to finite price paths without predicting prices."""

    values = np.asarray(returns, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("returns must have shape [days, sectors]")
    initial = np.asarray(initial_price, dtype=np.float64)
    if initial.ndim == 0:
        initial = np.full(values.shape[1], float(initial))
    if initial.shape != (values.shape[1],):
        raise ValueError("initial_price must be scalar or one value per sector")
    if not np.isfinite(values).all() or not np.isfinite(initial).all() or (initial <= 0).any():
        raise ValueError("returns and initial_price must be finite; initial prices must be positive")
    prices = initial[None, :] * np.exp(np.cumsum(values, axis=0))
    if not np.isfinite(prices).all():
        raise FloatingPointError("Generated prices became non-finite")
    return prices


def ohlc_from_returns(
    returns: np.ndarray | Iterable[Iterable[float]],
    close_prices: np.ndarray | Iterable[Iterable[float]],
    *,
    initial_price: float | Iterable[float] = 100.0,
    seed: int = 42,
    gap_ratio: float = 0.35,
    range_scale: float = 0.8,
    minimum_range: float = 0.001,
) -> dict[str, np.ndarray]:
    """Create continuous OHLC bars while preserving the generated closes.

    The model predicts close-to-close log returns.  An overnight gap is sampled
    first, and the intraday candle is then constructed so that its close is
    exactly the model-generated close.  High and low are always outside the
    open/close body, which makes the result safe for candlestick renderers.
    """

    values = np.asarray(returns, dtype=np.float64)
    closes = np.asarray(close_prices, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if closes.ndim == 1:
        closes = closes[:, None]
    if values.ndim != 2 or closes.shape != values.shape:
        raise ValueError("returns and close_prices must both have shape [days, sectors]")
    initial = np.asarray(initial_price, dtype=np.float64)
    if initial.ndim == 0:
        initial = np.full(values.shape[1], float(initial))
    if initial.shape != (values.shape[1],):
        raise ValueError("initial_price must be scalar or one value per sector")
    if not np.isfinite(values).all() or not np.isfinite(closes).all() or (closes <= 0).any():
        raise ValueError("returns and close_prices must be finite and positive where applicable")
    if not np.isfinite(initial).all() or (initial <= 0).any():
        raise ValueError("initial_price must be finite and positive")
    if gap_ratio < 0 or range_scale < 0 or minimum_range <= 0:
        raise ValueError("gap_ratio and range_scale must be non-negative; minimum_range must be positive")

    previous_close = np.vstack([initial[None, :], closes[:-1]])
    rng = np.random.default_rng(int(seed))
    gap_noise_scale = np.maximum(np.abs(values) * 0.20, minimum_range * 0.5)
    gaps = float(gap_ratio) * values + rng.normal(scale=gap_noise_scale, size=values.shape)
    opens = previous_close * np.exp(gaps)
    intraday_returns = np.log(closes / opens)
    excursion_scale = np.maximum(np.abs(intraday_returns) * float(range_scale), minimum_range)
    upper_excursion = np.abs(rng.normal(scale=excursion_scale, size=values.shape))
    lower_excursion = np.abs(rng.normal(scale=excursion_scale, size=values.shape))
    highs = np.maximum(opens, closes) * np.exp(upper_excursion)
    lows = np.minimum(opens, closes) * np.exp(-lower_excursion)
    result = {"open": opens, "high": highs, "low": lows, "close": closes}
    for name, array in result.items():
        if not np.isfinite(array).all() or (array <= 0).any():
            raise FloatingPointError(f"Generated OHLC {name} became invalid")
    if not (highs >= np.maximum(opens, closes)).all() or not (lows <= np.minimum(opens, closes)).all():
        raise AssertionError("OHLC invariant was violated")
    return result


def ohlc_frame(
    dates: Iterable[Any],
    returns: np.ndarray | Iterable[Iterable[float]],
    close_prices: np.ndarray | Iterable[Iterable[float]],
    *,
    initial_price: float | Iterable[float] = 100.0,
    seed: int = 42,
    gap_ratio: float = 0.35,
    range_scale: float = 0.8,
    minimum_range: float = 0.001,
) -> pd.DataFrame:
    """Return a date-indexed-compatible frame with close aliases and OHLC columns."""

    values = np.asarray(returns, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    bars = ohlc_from_returns(
        values,
        close_prices,
        initial_price=initial_price,
        seed=seed,
        gap_ratio=gap_ratio,
        range_scale=range_scale,
        minimum_range=minimum_range,
    )
    date_values = pd.to_datetime(list(dates)).normalize()
    if len(date_values) != values.shape[0]:
        raise ValueError("dates must contain one value per generated day")
    result = pd.DataFrame({"date": date_values})
    # Keep the original sector-id columns as close aliases for evaluation and
    # existing Unity/data consumers.
    for index, sector_id in enumerate(SECTOR_IDS):
        result[sector_id] = bars["close"][:, index]
    for sector_index, sector_id in enumerate(SECTOR_IDS):
        for field in OHLC_FIELDS:
            result[f"{sector_id}__{field}"] = bars[field][:, sector_index]
    return result


def extract_ohlc_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Select only date and ``sector__open/high/low/close`` columns."""

    columns = ["date"] + [f"{sector_id}__{field}" for sector_id in SECTOR_IDS for field in OHLC_FIELDS]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"OHLC frame is missing columns: {missing[:5]}")
    return frame[columns].copy()


def save_candlestick_chart(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    sector_id: str = "energy",
    title: str | None = None,
) -> Path:
    """Render one generated sector with mplfinance and save a PNG image."""

    if sector_id not in SECTOR_IDS:
        raise ValueError(f"Unknown sector_id: {sector_id}")
    try:
        import mplfinance as mpf
    except ImportError as exc:  # pragma: no cover - dependency is in the reports extra
        raise RuntimeError("Install the reports extra to render candlesticks: pip install '.[reports]'") from exc
    columns = {field: f"{sector_id}__{field}" for field in OHLC_FIELDS}
    missing = [column for column in columns.values() if column not in frame.columns]
    if "date" not in frame.columns or missing:
        raise ValueError(f"Frame does not contain OHLC columns for {sector_id}")
    chart = frame[["date", *columns.values()]].copy()
    chart["date"] = pd.to_datetime(chart["date"])
    chart = chart.rename(columns={value: key.capitalize() for key, value in columns.items()}).set_index("date")
    output = ensure_parent(path)
    mpf.plot(
        chart,
        type="candle",
        style="yahoo",
        volume=False,
        title=title or f"Generated {sector_id} OHLC",
        savefig={"fname": str(output), "dpi": 150, "bbox_inches": "tight"},
        closefig=True,
    )
    return output


def _load_checkpoint(path: str | Path, device: Any) -> tuple[Any, dict[str, Any]]:
    require_torch()
    import torch

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.6
        checkpoint = torch.load(path, map_location=device)
    model = MarketGRU(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _event_list(event_schedule: Any) -> list[dict[str, Any]]:
    if event_schedule is None:
        return []
    if isinstance(event_schedule, dict):
        event_schedule = event_schedule.get("events", [])
    if not isinstance(event_schedule, list):
        raise TypeError("event_schedule must be a list or a mapping with an events list")
    return [dict(item) for item in event_schedule]


def _affected_ids(value: Any) -> set[str]:
    if value is None or value == "" or value == []:
        return set(SECTOR_IDS)
    values = value if isinstance(value, (list, tuple, set)) else [value]
    result: set[str] = set()
    names = {name: sector_id for sector_id, name in SECTOR_DEFINITIONS}
    for item in values:
        text = str(item)
        if text in SECTOR_IDS:
            result.add(text)
        elif text in names:
            result.add(names[text])
    return result


def event_vector(
    date: str | dt.date | pd.Timestamp,
    event_schedule: Any,
    event_columns: list[str],
) -> np.ndarray:
    """Build type and sector-specific event inputs with exponential decay."""

    current = pd.Timestamp(date).date()
    values = np.zeros(len(event_columns), dtype=np.float64)
    column_index = {column: index for index, column in enumerate(event_columns)}
    for event in _event_list(event_schedule):
        event_type = str(event.get("event_type", "")).strip()
        if not event_type:
            LOGGER.warning("Ignoring event without event_type: %s", event)
            continue
        try:
            start = pd.Timestamp(event.get("start_date", event.get("date"))).date()
        except (TypeError, ValueError):
            LOGGER.warning("Ignoring event with invalid date: %s", event)
            continue
        intensity = float(event.get("intensity", 1.0))
        duration = max(1, int(event.get("duration", 1)))
        elapsed = (current - start).days
        if elapsed < 0 or elapsed >= duration:
            continue
        decay_rate = max(0.0, float(event.get("decay_rate", 0.0)))
        amount = intensity * np.exp(-decay_rate * elapsed)
        affected = _affected_ids(event.get("affected_sectors"))
        global_column = f"event__{event_type}"
        if global_column in column_index:
            values[column_index[global_column]] += amount
        for sector_id in affected:
            sector_column = f"event__{event_type}__{sector_id}"
            if sector_column in column_index:
                values[column_index[sector_column]] += amount
    return values


def _set_if_present(row: pd.Series, column: str, value: float) -> None:
    if column in row.index:
        row[column] = float(value)


def _advance_raw_row(
    previous: pd.Series,
    *,
    date: pd.Timestamp,
    returns: np.ndarray,
    generated_returns: list[np.ndarray],
    event_values: np.ndarray,
    event_columns: list[str],
) -> pd.Series:
    row = previous.copy()
    row["date"] = date
    for index, (sector_id, _) in enumerate(SECTOR_DEFINITIONS):
        _set_if_present(row, f"{sector_id}__return_1d", returns[index])
        history = generated_returns[-19:] + [returns]
        five = np.asarray(history[-5:])[:, index]
        twenty = np.asarray(history[-20:])[:, index]
        _set_if_present(row, f"{sector_id}__return_5d", float(five.sum()))
        _set_if_present(row, f"{sector_id}__vol_5d", float(np.std(five, ddof=0)))
        _set_if_present(row, f"{sector_id}__vol_20d", float(np.std(twenty, ddof=0)))
    n225 = float(np.mean(returns))
    topix = float(np.mean(returns))
    _set_if_present(row, "market__n225_return_1d", n225)
    _set_if_present(row, "market__topix_return_1d", topix)
    _set_if_present(row, "market__return_spread", n225 - topix)
    market_history = generated_returns[-19:] + [returns]
    market_n225 = np.asarray([np.mean(item) for item in market_history])
    _set_if_present(row, "market__n225_volatility", float(np.std(market_n225, ddof=0)))
    _set_if_present(row, "market__topix_volatility", float(np.std(market_n225, ddof=0)))
    _set_if_present(row, "market__up_sector_count", float((returns > 0).sum()))
    _set_if_present(row, "market__down_sector_count", float((returns < 0).sum()))
    _set_if_present(row, "market__sector_cross_std", float(np.std(returns, ddof=0)))
    for column, value in zip(event_columns, event_values, strict=True):
        row[column] = float(value)
    return row


def _bounds(
    checkpoint: dict[str, Any],
    training_frame: pd.DataFrame,
    return_columns: list[str],
    percentile: tuple[float, float] | list[float] | None,
    *,
    hard_clip: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not hard_clip:
        # The model output is sampled from a continuous distribution.  Do not
        # silently turn it into a percentile-bounded two-point distribution.
        return (
            np.full(len(return_columns), -np.inf, dtype=float),
            np.full(len(return_columns), np.inf, dtype=float),
        )
    if percentile is None:
        raw = np.asarray(checkpoint.get("return_bounds", []), dtype=float)
        if raw.shape == (len(return_columns), 2):
            return raw[:, 0], raw[:, 1]
    train_end = pd.Timestamp(checkpoint.get("train_end", "2021-12-31"))
    train = training_frame[pd.to_datetime(training_frame["date"]) <= train_end]
    values = train[return_columns].to_numpy(dtype=float)
    quantiles = percentile or [0.5, 99.5]
    if len(quantiles) != 2 or not (0 <= float(quantiles[0]) < float(quantiles[1]) <= 100):
        raise ValueError("return clipping percentiles must be 0 <= low < high <= 100")
    low, high = np.percentile(values, [float(quantiles[0]), float(quantiles[1])], axis=0)
    return low, high


def generate_price_paths(
    model: Any,
    history_frame: pd.DataFrame,
    scaler: FeatureScaler,
    checkpoint: dict[str, Any],
    *,
    days: int,
    seed: int = 42,
    initial_price: float | Iterable[float] = 100.0,
    event_schedule: Any = None,
    volatility_scale: float = 1.0,
    return_clipping_percentile: tuple[float, float] | list[float] | None = None,
    training_frame: pd.DataFrame | None = None,
    hard_clip: bool = False,
    soft_clip: float | None = 0.08,
    raw_returns_path: str | Path | None = None,
    feature_z_clip: float | None = 6.0,
    volatility_persistence: float = 0.9,
    volatility_shock_scale: float = 0.18,
    generate_ohlc: bool = True,
    ohlc_gap_ratio: float = 0.35,
    ohlc_range_scale: float = 0.8,
    ohlc_minimum_range: float = 0.001,
) -> pd.DataFrame:
    """Autoregressively sample sectors while updating relative features.

    ``raw_returns_path`` stores samples before any safety transform.  Hard
    percentile clipping is opt-in because exact clipping creates artificial
    quantile plateaus.  The default soft clip is continuous and only affects
    extreme outliers.
    """

    require_torch()
    import torch

    if days < 1:
        raise ValueError("days must be positive")
    if volatility_scale < 0:
        raise ValueError("volatility_scale must be non-negative")
    if not 0.0 <= volatility_persistence < 1.0:
        raise ValueError("volatility_persistence must be between 0 and 1")
    if volatility_shock_scale < 0:
        raise ValueError("volatility_shock_scale must be non-negative")
    feature_columns = list(checkpoint["feature_columns"])
    return_columns = list(checkpoint["return_columns"])
    event_columns = list(checkpoint.get("event_columns", [c for c in feature_columns if c.startswith("event__")]))
    sequence_length = int(checkpoint.get("sequence_length", 60))
    if len(history_frame) < sequence_length:
        raise ValueError(f"history_frame needs at least {sequence_length} rows")
    history = history_frame.copy().sort_values("date").reset_index(drop=True)
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    for column in feature_columns + return_columns:
        if column not in history:
            raise ValueError(f"history_frame missing {column}")
    if training_frame is None:
        training_frame = history_frame
    low, high = _bounds(
        checkpoint,
        training_frame,
        return_columns,
        return_clipping_percentile,
        hard_clip=hard_clip,
    )
    if low.shape != (len(SECTOR_IDS),) or high.shape != (len(SECTOR_IDS),):
        raise ValueError("Return clipping bounds must contain one pair per sector")
    if soft_clip is not None and soft_clip <= 0:
        raise ValueError("soft_clip must be positive or None")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    model.eval()
    raw_window = history.tail(sequence_length)[["date", *feature_columns]].copy()
    generated_returns: list[np.ndarray] = []
    raw_sampled_returns: list[np.ndarray] = []
    initial = np.asarray(initial_price, dtype=float)
    if initial.ndim == 0:
        initial = np.full(len(SECTOR_IDS), float(initial))
    if initial.shape != (len(SECTOR_IDS),):
        raise ValueError("initial_price must be scalar or contain 11 sector prices")
    if (initial <= 0).any() or not np.isfinite(initial).all():
        raise ValueError("initial_price must be finite and positive")
    output_dates: list[pd.Timestamp] = []
    volatility_state = 0.0
    for step in range(days):
        next_date = pd.bdate_range(raw_window["date"].iloc[-1] + pd.Timedelta(days=1), periods=1)[0].normalize()
        current_event = event_vector(next_date, event_schedule, event_columns)
        current = raw_window.iloc[-1].copy()
        current["date"] = raw_window["date"].iloc[-1]
        for column, value in zip(event_columns, current_event, strict=True):
            current[column] = value
        model_input = scaler.transform(raw_window[feature_columns].assign(**{column: current[column] for column in event_columns}))
        values = np.asarray(model_input[feature_columns], dtype=np.float32)[None, :, :]
        if feature_z_clip is not None:
            if feature_z_clip <= 0:
                raise ValueError("feature_z_clip must be positive or None")
            values = np.clip(values, -float(feature_z_clip), float(feature_z_clip))
        x = torch.from_numpy(values).to(device)
        volatility_innovation = float(torch.randn(1, generator=generator, device=device).item())
        volatility_state = (
            volatility_persistence * volatility_state
            + np.sqrt(1.0 - volatility_persistence**2) * volatility_innovation
        )
        volatility_multiplier = float(
            np.exp(volatility_shock_scale * volatility_state - 0.5 * volatility_shock_scale**2)
        )
        with torch.no_grad():
            params = model(x)
            sampled = sample_returns(
                params,
                generator=generator,
                volatility_scale=volatility_scale * volatility_multiplier,
                factor_loadings=getattr(model, "factor_loadings", None),
                common_noise_weight=float(getattr(model, "common_noise_weight", 0.0)),
            )
            if step == 0:
                sigma = sigma_from_log_sigma(params[..., 1])
                LOGGER.info(
                    "generation units: mu=[%.6f, %.6f], log_sigma=[%.6f, %.6f], sigma=[%.6f, %.6f]",
                    float(params[..., 0].min()),
                    float(params[..., 0].max()),
                    float(params[..., 1].min()),
                    float(params[..., 1].max()),
                    float(sigma.min()),
                    float(sigma.max()),
                )
        returns = sampled[0].detach().cpu().numpy().astype(float)
        raw_sampled_returns.append(returns.copy())
        if soft_clip is not None:
            returns = float(soft_clip) * np.tanh(returns / float(soft_clip))
        if hard_clip:
            returns = np.clip(returns, low, high)
        if not np.isfinite(returns).all():
            raise FloatingPointError("Model generated a non-finite return")
        generated_returns.append(returns)
        next_row = _advance_raw_row(
            current,
            date=next_date,
            returns=returns,
            generated_returns=generated_returns[:-1],
            event_values=current_event,
            event_columns=event_columns,
        )
        raw_window = pd.concat([raw_window, next_row.to_frame().T], ignore_index=True).tail(sequence_length)
        output_dates.append(next_date)
    prices = prices_from_returns(np.asarray(generated_returns), initial_price=initial)
    if generate_ohlc:
        result = ohlc_frame(
            output_dates,
            np.asarray(generated_returns),
            prices,
            initial_price=initial,
            seed=int(seed) + 1_000_003,
            gap_ratio=ohlc_gap_ratio,
            range_scale=ohlc_range_scale,
            minimum_range=ohlc_minimum_range,
        )
    else:
        result = pd.DataFrame(prices, columns=SECTOR_IDS)
        result.insert(0, "date", output_dates)
    if raw_returns_path is not None:
        raw_result = pd.DataFrame(np.asarray(raw_sampled_returns), columns=SECTOR_IDS)
        raw_result.insert(0, "date", output_dates)
        raw_result.to_csv(ensure_parent(raw_returns_path), index=False)
    return result


def generate_from_config(
    config: dict[str, Any],
    *,
    days: int,
    seed: int,
    initial_price: float | None = None,
    event_schedule: Any = None,
    volatility_scale: float | None = None,
    return_clipping_percentile: tuple[float, float] | None = None,
    hard_clip: bool | None = None,
    soft_clip: float | None = None,
    generate_ohlc: bool | None = None,
) -> pd.DataFrame:
    require_torch()
    import torch

    seed_everything(seed)
    requested_device = str(config.get("training", {}).get("device", "auto"))
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    model, checkpoint = _load_checkpoint(config_path(config, "checkpoint"), device)
    scaler = FeatureScaler.load(config_path(config, "scaler"))
    history = read_parquet(config_path(config, "sector_data"))
    generation_config = config.get("generation", {})
    resolved_hard_clip = bool(generation_config.get("hard_clip", False)) if hard_clip is None else hard_clip
    resolved_soft_clip = generation_config.get("return_soft_clip", 0.08) if soft_clip is None else soft_clip
    resolved_generate_ohlc = bool(generation_config.get("generate_ohlc", True)) if generate_ohlc is None else generate_ohlc
    raw_returns_path = None
    if "raw_generated_returns" in config.get("paths", {}):
        raw_returns_path = config_path(config, "raw_generated_returns")
    result = generate_price_paths(
        model,
        history,
        scaler,
        checkpoint,
        days=days,
        seed=seed,
        initial_price=initial_price if initial_price is not None else generation_config.get("initial_price", 100.0),
        event_schedule=event_schedule,
        volatility_scale=volatility_scale if volatility_scale is not None else generation_config.get("volatility_scale", 1.0),
        return_clipping_percentile=return_clipping_percentile,
        training_frame=history,
        hard_clip=resolved_hard_clip,
        soft_clip=resolved_soft_clip,
        raw_returns_path=raw_returns_path,
        feature_z_clip=generation_config.get("feature_z_clip", 6.0),
        volatility_persistence=float(generation_config.get("volatility_persistence", 0.9)),
        volatility_shock_scale=float(generation_config.get("volatility_shock_scale", 0.18)),
        generate_ohlc=resolved_generate_ohlc,
        ohlc_gap_ratio=float(generation_config.get("ohlc_gap_ratio", 0.35)),
        ohlc_range_scale=float(generation_config.get("ohlc_range_scale", 0.8)),
        ohlc_minimum_range=float(generation_config.get("ohlc_minimum_range", 0.001)),
    )
    write_parquet(result, config_path(config, "generated_prices"))
    if resolved_generate_ohlc and "generated_ohlc" in config.get("paths", {}):
        write_parquet(extract_ohlc_frame(result), config_path(config, "generated_ohlc"))
    if resolved_generate_ohlc and "candlestick_test_image" in config.get("paths", {}):
        try:
            save_candlestick_chart(
                result,
                config_path(config, "candlestick_test_image"),
                sector_id=str(generation_config.get("candlestick_sector", "energy")),
            )
        except RuntimeError as exc:
            LOGGER.warning("Candlestick image was not created: %s", exc)
    return result


def _read_events(path: str | None) -> Any:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        if path.endswith((".yaml", ".yml")):
            import yaml

            return yaml.safe_load(handle)
        return json.load(handle)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-price", type=float, default=None)
    parser.add_argument("--events", default=None, help="JSON/YAML event schedule")
    parser.add_argument("--volatility-scale", type=float, default=None)
    parser.add_argument("--clip", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--hard-clip", action="store_true", help="Enable percentile clipping of sampled returns")
    parser.add_argument("--soft-clip", type=float, default=None, help="Continuous return soft-clip scale")
    parser.add_argument("--no-ohlc", action="store_true", help="Only write the legacy close columns")
    args = parser.parse_args(argv)
    if args.soft_clip is not None and args.soft_clip <= 0:
        parser.error("--soft-clip must be positive")
    setup_logging()
    config = load_config(args.config)
    generate_from_config(
        config,
        days=args.days,
        seed=args.seed,
        initial_price=args.initial_price,
        event_schedule=_read_events(args.events),
        volatility_scale=args.volatility_scale,
        return_clipping_percentile=tuple(args.clip) if args.clip else None,
        hard_clip=args.hard_clip or bool(args.clip),
        soft_clip=args.soft_clip,
        generate_ohlc=not args.no_ohlc,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
