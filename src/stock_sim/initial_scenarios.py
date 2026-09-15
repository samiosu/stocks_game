"""Select distinct observed market regimes and export Unity initial-window JSONs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import config_path, ensure_parent, load_config, setup_logging
from .initial_window import build_initial_window, validate_input_metadata
from .io import read_parquet

LOGGER = logging.getLogger(__name__)
REGIMES = {"bull": "上昇", "bear": "下落", "sideways": "横ばい", "volatile": "荒れ相場"}


def measure_windows(
    frame: pd.DataFrame, metadata: dict, trading_dates: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Measure only intact consecutive windows; never drop bad rows and stitch gaps."""

    validate_input_metadata(metadata)
    length = metadata["sequenceLength"]
    if length < 3:
        raise ValueError("Regime selection requires at least three rows per window")
    columns = metadata["ohlcvColumns"]
    missing = [column for column in ["date", *columns] if column not in frame.columns]
    if missing or frame.columns.duplicated().any():
        raise ValueError(f"Invalid source columns; missing: {missing}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("Source dates must be present and unique")
    frame = frame.sort_values("date").reset_index(drop=True)
    if len(frame) < length:
        raise ValueError(f"At least {length} source rows are required")
    calendar = pd.DatetimeIndex(pd.to_datetime(trading_dates)).normalize().unique().sort_values()
    if calendar.empty or calendar.hasnans:
        raise ValueError("A nonempty trading calendar without missing dates is required")
    positions = calendar.get_indexer(frame["date"])
    if (positions < 0).any():
        raise ValueError("Source dates are absent from the reference trading calendar")
    with np.errstate(over="ignore", invalid="ignore"):
        bars = frame[columns].to_numpy(np.float32).reshape(len(frame), -1, 5)
    valid = (
        np.isfinite(bars).all(axis=(1, 2))
        & (bars > 0).all(axis=(1, 2))
        & (bars[..., 1] >= np.maximum(bars[..., 0], bars[..., 3])).all(axis=1)
        & (bars[..., 2] <= np.minimum(bars[..., 0], bars[..., 3])).all(axis=1)
    )
    quality = {
        "sourceRows": len(frame),
        "invalidOhlcvRows": int((~valid).sum()),
        "invalidOhlcvDates": frame.loc[~valid, "date"].dt.strftime("%Y-%m-%d").tolist(),
        "possibleWindows": max(0, len(frame) - length + 1),
        "excludedInvalidOhlcvWindows": 0,
        "excludedNonconsecutiveWindows": 0,
    }
    time = np.arange(length, dtype=float)
    time -= time.mean()
    rows = []
    for start in range(len(frame) - length + 1):
        stop = start + length
        if not valid[start:stop].all():
            quality["excludedInvalidOhlcvWindows"] += 1
            continue
        if not (np.diff(positions[start:stop]) == 1).all():
            quality["excludedNonconsecutiveWindows"] += 1
            continue
        log_close = np.log(bars[start:stop, :, 3].astype(np.float64))
        # Geometric mean of sector closes normalized to the first day, not a
        # price-level-weighted average and not an exchange index.
        market = (log_close - log_close[0]).mean(axis=1)
        returns = np.diff(market)
        centered = market - market.mean()
        denominator = float((time @ time) * (centered @ centered))
        r_squared = float((time @ centered) ** 2 / denominator) if denominator > 1e-20 else 0.0
        rows.append(
            {
                "startIndex": start,
                "startDate": frame["date"].iloc[start].strftime("%Y-%m-%d"),
                "endDate": frame["date"].iloc[stop - 1].strftime("%Y-%m-%d"),
                "totalReturn": float(np.expm1(market[-1])),
                "dailyVolatility": float(returns.std(ddof=1)),
                "priceRange": float(np.expm1(np.ptp(market))),
                "trendRSquared": float(np.clip(r_squared, 0, 1)),
                "risingSectorFraction": float((log_close[-1] > log_close[0]).mean()),
                "fallingSectorFraction": float((log_close[-1] < log_close[0]).mean()),
            }
        )
    quality["validWindows"] = len(rows)
    if not rows:
        raise ValueError("No valid consecutive OHLCV windows are available")
    return frame, pd.DataFrame(rows), quality


def select_scenarios(windows: pd.DataFrame, length: int, per_regime: int) -> tuple[list, dict]:
    """Rank historical conditions and greedily select non-overlapping periods."""

    if type(per_regime) is not int or per_regime < 1:
        raise ValueError("per_regime must be a positive integer")
    calm_limit = float(windows["dailyVolatility"].quantile(0.5))
    volatile_limit = float(windows["dailyVolatility"].quantile(0.9))
    rules = {
        "minimumTrendReturn": 0.05,
        "minimumTrendRSquared": 0.5,
        "minimumSectorAgreement": 0.7,
        "maximumSidewaysAbsoluteReturn": 0.02,
        "maximumSidewaysPriceRange": 0.08,
        "calmVolatilityQuantile": 0.5,
        "calmDailyVolatilityLimit": calm_limit,
        "volatileVolatilityQuantile": 0.9,
        "volatileDailyVolatilityMinimum": volatile_limit,
    }
    trend = (windows["trendRSquared"] >= rules["minimumTrendRSquared"]) & (
        windows["dailyVolatility"] < volatile_limit
    )
    masks = {
        "bull": trend
        & (windows["totalReturn"] >= rules["minimumTrendReturn"])
        & (windows["risingSectorFraction"] >= rules["minimumSectorAgreement"]),
        "bear": trend
        & (windows["totalReturn"] <= -rules["minimumTrendReturn"])
        & (windows["fallingSectorFraction"] >= rules["minimumSectorAgreement"]),
        "sideways": (windows["totalReturn"].abs() <= rules["maximumSidewaysAbsoluteReturn"])
        & (windows["priceRange"] <= rules["maximumSidewaysPriceRange"])
        & (windows["dailyVolatility"] <= calm_limit),
        "volatile": (windows["dailyVolatility"] >= volatile_limit)
        & (windows["dailyVolatility"] > 0),
    }
    scores = {
        "bull": windows["totalReturn"] * windows["trendRSquared"],
        "bear": -windows["totalReturn"] * windows["trendRSquared"],
        "sideways": -windows["priceRange"] - windows["totalReturn"].abs(),
        "volatile": windows["dailyVolatility"],
    }
    counts = {regime: int(mask.sum()) for regime, mask in masks.items()}
    # Prioritize scarcer regimes so common rising windows cannot crowd them out.
    priority = sorted(REGIMES, key=lambda regime: counts[regime])
    chosen = []
    for regime in priority:
        candidates = windows.loc[masks[regime]].assign(score=scores[regime])
        candidates = candidates.sort_values(["score", "startIndex"], ascending=[False, True])
        selected = 0
        for row in candidates.to_dict("records"):
            if any(abs(row["startIndex"] - other["startIndex"]) < length for other in chosen):
                continue
            selected += 1
            chosen.append(dict(row, regime=regime, id=f"{regime}_{selected:02d}"))
            if selected == per_regime:
                break
        if selected != per_regime:
            raise ValueError(
                f"Could only select {selected}/{per_regime} non-overlapping {regime} windows. "
                "Try a smaller --per-regime or provide more history; thresholds are not relaxed."
            )
    chosen.sort(key=lambda row: (list(REGIMES).index(row["regime"]), row["id"]))
    return chosen, {"criteria": rules, "candidateCounts": counts, "selectionPriority": priority}


def export_initial_scenarios(
    sector_data_path: str | Path,
    metadata_path: str | Path,
    calendar_path: str | Path,
    output_dir: str | Path,
    *,
    per_regime: int = 3,
) -> Path:
    """Write a catalogue and raw windows; leave the single latest-window JSON alone."""

    metadata_bytes = Path(metadata_path).read_bytes()
    metadata = json.loads(metadata_bytes)
    calendar = pd.read_parquet(calendar_path, columns=["date"])["date"]
    frame, windows, quality = measure_windows(read_parquet(sector_data_path), metadata, calendar)
    chosen, selection = select_scenarios(windows, metadata["sequenceLength"], per_regime)
    length = metadata["sequenceLength"]
    output_dir = Path(output_dir)
    serialized_windows = []
    entries = []
    for selected in chosen:
        start = selected["startIndex"]
        payload = build_initial_window(frame.iloc[start : start + length], metadata)
        filename = selected["id"] + ".json"
        serialized_windows.append((filename, json.dumps(payload, indent=2, allow_nan=False) + "\n"))
        entries.append(
            {
                "id": selected["id"],
                "regime": selected["regime"],
                "label": REGIMES[selected["regime"]],
                "file": filename,
                "startDate": selected["startDate"],
                "endDate": selected["endDate"],
                "metrics": {
                    key: selected[key]
                    for key in (
                        "totalReturn",
                        "dailyVolatility",
                        "priceRange",
                        "trendRSquared",
                        "risingSectorFraction",
                        "fallingSectorFraction",
                    )
                },
            }
        )
    catalog = {
        "schemaVersion": 1,
        "sequenceLength": length,
        "featureSize": metadata["featureSize"],
        "perRegime": per_regime,
        "windowCount": len(entries),
        "metadataFile": Path(metadata_path).name,
        "metadataSha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "sourceDataFile": Path(sector_data_path).name,
        "tradingCalendarFile": Path(calendar_path).name,
        "marketDefinition": f"Geometric mean of {len(metadata['sectorIds'])} sector closes normalized to each window's first day",
        "metricUnits": "Returns/ranges are fractions; daily volatility is sample std of log returns (not annualized)",
        "continuity": f"{length} consecutive observed trading dates in the reference price data; no filling or stitching",
        "overlapPolicy": "No shared dates between any selected windows, including different regimes",
        "purpose": "Initial game conditions only; not an evaluation split or a guarantee of future regime",
        "sourceDataQuality": quality,
        "selection": selection,
        "scenarios": entries,
    }
    # Complete selection and validation before creating any output files.
    serialized_catalog = json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    for filename, serialized in serialized_windows:
        ensure_parent(output_dir / filename).write_text(serialized, encoding="utf-8")
    catalog_path = ensure_parent(output_dir / "catalog.json")
    catalog_path.write_text(serialized_catalog, encoding="utf-8")
    for entry in entries:
        LOGGER.info(
            "%s: %s to %s, return=%+.2f%%, daily volatility=%.2f%%",
            entry["id"],
            entry["startDate"],
            entry["endDate"],
            100 * entry["metrics"]["totalReturn"],
            100 * entry["metrics"]["dailyVolatility"],
        )
    LOGGER.info("Exported %d initial windows and catalogue: %s", len(entries), catalog_path)
    return catalog_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--metadata", help="Metadata JSON accompanying the Unity ONNX model")
    parser.add_argument(
        "--calendar", help="Parquet with observed trading dates; defaults to raw_prices"
    )
    parser.add_argument("--output-dir", help="Defaults to initial_windows beside sector_data")
    parser.add_argument("--per-regime", type=int, default=3)
    args = parser.parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    sector_data = config_path(config, "sector_data")
    export_initial_scenarios(
        sector_data,
        args.metadata or config_path(config, "checkpoint").with_suffix(".metadata.json"),
        args.calendar or config_path(config, "raw_prices"),
        args.output_dir or sector_data.with_name("initial_windows"),
        per_regime=args.per_regime,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
