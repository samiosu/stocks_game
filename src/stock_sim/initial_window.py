"""Export the latest raw OHLCV window as a Unity-readable JSON file."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import config_path, ensure_parent, load_config, setup_logging
from .constants import OHLCV_FIELDS
from .io import read_parquet

LOGGER = logging.getLogger(__name__)


def validate_input_metadata(metadata: dict) -> None:
    """Check the raw OHLCV layout expected by the exported model."""

    sequence_length = metadata["sequenceLength"]
    feature_size = metadata["featureSize"]
    columns = metadata["ohlcvColumns"]
    sectors = metadata["sectorIds"]
    expected_columns = [f"{sector}__{field}" for sector in sectors for field in OHLCV_FIELDS]
    if (
        metadata["inputType"] != "OHLCV"
        or metadata["ohlcvFields"] != list(OHLCV_FIELDS)
        or not sectors
        or len(set(sectors)) != len(sectors)
        or columns != expected_columns
        or feature_size != len(columns)
        or type(sequence_length) is not int
        or sequence_length < 1
        or metadata["onnx"]["inputShape"] != [1, sequence_length, feature_size]
    ):
        raise ValueError("Inconsistent OHLCV input metadata")


def build_initial_window(frame: pd.DataFrame, metadata: dict) -> dict:
    """Build a validated payload from the latest rows of a supplied history slice."""

    validate_input_metadata(metadata)
    sequence_length = metadata["sequenceLength"]
    feature_size = metadata["featureSize"]
    columns = metadata["ohlcvColumns"]
    sectors = metadata["sectorIds"]
    missing = [column for column in ["date", *columns] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing initial window columns: {missing}")
    if frame.columns.duplicated().any():
        raise ValueError("Initial window source has duplicate columns")
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if dates.isna().any():
        raise ValueError("Initial window source contains missing dates")
    window = frame.assign(date=dates).sort_values("date").tail(sequence_length)
    if len(window) != sequence_length:
        raise ValueError(f"{sequence_length} rows are required, got {len(window)}")
    if window["date"].duplicated().any():
        raise ValueError("Initial window contains duplicate dates")

    values = window[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Initial window must contain finite positive float32 OHLCV values")
    bars = values.reshape(sequence_length, len(sectors), len(OHLCV_FIELDS))
    if (bars[..., 1] < np.maximum(bars[..., 0], bars[..., 3])).any() or (
        bars[..., 2] > np.minimum(bars[..., 0], bars[..., 3])
    ).any():
        raise ValueError("Initial window high/low must contain the candle body")

    return {
        "sequenceLength": sequence_length,
        "featureSize": feature_size,
        "ohlcvColumns": columns,
        "dates": window["date"].dt.strftime("%Y-%m-%d").tolist(),
        "values": values.reshape(-1).tolist(),
    }


def export_initial_window(
    sector_data_path: str | Path,
    metadata_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Save chronological, time-major float32 values without scaling or clipping."""

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    payload = build_initial_window(read_parquet(sector_data_path), metadata)
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    output = ensure_parent(output_path)
    output.write_text(serialized, encoding="utf-8")
    LOGGER.info(
        "Exported initial OHLCV window: %s (%s to %s, %d x %d = %d raw values)",
        output,
        payload["dates"][0],
        payload["dates"][-1],
        payload["sequenceLength"],
        payload["featureSize"],
        len(payload["values"]),
    )
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--metadata", help="Metadata JSON exported alongside the Unity ONNX model")
    parser.add_argument(
        "--output", help="Output JSON; defaults to initial_ohlcv_window.json beside sector_data"
    )
    args = parser.parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    sector_data = config_path(config, "sector_data")
    export_initial_window(
        sector_data,
        args.metadata or config_path(config, "checkpoint").with_suffix(".metadata.json"),
        args.output or sector_data.with_name("initial_ohlcv_window.json"),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
