"""Recover JSON scalers from a weights-only checkpoint without reading legacy pickle files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import config_path, load_config
from .generate import _load_checkpoint
from .preprocessing import FeatureScaler


def migrate_scalers(config: dict[str, Any]) -> list[Path]:
    _, checkpoint = _load_checkpoint(config_path(config, "checkpoint"), "cpu")
    outputs: list[tuple[Path, FeatureScaler]] = []
    for key, metadata_key, columns_key in (
        ("scaler", "scaler_metadata", "ohlcv_columns"),
        ("relative_scaler", "relative_scaler_metadata", "relative_columns"),
    ):
        path = config_path(config, key)
        if path.suffix.lower() != ".json":
            raise ValueError(f"paths.{key} must point to a .json file before migration")
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing scaler: {path}")
        scaler = FeatureScaler.from_metadata(checkpoint.get(metadata_key))
        if scaler.columns != checkpoint.get(columns_key):
            raise ValueError(f"{metadata_key} columns do not match the checkpoint")
        outputs.append((path, scaler))
    for path, scaler in outputs:
        scaler.save(path)
    return [path for path, _ in outputs]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)
    for path in migrate_scalers(load_config(args.config)):
        print(f"Saved JSON scaler: {path}")


if __name__ == "__main__":
    main()
