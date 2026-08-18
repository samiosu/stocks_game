"""Train the shared probabilistic GRU with chronological splits."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import config_path, ensure_parent, load_config, seed_everything, setup_logging
from .dataset import MarketWindowDataset, make_split_datasets
from .factors import estimate_factor_structure
from .io import read_parquet
from .model import MarketGRU, model_hyperparameters, probabilistic_loss, require_torch
from .preprocessing import FeatureScaler, assert_finite

LOGGER = logging.getLogger(__name__)


def _load_schema(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _device(torch, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _loader(torch, dataset: MarketWindowDataset, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _run_epoch(
    torch,
    model,
    loader,
    device,
    optimizer=None,
    volatility_loss_weight=0.1,
    mean_loss_weight=0.25,
) -> float:
    training = optimizer is not None
    model.train(training)
    values: list[float] = []
    for features, target_returns, target_log_vol in loader:
        features = features.to(device)
        target_returns = target_returns.to(device)
        target_log_vol = target_log_vol.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        params = model(features)
        loss = probabilistic_loss(
            params,
            target_returns,
            target_log_vol,
            volatility_loss_weight=volatility_loss_weight,
            mean_loss_weight=mean_loss_weight,
        )
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        values.append(float(loss.detach().cpu()))
    return float(np.mean(values)) if values else float("nan")


def train_model(config: dict[str, Any]) -> dict[str, Any]:
    require_torch()
    import torch

    seed = int(config.get("training", {}).get("seed", 42))
    seed_everything(seed)
    frame = read_parquet(config_path(config, "sector_data"))
    schema = _load_schema(config_path(config, "feature_schema"))
    feature_columns = list(schema["feature_columns"])
    return_columns = list(schema["return_columns"])
    volatility_columns = list(schema["volatility_columns"])
    assert_finite(frame, feature_columns + return_columns + volatility_columns)
    data_config = config.get("data", {})
    sequence_length = int(data_config.get("sequence_length", 60))
    horizon = int(data_config.get("horizon", 1))
    splits = data_config.get("splits", {})
    train_end = str(splits.get("train_end", "2021-12-31"))
    validation_end = str(splits.get("validation_end", "2023-12-31"))
    train_rows = frame[pd.to_datetime(frame["date"]) <= pd.Timestamp(train_end)]
    if train_rows.empty:
        raise ValueError(f"No rows in training period ending {train_end}")
    # Factor loadings describe the game-market co-movement distribution, not a
    # predictive signal.  Estimate them from the complete historical panel so
    # the generated correlation target uses all available sector observations.
    factor_metadata = estimate_factor_structure(frame[return_columns].to_numpy(dtype=float))
    scaler = FeatureScaler(feature_columns).fit(train_rows[feature_columns])
    scaler.save(config_path(config, "scaler"))
    LOGGER.info(
        "scaler units: feature_count=%d mean_range=[%.6f, %.6f] scale_range=[%.6f, %.6f]",
        len(scaler.columns),
        float(scaler.mean_.min()),
        float(scaler.mean_.max()),
        float(scaler.scale_.min()),
        float(scaler.scale_.max()),
    )
    train_return_values = train_rows[return_columns].to_numpy(dtype=float)
    LOGGER.info(
        "raw return units: mean=%.6f std=%.6f min=%.6f max=%.6f",
        float(train_return_values.mean()),
        float(train_return_values.std(ddof=0)),
        float(train_return_values.min()),
        float(train_return_values.max()),
    )
    datasets = make_split_datasets(
        frame,
        feature_columns=feature_columns,
        return_columns=return_columns,
        volatility_columns=volatility_columns,
        scaler=scaler,
        sequence_length=sequence_length,
        horizon=horizon,
        train_end=train_end,
        validation_end=validation_end,
    )
    if len(datasets["train"]) == 0:
        raise ValueError("Training split has no windows; provide more history or reduce sequence_length")
    train_config = config.get("training", {})
    batch_size = int(train_config.get("batch_size", 64))
    train_loader = _loader(torch, datasets["train"], batch_size, True, seed)
    validation_loader = _loader(torch, datasets["validation"], batch_size, False, seed) if len(datasets["validation"]) else None
    hyperparameters = model_hyperparameters(
        config,
        len(feature_columns),
        factor_loadings=factor_metadata["factor_loadings"],
        common_noise_weight=float(factor_metadata["common_noise_weight"]),
    )
    model = MarketGRU(**hyperparameters)
    device = _device(torch, str(train_config.get("device", "auto")))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 0.001)),
        weight_decay=float(train_config.get("weight_decay", 1e-5)),
    )
    epochs = int(train_config.get("epochs", 50))
    patience = int(train_config.get("patience", 10))
    volatility_loss_weight = float(train_config.get("volatility_loss_weight", 0.1))
    mean_loss_weight = float(train_config.get("mean_loss_weight", 0.25))
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    wait = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(
            torch,
            model,
            train_loader,
            device,
            optimizer,
            volatility_loss_weight,
            mean_loss_weight,
        )
        validation_loss = (
            _run_epoch(
                torch,
                model,
                validation_loader,
                device,
                None,
                volatility_loss_weight,
                mean_loss_weight,
            )
            if validation_loader is not None
            else train_loss
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        LOGGER.info("epoch=%s train_loss=%.6f validation_loss=%.6f", epoch, train_loss, validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                LOGGER.info("Early stopping at epoch %s", epoch)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    percentile = config.get("generation", {}).get("return_clipping_percentile", [0.5, 99.5])
    train_returns = train_rows[return_columns].to_numpy(dtype=float)
    return_bounds = np.percentile(train_returns, [float(percentile[0]), float(percentile[1])], axis=0).T.tolist()
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_kwargs": hyperparameters,
        "feature_columns": feature_columns,
        "return_columns": return_columns,
        "volatility_columns": volatility_columns,
        "sequence_length": sequence_length,
        "horizon": horizon,
        "return_bounds": return_bounds,
        "event_columns": list(schema.get("event_columns", [])),
        "train_end": train_end,
        "validation_end": validation_end,
        "generation_seed": seed,
        "generation_config": config.get("generation", {}),
        "factor_metadata": factor_metadata,
        "scaler_metadata": {"columns": scaler.columns, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
    }
    checkpoint_path = config_path(config, "checkpoint")
    ensure_parent(checkpoint_path)
    torch.save(checkpoint, checkpoint_path)
    history_path = checkpoint_path.parent / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    LOGGER.info("Saved checkpoint to %s", checkpoint_path)
    return {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "best_validation_loss": best_loss,
        "epochs": len(history),
        "window_counts": {name: len(dataset) for name, dataset in datasets.items()},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    train_model(config)


if __name__ == "__main__":  # pragma: no cover
    main()
