"""Train the shared LSTM to predict the next sector OHLCV bar."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import config_path, ensure_parent, load_config, seed_everything, setup_logging
from .constants import OHLCV_FIELDS, SECTOR_IDS
from .dataset import make_ohlcv_split_datasets
from .io import read_parquet
from .model import MarketLSTM, model_hyperparameters, ohlcv_loss, require_torch
from .ohlcv import bars_to_relative, relative_column_names
from .preprocessing import FeatureScaler, assert_finite

LOGGER = logging.getLogger(__name__)


def _load_schema(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _device(torch, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _loader(torch, dataset: Any, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _run_epoch(
    torch,
    model,
    loader: Any,
    device,
    optimizer=None,
) -> float:
    training = optimizer is not None
    model.train(training)
    values: list[float] = []
    for features, target_relative, teacher_features in loader:
        features = features.to(device)
        target_relative = target_relative.to(device)
        teacher_features = teacher_features.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        window = features
        predictions = []
        for step in range(target_relative.shape[1]):
            predictions.append(model(window))
            if step + 1 < target_relative.shape[1]:
                window = torch.cat(
                    [window[:, 1:, :], teacher_features[:, step : step + 1, :]],
                    dim=1,
                )
        prediction = torch.stack(predictions, dim=1)
        loss = ohlcv_loss(prediction, target_relative)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        values.append(float(loss.detach().cpu()))
    return float(np.mean(values)) if values else float("nan")


def _estimate_noise_scale(torch, model, loader: Any, device) -> np.ndarray:
    """Estimate standardized one-step residual noise by relative field."""

    model.eval()
    residuals: list[np.ndarray] = []
    with torch.no_grad():
        for features, target_relative, _teacher_features in loader:
            prediction = model(features.to(device)).detach().cpu().numpy()
            prediction = prediction.reshape(target_relative.shape[0], -1)
            target = target_relative[:, 0].numpy().reshape(target_relative.shape[0], -1)
            residuals.append((target - prediction).reshape(-1, len(OHLCV_FIELDS)))
    if not residuals:
        return np.ones(len(OHLCV_FIELDS), dtype=np.float64)
    scale = np.std(np.concatenate(residuals, axis=0), axis=0, ddof=0)
    return np.clip(np.nan_to_num(scale, nan=0.5, posinf=1.5, neginf=0.5), 0.15, 1.5)


def train_model(config: dict[str, Any]) -> dict[str, Any]:
    require_torch()
    import torch

    seed = int(config.get("training", {}).get("seed", 42))
    seed_everything(seed)
    frame = read_parquet(config_path(config, "sector_data")).sort_values("date").reset_index(drop=True)
    schema = _load_schema(config_path(config, "feature_schema"))
    ohlcv_columns = list(schema["ohlcv_columns"])
    if len(ohlcv_columns) != len(SECTOR_IDS) * len(OHLCV_FIELDS):
        raise ValueError("Expected one OHLCV bar for each of the 11 sectors")
    assert_finite(frame, ohlcv_columns)
    data_config = config.get("data", {})
    sequence_length = int(data_config.get("sequence_length", 60))
    horizon = int(data_config.get("horizon", 1))
    forecast_steps = int(data_config.get("forecast_steps", 5))
    volume_lookback = int(data_config.get("volume_lookback", 20))
    if volume_lookback < 1 or volume_lookback > sequence_length:
        raise ValueError("data.volume_lookback must be between one and sequence_length")
    splits = data_config.get("splits", {})
    train_end = str(splits.get("train_end", "2021-12-31"))
    validation_end = str(splits.get("validation_end", "2023-12-31"))
    train_rows = frame[pd.to_datetime(frame["date"]) <= pd.Timestamp(train_end)]
    if train_rows.empty:
        raise ValueError(f"No rows in training period ending {train_end}")
    scaler = FeatureScaler(ohlcv_columns).fit(train_rows[ohlcv_columns])
    scaler.save(config_path(config, "scaler"))
    relative_columns = relative_column_names()
    raw_bars = frame[ohlcv_columns].to_numpy(dtype=np.float64).reshape(
        len(frame), len(SECTOR_IDS), len(OHLCV_FIELDS)
    )
    raw_log_volumes = np.log(raw_bars[..., 4])
    volume_references = np.stack(
        [
            np.exp(raw_log_volumes[index - volume_lookback : index].mean(axis=0))
            for index in range(volume_lookback, len(frame))
        ],
        axis=0,
    )
    relative_values = bars_to_relative(
        raw_bars[volume_lookback:],
        raw_bars[volume_lookback - 1 : -1],
        volume_reference=volume_references,
    ).reshape(len(frame) - volume_lookback, -1)
    relative_dates = pd.to_datetime(frame["date"]).iloc[volume_lookback:].reset_index(drop=True)
    relative_train_mask = relative_dates <= pd.Timestamp(train_end)
    relative_scaler = FeatureScaler(relative_columns).fit(relative_values[relative_train_mask])
    relative_scaler.save(config_path(config, "relative_scaler"))
    LOGGER.info(
        "scaler units: feature_count=%d mean_range=[%.6f, %.6f] scale_range=[%.6f, %.6f]",
        len(scaler.columns),
        float(scaler.mean_.min()),
        float(scaler.mean_.max()),
        float(scaler.scale_.min()),
        float(scaler.scale_.max()),
    )
    datasets = make_ohlcv_split_datasets(
        frame,
        ohlcv_columns=ohlcv_columns,
        scaler=scaler,
        sector_count=len(SECTOR_IDS),
        field_count=len(OHLCV_FIELDS),
        sequence_length=sequence_length,
        horizon=horizon,
        target_scaler=relative_scaler,
        forecast_steps=forecast_steps,
        volume_lookback=volume_lookback,
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
        len(ohlcv_columns),
    )
    model = MarketLSTM(**hyperparameters)
    device = _device(torch, str(train_config.get("device", "auto")))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 0.001)),
        weight_decay=float(train_config.get("weight_decay", 1e-5)),
    )
    epochs = int(train_config.get("epochs", 50))
    patience = int(train_config.get("patience", 10))
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
        )
        validation_loss = (
            _run_epoch(
                torch,
                model,
                validation_loader,
                device,
                None,
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
    calibration_loader = validation_loader if validation_loader is not None else train_loader
    relative_noise_scale = _estimate_noise_scale(torch, model, calibration_loader, device)

    checkpoint = {
        "model_type": "lstm_relative_ohlcv",
        "input_type": "ohlcv",
        "target_type": "relative_ohlcv",
        "state_dict": model.state_dict(),
        "model_kwargs": hyperparameters,
        "ohlcv_columns": ohlcv_columns,
        "relative_columns": relative_columns,
        "sequence_length": sequence_length,
        "horizon": horizon,
        "forecast_steps": forecast_steps,
        "volume_lookback": volume_lookback,
        "train_end": train_end,
        "validation_end": validation_end,
        "generation_seed": seed,
        "generation_config": config.get("generation", {}),
        "scaler_metadata": {"columns": scaler.columns, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "relative_scaler_metadata": {
            "columns": relative_scaler.columns,
            "mean": relative_scaler.mean_.tolist(),
            "scale": relative_scaler.scale_.tolist(),
        },
        "relative_noise_scale": relative_noise_scale.tolist(),
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
