"""Compare generated and historical sector distributions and charts."""

from __future__ import annotations

import argparse
import html
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import config_path, ensure_parent, load_config, setup_logging
from .constants import SECTOR_IDS
from .generate import _load_checkpoint, generate_price_paths
from .io import read_parquet
from .preprocessing import FeatureScaler

LOGGER = logging.getLogger(__name__)


def _autocorrelation(values: np.ndarray) -> float:
    if len(values) < 3 or np.std(values[:-1]) < 1e-12 or np.std(values[1:]) < 1e-12:
        return 0.0
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _absolute_autocorrelation(values: np.ndarray) -> float:
    """Autocorrelation of absolute returns, a simple volatility-clustering check."""

    return _autocorrelation(np.abs(values))


def _max_drawdown(returns: np.ndarray) -> float:
    wealth = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    return float(drawdown.min())


def _rotation_frequency(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    signs = np.sign(returns)
    changed = ((signs[1:] != signs[:-1]) & (signs[1:] != 0) & (signs[:-1] != 0)).any(axis=1)
    return float(changed.mean())


def _market_distribution(returns: np.ndarray) -> dict[str, float]:
    up_count = (returns > 0).sum(axis=1)
    down_count = (returns < 0).sum(axis=1)
    return {
        "rotation_frequency": _rotation_frequency(returns),
        "full_up_frequency": float((up_count == returns.shape[1]).mean()) if len(returns) else 0.0,
        "full_down_frequency": float((down_count == returns.shape[1]).mean()) if len(returns) else 0.0,
        "mean_up_sector_count": float(up_count.mean()) if len(returns) else 0.0,
        "mean_down_sector_count": float(down_count.mean()) if len(returns) else 0.0,
        "up_sector_count_q05": float(np.quantile(up_count, 0.05)) if len(returns) else 0.0,
        "up_sector_count_q50": float(np.quantile(up_count, 0.50)) if len(returns) else 0.0,
        "up_sector_count_q95": float(np.quantile(up_count, 0.95)) if len(returns) else 0.0,
        "down_sector_count_q05": float(np.quantile(down_count, 0.05)) if len(returns) else 0.0,
        "down_sector_count_q50": float(np.quantile(down_count, 0.50)) if len(returns) else 0.0,
        "down_sector_count_q95": float(np.quantile(down_count, 0.95)) if len(returns) else 0.0,
    }


def calculate_metrics(returns: np.ndarray, sector_ids: list[str] | tuple[str, ...] = SECTOR_IDS) -> pd.DataFrame:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(sector_ids):
        raise ValueError(f"returns must have shape [days, {len(sector_ids)}]")
    if not np.isfinite(values).all():
        raise ValueError("returns contain NaN or infinite values")
    rows: list[dict[str, Any]] = []
    for index, sector_id in enumerate(sector_ids):
        series = values[:, index]
        rows.append(
            {
                "scope": "sector",
                "sector_id": sector_id,
                "mean_return": float(series.mean()),
                "std_return": float(series.std(ddof=0)),
                "up_day_ratio": float((series > 0).mean()),
                "down_day_ratio": float((series < 0).mean()),
                "autocorrelation_1": _autocorrelation(series),
                "absolute_autocorrelation_1": _absolute_autocorrelation(series),
                "max_up_return": float(series.max()),
                "max_down_return": float(series.min()),
                "max_drawdown": _max_drawdown(series),
                "return_q01": float(np.quantile(series, 0.01)),
                "return_q05": float(np.quantile(series, 0.05)),
                "return_q50": float(np.quantile(series, 0.50)),
                "return_q95": float(np.quantile(series, 0.95)),
                "return_q99": float(np.quantile(series, 0.99)),
                "mean_rolling_volatility_20d": float(pd.Series(series).rolling(20).std(ddof=0).mean()),
            }
        )
    rows.append({"scope": "market", "sector_id": "__market__", **_market_distribution(values)})
    return pd.DataFrame(rows)


def _returns_from_generated(
    prices: pd.DataFrame,
    *,
    initial_price: float | np.ndarray = 100.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    missing = set(SECTOR_IDS) - set(prices.columns)
    if missing:
        raise ValueError(f"Generated prices missing sectors: {sorted(missing)}")
    values = prices[list(SECTOR_IDS)].to_numpy(dtype=float)
    if (values <= 0).any() or not np.isfinite(values).all():
        raise ValueError("Generated prices must be finite and positive")
    initial = np.asarray(initial_price, dtype=float)
    if initial.ndim == 0:
        initial = np.full(values.shape[1], float(initial))
    if initial.shape != (values.shape[1],) or (initial <= 0).any():
        raise ValueError("initial_price must be a positive scalar or one value per sector")
    first = np.log(values[0] / initial)
    following = np.diff(np.log(values), axis=0)
    return prices, np.vstack([first, following])


def _correlation_matrix(returns: np.ndarray) -> np.ndarray:
    matrix = np.corrcoef(np.asarray(returns, dtype=float).T)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def _off_diagonal_mean(matrix: np.ndarray) -> float:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(matrix[mask].mean())


def _real_returns(sector_data: pd.DataFrame, schema_path: str | Path | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    if schema_path is not None and Path(schema_path).exists():
        import json

        with Path(schema_path).open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        return_columns = schema["return_columns"]
    else:
        return_columns = [f"{sector_id}__return_1d" for sector_id in SECTOR_IDS]
    values = sector_data[return_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Real sector returns contain NaN or infinite values")
    return sector_data, values


def _save_figures(
    real_returns: np.ndarray,
    generated_returns: np.ndarray,
    output_dir: str | Path,
) -> list[Path]:
    """Save only close-path comparison visuals for the OHLCV model.

    ``real_returns`` and ``generated_returns`` remain the evaluation API used
    by the numeric metrics, but volatility, return-distribution, and
    correlation plots belong to the removed return-distribution model and are
    intentionally no longer rendered here.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional report dependency
        LOGGER.warning("matplotlib is not installed; skipping PNG figures")
        return []
    paths: list[Path] = []
    real_prices = 100.0 * np.exp(np.cumsum(real_returns, axis=0))
    generated_prices = 100.0 * np.exp(np.cumsum(generated_returns, axis=0))
    figure, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    axes[0].plot(real_prices, alpha=0.45)
    axes[0].set_title("Historical close paths (normalized to 100)")
    axes[0].set_ylabel("Normalized close")
    axes[1].plot(generated_prices, alpha=0.75)
    axes[1].set_title("Generated close paths (normalized to 100)")
    axes[1].set_ylabel("Normalized close")
    axes[1].set_xlabel("Trading day")
    path = output / "real_vs_generated.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths.append(path)
    return paths


def evaluate_frames(
    real_returns: np.ndarray,
    generated_returns: np.ndarray,
    *,
    report_path: str | Path,
    figures_dir: str | Path,
    metrics_path: str | Path | None = None,
) -> pd.DataFrame:
    real = calculate_metrics(real_returns)
    generated = calculate_metrics(generated_returns)
    real.insert(1, "dataset", "real")
    generated.insert(1, "dataset", "generated")
    metrics = pd.concat([real, generated], ignore_index=True)
    if metrics_path is None:
        metrics_path = Path(report_path).parent / "evaluation_metrics.csv"
    ensure_parent(metrics_path)
    metrics.to_csv(metrics_path, index=False)
    real_correlation = _correlation_matrix(real_returns)
    generated_correlation = _correlation_matrix(generated_returns)
    correlations = pd.DataFrame(real_correlation, index=SECTOR_IDS, columns=SECTOR_IDS)
    correlations.to_csv(Path(metrics_path).parent / "real_sector_correlation.csv")
    pd.DataFrame(generated_correlation, index=SECTOR_IDS, columns=SECTOR_IDS).to_csv(
        Path(metrics_path).parent / "generated_sector_correlation.csv"
    )
    correlation_mae = float(
        np.abs(real_correlation - generated_correlation)[~np.eye(len(SECTOR_IDS), dtype=bool)].mean()
    )
    correlation_metrics = pd.DataFrame(
        [
            {
                "scope": "correlation",
                "dataset": "real",
                "sector_id": "__offdiag__",
                "correlation_mean": _off_diagonal_mean(real_correlation),
                "correlation_mae": 0.0,
            },
            {
                "scope": "correlation",
                "dataset": "generated",
                "sector_id": "__offdiag__",
                "correlation_mean": _off_diagonal_mean(generated_correlation),
                "correlation_mae": correlation_mae,
            },
        ]
    )
    correlation_metrics.to_csv(Path(metrics_path).parent / "correlation_metrics.csv", index=False)
    metrics = pd.concat([metrics, correlation_metrics], ignore_index=True)
    metrics.to_csv(metrics_path, index=False)
    figures = _save_figures(real_returns, generated_returns, figures_dir)
    report = Path(report_path)
    ensure_parent(report)
    table = metrics.to_html(index=False, float_format=lambda value: f"{value:.8f}")
    figure_html = "".join(f"<p><img src='{html.escape(path.parent.name + '/' + path.name)}' style='max-width:100%'></p>" for path in figures)
    report.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Stock simulation evaluation</title>"
        "<style>body{font-family:sans-serif}table{border-collapse:collapse}td,th{padding:4px;border:1px solid #ccc}</style>"
        f"</head><body><h1>Real vs generated sector market</h1>{table}{figure_html}</body></html>",
        encoding="utf-8",
    )
    return metrics


def _metrics_for_runs(runs: list[np.ndarray], dataset: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for scenario_id, values in enumerate(runs):
        frame = calculate_metrics(values)
        frame.insert(1, "dataset", dataset)
        frame.insert(2, "scenario_id", scenario_id)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _aggregate_sector_metrics(run_metrics: pd.DataFrame) -> pd.DataFrame:
    sector = run_metrics[run_metrics["scope"] == "sector"].copy()
    identifiers = ["scope", "dataset", "sector_id"]
    numeric = [
        "mean_return",
        "std_return",
        "up_day_ratio",
        "down_day_ratio",
        "autocorrelation_1",
        "absolute_autocorrelation_1",
        "max_up_return",
        "max_down_return",
        "max_drawdown",
        "return_q01",
        "return_q05",
        "return_q50",
        "return_q95",
        "return_q99",
        "mean_rolling_volatility_20d",
    ]
    summary = sector.groupby(identifiers, as_index=False)[numeric].mean()
    spread = sector.groupby(identifiers, as_index=False)[numeric].std(ddof=0)
    spread = spread.rename(columns={column: f"{column}_scenario_std" for column in numeric})
    summary = summary.merge(spread, on=identifiers, how="left")
    counts = sector.groupby(identifiers, as_index=False)["scenario_id"].nunique().rename(
        columns={"scenario_id": "scenario_count"}
    )
    return summary.merge(counts, on=identifiers, how="left")


def _run_distribution_rows(
    runs: list[np.ndarray],
    dataset: str,
    reference_correlation: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    mask = ~np.eye(len(SECTOR_IDS), dtype=bool)
    for scenario_id, values in enumerate(runs):
        metrics = calculate_metrics(values)
        sector = metrics[metrics["scope"] == "sector"]
        correlation = _correlation_matrix(values)
        terminal = 100.0 * np.exp(np.cumsum(values, axis=0))[-1]
        rows.append(
            {
                "dataset": dataset,
                "scenario_id": scenario_id,
                "mean_return": float(values.mean()),
                "std_return": float(values.std(ddof=0)),
                "up_day_ratio": float((values > 0).mean()),
                "autocorrelation_1": float(sector["autocorrelation_1"].mean()),
                "absolute_autocorrelation_1": float(sector["absolute_autocorrelation_1"].mean()),
                "mean_rolling_volatility_20d": float(sector["mean_rolling_volatility_20d"].mean()),
                "mean_max_drawdown": float(sector["max_drawdown"].mean()),
                "correlation_mean": _off_diagonal_mean(correlation),
                "correlation_mae": float(np.abs(correlation - reference_correlation)[mask].mean()),
                "terminal_price_mean": float(terminal.mean()),
                "terminal_price_median": float(np.median(terminal)),
                "terminal_price_min": float(terminal.min()),
                "terminal_price_max": float(terminal.max()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_scenarios(
    config: dict[str, Any],
    *,
    horizon: int | None = None,
    real_window_count: int | None = None,
    generated_seed_count: int | None = None,
) -> pd.DataFrame:
    """Evaluate matched historical windows against many generated seeds.

    Historical prices are never compared at their absolute level.  Every
    window starts at 100, and all aggregate metrics are computed from log
    returns or normalized terminal prices.
    """

    evaluation_config = config.get("evaluation", {})
    horizon = int(horizon or evaluation_config.get("horizon", 120))
    real_window_count = int(real_window_count or evaluation_config.get("real_windows", 100))
    generated_seed_count = int(generated_seed_count or evaluation_config.get("seeds", 100))
    if horizon < 2 or real_window_count < 1 or generated_seed_count < 1:
        raise ValueError("horizon must be >= 2 and scenario counts must be positive")

    real_frame = read_parquet(config_path(config, "sector_data")).sort_values("date").reset_index(drop=True)
    _, real_returns = _real_returns(real_frame, config_path(config, "feature_schema"))
    import torch

    requested_device = str(config.get("training", {}).get("device", "auto"))
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    model, checkpoint = _load_checkpoint(config_path(config, "checkpoint"), device)
    scaler = FeatureScaler.load(config_path(config, "scaler"))
    sequence_length = int(checkpoint.get("sequence_length", config.get("data", {}).get("sequence_length", 60)))
    valid_starts = np.arange(sequence_length, len(real_returns) - horizon + 1)
    if len(valid_starts) == 0:
        raise ValueError("Not enough historical rows for the requested evaluation windows")
    base_seed = int(evaluation_config.get("seed", config.get("training", {}).get("seed", 42)))
    rng = np.random.default_rng(base_seed)
    starts = rng.choice(valid_starts, size=max(real_window_count, generated_seed_count), replace=len(valid_starts) < max(real_window_count, generated_seed_count))
    real_runs = [real_returns[int(start) : int(start) + horizon] for start in starts[:real_window_count]]

    generation_config = config.get("generation", {})
    report_path = config_path(config, "evaluation_report")
    output_dir = report_path.parent
    raw_path = output_dir / "raw_generated_returns.csv"
    generated_runs: list[np.ndarray] = []
    for scenario_id in range(generated_seed_count):
        start = int(starts[scenario_id % len(starts)])
        generated_prices = generate_price_paths(
            model,
            real_frame.iloc[:start],
            scaler,
            checkpoint,
            days=horizon,
            seed=base_seed + scenario_id,
            initial_price=100.0,
            volatility_scale=float(generation_config.get("volatility_scale", 1.0)),
            training_frame=real_frame,
            hard_clip=bool(generation_config.get("hard_clip", False)),
            soft_clip=generation_config.get("return_soft_clip", 0.08),
            raw_returns_path=raw_path if scenario_id == 0 else None,
            feature_z_clip=generation_config.get("feature_z_clip", 6.0),
            volatility_persistence=float(generation_config.get("volatility_persistence", 0.9)),
            volatility_shock_scale=float(generation_config.get("volatility_shock_scale", 0.18)),
        )
        previous_close_columns = [f"{sector_id}__close" for sector_id in SECTOR_IDS]
        previous_close = real_frame.iloc[start - 1][previous_close_columns].to_numpy(dtype=float)
        _, generated_returns = _returns_from_generated(generated_prices, initial_price=previous_close)
        generated_runs.append(generated_returns)

    real_run_metrics = _metrics_for_runs(real_runs, "real_window")
    generated_run_metrics = _metrics_for_runs(generated_runs, "generated_seed")
    run_metrics = pd.concat([real_run_metrics, generated_run_metrics], ignore_index=True)
    run_metrics.to_csv(output_dir / "window_metrics.csv", index=False)
    summary = pd.concat(
        [_aggregate_sector_metrics(real_run_metrics), _aggregate_sector_metrics(generated_run_metrics)],
        ignore_index=True,
    )

    real_correlation = np.mean([_correlation_matrix(values) for values in real_runs], axis=0)
    generated_correlation = np.mean([_correlation_matrix(values) for values in generated_runs], axis=0)
    pd.DataFrame(real_correlation, index=SECTOR_IDS, columns=SECTOR_IDS).to_csv(
        output_dir / "real_sector_correlation.csv"
    )
    pd.DataFrame(generated_correlation, index=SECTOR_IDS, columns=SECTOR_IDS).to_csv(
        output_dir / "generated_sector_correlation.csv"
    )
    mask = ~np.eye(len(SECTOR_IDS), dtype=bool)
    correlation_mae = float(np.abs(real_correlation - generated_correlation)[mask].mean())
    correlation_metrics = pd.DataFrame(
        [
            {
                "scope": "correlation",
                "dataset": "real_window",
                "sector_id": "__offdiag__",
                "correlation_mean": _off_diagonal_mean(real_correlation),
                "correlation_mae": 0.0,
                "scenario_count": len(real_runs),
            },
            {
                "scope": "correlation",
                "dataset": "generated_seed",
                "sector_id": "__offdiag__",
                "correlation_mean": _off_diagonal_mean(generated_correlation),
                "correlation_mae": correlation_mae,
                "scenario_count": len(generated_runs),
            },
        ]
    )
    correlation_metrics.to_csv(output_dir / "correlation_metrics.csv", index=False)
    summary = pd.concat([summary, correlation_metrics], ignore_index=True, sort=False)
    ensure_parent(output_dir / "evaluation_metrics.csv")
    summary.to_csv(output_dir / "evaluation_metrics.csv", index=False)

    reference_correlation = real_correlation
    seed_metrics = pd.concat(
        [
            _run_distribution_rows(real_runs, "real_window", reference_correlation),
            _run_distribution_rows(generated_runs, "generated_seed", reference_correlation),
        ],
        ignore_index=True,
    )
    seed_metrics.to_csv(output_dir / "seed_metrics.csv", index=False)

    terminal_rows: list[dict[str, Any]] = []
    for dataset, runs in (("real_window", real_runs), ("generated_seed", generated_runs)):
        for scenario_id, values in enumerate(runs):
            terminal = 100.0 * np.exp(np.cumsum(values, axis=0))[-1]
            terminal_rows.extend(
                {
                    "dataset": dataset,
                    "scenario_id": scenario_id,
                    "sector_id": sector_id,
                    "terminal_price": float(price),
                }
                for sector_id, price in zip(SECTOR_IDS, terminal, strict=True)
            )
    pd.DataFrame(terminal_rows).to_csv(output_dir / "terminal_price_distribution.csv", index=False)

    figures = _save_figures(
        real_runs[0],
        generated_runs[0],
        report_path.parent / "figures",
    )
    table = summary.to_html(index=False, float_format=lambda value: f"{value:.8f}")
    seed_table = seed_metrics.describe(include="all").to_html(float_format=lambda value: f"{value:.8f}")
    figure_html = "".join(
        f"<p><img src='{html.escape(path.parent.name + '/' + path.name)}' style='max-width:100%'></p>"
        for path in figures
    )
    ensure_parent(report_path)
    report_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Stock simulation evaluation</title>"
        "<style>body{font-family:sans-serif}table{border-collapse:collapse}td,th{padding:4px;border:1px solid #ccc}</style>"
        f"</head><body><h1>Windowed real vs generated sector market</h1>"
        f"<p>real windows={len(real_runs)}, generated seeds={len(generated_runs)}, horizon={horizon}</p>"
        f"{table}<h2>Scenario distribution</h2>{seed_table}{figure_html}</body></html>",
        encoding="utf-8",
    )
    return summary


def evaluate_from_config(config: dict[str, Any]) -> pd.DataFrame:
    return evaluate_scenarios(config)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--windows", type=int, default=None, help="Number of historical windows")
    parser.add_argument("--seeds", type=int, default=None, help="Number of generated seeds")
    args = parser.parse_args(argv)
    setup_logging()
    evaluate_scenarios(
        load_config(args.config),
        horizon=args.horizon,
        real_window_count=args.windows,
        generated_seed_count=args.seeds,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
