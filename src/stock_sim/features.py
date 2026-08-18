"""Create individual-stock and sector-level leakage-safe features."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import config_path, ensure_parent, load_config, setup_logging
from .constants import DEFAULT_EVENT_TYPES, SECTOR_DEFINITIONS, STOCK_FEATURES
from .io import read_parquet, write_parquet
from .preprocessing import assert_finite

LOGGER = logging.getLogger(__name__)


def _safe_log(series: pd.Series) -> pd.Series:
    return np.log(pd.to_numeric(series, errors="coerce").where(lambda value: value > 0))


def _market_return(index_frame: pd.DataFrame, ticker: str) -> pd.Series:
    subset = index_frame[index_frame["ticker"].eq(ticker)].sort_values("date")
    if subset.empty:
        return pd.Series(dtype=float, name=ticker)
    close = subset["adj_close"].where(subset["adj_close"] > 0, subset["close"])
    result = _safe_log(close).diff()
    result.index = pd.to_datetime(subset["date"]).dt.normalize()
    return result.rename(ticker)


def _one_stock_features(
    price_frame: pd.DataFrame,
    *,
    ticker: str,
    topix_return: pd.Series,
    n225_return: pd.Series,
) -> pd.DataFrame:
    subset = price_frame[price_frame["ticker"].eq(ticker)].sort_values("date").copy()
    if subset.empty:
        return pd.DataFrame(columns=["date", "ticker", *STOCK_FEATURES])
    subset["date"] = pd.to_datetime(subset["date"]).dt.normalize()
    close = pd.to_numeric(subset["adj_close"], errors="coerce")
    close = close.where(close > 0, pd.to_numeric(subset["close"], errors="coerce"))
    open_price = pd.to_numeric(subset["open"], errors="coerce")
    high = pd.to_numeric(subset["high"], errors="coerce")
    low = pd.to_numeric(subset["low"], errors="coerce")
    volume = pd.to_numeric(subset["volume"], errors="coerce")
    ret = _safe_log(close).diff()
    result = pd.DataFrame({
        "date": subset["date"].to_numpy(),
        "ticker": ticker,
        "return_1d": ret.to_numpy(),
        "return_5d": _safe_log(close).diff(5).to_numpy(),
        "oc_change": (_safe_log(close) - _safe_log(open_price)).to_numpy(),
        "hl_range": (_safe_log(high) - _safe_log(low)).to_numpy(),
        "volume_log_change": _safe_log(volume).diff().to_numpy(),
        "vol_5d": ret.rolling(5, min_periods=5).std(ddof=0).to_numpy(),
        "vol_20d": ret.rolling(20, min_periods=20).std(ddof=0).to_numpy(),
    })
    result["excess_topix"] = result["return_1d"] - result["date"].map(topix_return)
    result["excess_n225"] = result["return_1d"] - result["date"].map(n225_return)
    return result


def _weighted_mean(group: pd.DataFrame, column: str, weights: pd.Series | None) -> float:
    values = pd.to_numeric(group[column], errors="coerce")
    valid = values.notna()
    if not valid.any():
        return np.nan
    if weights is None:
        return float(values[valid].mean())
    selected_weights = pd.to_numeric(weights.loc[group.index[valid]], errors="coerce")
    selected_weights = selected_weights.where(selected_weights > 0)
    usable_indices = selected_weights.index[selected_weights.notna()]
    if len(usable_indices) == 0 or selected_weights.loc[usable_indices].sum() <= 0:
        return float(values[valid].mean())
    return float(np.average(values.loc[usable_indices], weights=selected_weights.loc[usable_indices]))


def _aggregate_sector_features(
    stock_features: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    method: str,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    if method not in {"equal_weighted", "market_cap_weighted"}:
        raise ValueError("sector aggregation method must be equal_weighted or market_cap_weighted")
    cap_by_ticker = selected.set_index(selected["ticker"].astype(str))["market_cap_jpy"] if "market_cap_jpy" in selected else pd.Series(dtype=float)
    stock = stock_features.copy()
    stock["sector_id"] = stock["ticker"].map(selected.set_index("ticker")["sector_id"])
    rows: list[dict[str, Any]] = []
    sector_feature_columns: list[str] = []
    return_columns: list[str] = []
    volatility_columns: list[str] = []
    for sector_id, _ in SECTOR_DEFINITIONS:
        for feature in STOCK_FEATURES:
            column = f"{sector_id}__{feature}"
            sector_feature_columns.append(column)
            if feature == "return_1d":
                return_columns.append(column)
            # The model has one log-sigma per sector. Use the short-horizon
            # realized volatility as its supervised target while retaining
            # both vol_5d and vol_20d as input features.
            if feature == "vol_5d":
                volatility_columns.append(column)
    for date, day in stock.groupby("date", sort=True):
        item: dict[str, Any] = {"date": pd.Timestamp(date).normalize()}
        for sector_id, _ in SECTOR_DEFINITIONS:
            sector_day = day[day["sector_id"].eq(sector_id)]
            weights = None
            if method == "market_cap_weighted" and not sector_day.empty:
                weights = sector_day["ticker"].map(cap_by_ticker)
            for feature in STOCK_FEATURES:
                column = f"{sector_id}__{feature}"
                item[column] = _weighted_mean(sector_day, feature, weights)
        rows.append(item)
    sector = pd.DataFrame(rows)
    if sector.empty:
        return sector, sector_feature_columns, return_columns, volatility_columns
    sector = sector.sort_values("date").reset_index(drop=True)
    return sector, sector_feature_columns, return_columns, volatility_columns


def _add_market_features(
    sector: pd.DataFrame,
    indices: pd.DataFrame,
    return_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    result = sector.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    n225 = _market_return(indices, "^N225").rename("market__n225_return_1d")
    topix = _market_return(indices, "^TPX").rename("market__topix_return_1d")
    market = pd.DataFrame({"date": result["date"]}).set_index("date")
    market = market.join(n225).join(topix)
    market["market__return_spread"] = market["market__n225_return_1d"] - market["market__topix_return_1d"]
    market["market__n225_volatility"] = market["market__n225_return_1d"].rolling(20, min_periods=20).std(ddof=0)
    market["market__topix_volatility"] = market["market__topix_return_1d"].rolling(20, min_periods=20).std(ddof=0)
    sector_returns = result.set_index("date")[return_columns]
    market["market__up_sector_count"] = (sector_returns > 0).sum(axis=1)
    market["market__down_sector_count"] = (sector_returns < 0).sum(axis=1)
    market["market__sector_cross_std"] = sector_returns.std(axis=1, ddof=0)
    market = market.reset_index()
    result = result.merge(market, on="date", how="left", validate="one_to_one")
    market_columns = [column for column in market.columns if column != "date"]
    return result, market_columns


def build_feature_frame(
    prices: pd.DataFrame,
    indices: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    aggregation_method: str = "equal_weighted",
    event_types: list[str] | tuple[str, ...] = DEFAULT_EVENT_TYPES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return ``(sector_data, stock_features, schema)`` from long OHLCV data."""

    required_price_columns = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required_price_columns - set(prices.columns)
    if missing:
        raise ValueError(f"Price data missing columns: {sorted(missing)}")
    if "adj_close" not in prices:
        prices = prices.copy()
        prices["adj_close"] = prices["close"]
    selected = selected.copy()
    selected["ticker"] = selected["ticker"].astype(str)
    tickers = selected["ticker"].tolist()
    topix_return = _market_return(indices, "^TPX")
    n225_return = _market_return(indices, "^N225")
    stock_parts = [
        _one_stock_features(prices, ticker=ticker, topix_return=topix_return, n225_return=n225_return)
        for ticker in tickers
    ]
    stock_features = pd.concat(stock_parts, ignore_index=True) if stock_parts else pd.DataFrame()
    if stock_features.empty:
        raise ValueError("No stock features could be built")
    stock_features = stock_features.merge(selected[["ticker", "sector_id"]], on="ticker", how="left", validate="many_to_one")
    sector, sector_columns, return_columns, volatility_columns = _aggregate_sector_features(
        stock_features,
        selected,
        method=aggregation_method,
    )
    sector, market_columns = _add_market_features(sector, indices, return_columns)
    event_columns: list[str] = []
    # Keep a global type signal and a sector-specific mask/intensity signal.
    # The latter makes affected_sectors part of the model input rather than a
    # piece of metadata that is lost before inference.
    event_columns: list[str] = []
    event_frame_columns: list[str] = []
    for event_type in event_types:
        column = f"event__{event_type}"
        event_columns.append(column)
        event_frame_columns.append(column)
        for sector_id, _ in SECTOR_DEFINITIONS:
            column = f"event__{event_type}__{sector_id}"
            event_columns.append(column)
            event_frame_columns.append(column)
    event_frame = pd.DataFrame(0.0, index=sector.index, columns=event_frame_columns)
    sector = pd.concat([sector, event_frame], axis=1)
    feature_columns = sector_columns + market_columns + event_columns
    # Use only observations for which every sector and market feature is known.
    # This removes initial rolling windows and incomplete trading days instead
    # of forward filling information across a missing observation.
    sector[feature_columns] = sector[feature_columns].replace([np.inf, -np.inf], np.nan)
    before = len(sector)
    sector = sector.dropna(subset=feature_columns).sort_values("date").reset_index(drop=True)
    LOGGER.info("Dropped %s incomplete feature rows", before - len(sector))
    stock_features = stock_features.replace([np.inf, -np.inf], np.nan).sort_values(["ticker", "date"])
    schema = {
        "feature_columns": feature_columns,
        "sector_feature_columns": sector_columns,
        "market_feature_columns": market_columns,
        "event_columns": event_columns,
        "return_columns": return_columns,
        "volatility_columns": volatility_columns,
        "sector_ids": [sector_id for sector_id, _ in SECTOR_DEFINITIONS],
        "sequence_length_default": 60,
    }
    if sector.empty:
        raise ValueError("No complete sector feature rows remain after NaN filtering")
    assert_finite(sector, feature_columns)
    return sector, stock_features, schema


def build_features(
    prices_path: str | Path,
    indices_path: str | Path,
    selected_path: str | Path,
    *,
    sector_data_path: str | Path,
    stock_features_path: str | Path,
    schema_path: str | Path,
    aggregation_method: str = "equal_weighted",
    event_types: list[str] | tuple[str, ...] = DEFAULT_EVENT_TYPES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prices = read_parquet(prices_path)
    indices = read_parquet(indices_path)
    selected = pd.read_csv(selected_path)
    sector, stock, schema = build_feature_frame(
        prices,
        indices,
        selected,
        aggregation_method=aggregation_method,
        event_types=event_types,
    )
    write_parquet(sector, sector_data_path)
    write_parquet(stock, stock_features_path)
    ensure_parent(schema_path)
    with Path(schema_path).open("w", encoding="utf-8") as handle:
        json.dump(schema, handle, ensure_ascii=False, indent=2)
    return sector, stock, schema


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--aggregation", choices=["equal_weighted", "market_cap_weighted"], default=None)
    args = parser.parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    feature_config = config.get("features", {})
    build_features(
        config_path(config, "raw_prices"),
        config_path(config, "raw_indices"),
        config_path(config, "selected_universe"),
        sector_data_path=config_path(config, "sector_data"),
        stock_features_path=config_path(config, "stock_features"),
        schema_path=config_path(config, "feature_schema"),
        aggregation_method=args.aggregation or config.get("sector_aggregation", {}).get("method", "equal_weighted"),
        event_types=feature_config.get("event_types", DEFAULT_EVENT_TYPES),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
