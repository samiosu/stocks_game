"""Batch-download and cache OHLCV data from Yahoo Finance."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    config_path,
    ensure_parent,
    load_config,
    runtime_as_of,
    setup_logging,
)
from .io import read_parquet, write_parquet

LOGGER = logging.getLogger(__name__)
PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


def _yf_module():
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("yfinance is required for fetch_data") from exc
    return yf


def _normalise_download(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        levels = [list(level) for level in frame.columns.levels]
        # group_by='ticker' produces ticker -> OHLCV. Some yfinance versions
        # return the reverse order, so inspect the level containing a ticker.
        ticker_level = 0 if any(t in levels[0] for t in tickers) else 1
        parts: list[pd.DataFrame] = []
        for ticker in tickers:
            if ticker not in levels[ticker_level]:
                continue
            sub = frame.xs(ticker, axis=1, level=ticker_level, drop_level=True).copy()
            sub.columns = [str(column).lower().replace(" ", "_") for column in sub.columns]
            sub["ticker"] = ticker
            parts.append(sub.reset_index(names="date"))
        if not parts:
            return pd.DataFrame(columns=PRICE_COLUMNS)
        frame = pd.concat(parts, ignore_index=True)
    else:
        ticker = tickers[0] if tickers else ""
        frame = frame.copy()
        frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
        frame["ticker"] = ticker
        frame = frame.reset_index(names="date")
    rename = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj_close": "adj_close",
        "volume": "volume",
    }
    frame = frame.rename(columns=rename)
    if "adj_close" not in frame:
        frame["adj_close"] = frame.get("close", np.nan)
    for column in PRICE_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_localize(None).dt.normalize()
    for column in PRICE_COLUMNS[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[PRICE_COLUMNS]


def _download_batch(
    tickers: list[str],
    start: dt.date,
    end: dt.date,
    *,
    retries: int,
    timeout_seconds: int,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    yf = _yf_module()
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            # Yahoo's end parameter is exclusive, so add one calendar day.
            data = yf.download(
                tickers=tickers,
                start=start.isoformat(),
                end=(end + dt.timedelta(days=1)).isoformat(),
                group_by="ticker",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=True,
                timeout=timeout_seconds,
            )
            result = _normalise_download(data, tickers)
            if result.empty:
                raise RuntimeError("Yahoo Finance returned no rows")
            return result
        except Exception as exc:  # noqa: BLE001 - provider exceptions vary
            last_error = exc
            LOGGER.warning(
                "Yahoo price batch attempt %s/%s failed for %s: %s",
                attempt + 1,
                retries,
                ",".join(tickers),
                exc,
            )
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Price download failed for {tickers}: {last_error}")


def _load_cache(path: str | Path) -> pd.DataFrame:
    try:
        frame = read_parquet(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    for column in PRICE_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    frame = frame[PRICE_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def _coverage_complete(cache: pd.DataFrame, ticker: str, start: dt.date, end: dt.date) -> bool:
    rows = cache[cache["ticker"].eq(ticker)]
    if rows.empty:
        return False
    dates = pd.to_datetime(rows["date"])
    # A small calendar-day tolerance covers weekends and exchange holidays;
    # the quality report still records the actual first/last trading date.
    return dates.min().date() <= start + dt.timedelta(days=7) and dates.max().date() >= end - dt.timedelta(days=7)


def _history_is_usable(frame: pd.DataFrame, ticker: str, start: dt.date, end: dt.date) -> bool:
    """Check coverage using finite OHLCV rows rather than date labels alone."""

    subset = frame[frame["ticker"].eq(ticker)].copy()
    if subset.empty:
        return False
    numeric = subset[PRICE_COLUMNS[2:]].apply(pd.to_numeric, errors="coerce")
    dates = pd.to_datetime(subset["date"], errors="coerce")
    valid = dates.notna() & np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    dates = dates.loc[valid]
    if len(dates) < 20:
        return False
    return dates.min().date() <= start + dt.timedelta(days=7) and dates.max().date() >= end - dt.timedelta(days=7)


def _quality_report(frame: pd.DataFrame, tickers: Iterable[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        subset = frame[frame["ticker"].eq(ticker)].sort_values("date")
        finite = np.isfinite(subset[["open", "high", "low", "close", "adj_close"]].to_numpy(dtype=float)) if not subset.empty else np.empty((0, 5), dtype=bool)
        rows.append(
            {
                "ticker": ticker,
                "rows": len(subset),
                "first_date": subset["date"].min().date().isoformat() if not subset.empty else "",
                "last_date": subset["date"].max().date().isoformat() if not subset.empty else "",
                "missing_cells": int(subset.isna().sum().sum()) if not subset.empty else 0,
                "nonfinite_ohlc_cells": int((~finite).sum()) if finite.size else 0,
                "covers_requested_start": bool(not subset.empty and subset["date"].min().date() <= start + dt.timedelta(days=7)),
                "covers_requested_end": bool(not subset.empty and subset["date"].max().date() >= end - dt.timedelta(days=7)),
            }
        )
    return pd.DataFrame(rows)


def fetch_price_data(
    tickers: Iterable[str],
    *,
    start: str | dt.date,
    end: str | dt.date,
    output_path: str | Path,
    quality_path: str | Path | None = None,
    error_path: str | Path | None = None,
    retries: int = 3,
    timeout_seconds: int = 30,
    batch_size: int = 50,
) -> pd.DataFrame:
    """Fetch only missing/incomplete ticker histories and merge into a cache."""

    ticker_list = list(dict.fromkeys(str(ticker) for ticker in tickers))
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    if end_date < start_date:
        raise ValueError("end must not precede start")
    cache = _load_cache(output_path)
    needs_download = [ticker for ticker in ticker_list if not _coverage_complete(cache, ticker, start_date, end_date)]
    downloaded: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    for offset in range(0, len(needs_download), max(1, batch_size)):
        batch = needs_download[offset : offset + max(1, batch_size)]
        try:
            downloaded.append(
                _download_batch(
                    batch,
                    start_date,
                    end_date,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                )
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Batch permanently failed for %s: %s", batch, exc)
            errors.append({"timestamp": pd.Timestamp.utcnow().isoformat(), "tickers": ",".join(batch), "error": str(exc), "start": start_date.isoformat(), "end": end_date.isoformat()})
    if downloaded:
        cache = pd.concat([cache, *downloaded], ignore_index=True)
    if not cache.empty:
        cache = cache[cache["ticker"].isin(ticker_list)].copy()
        cache = cache.drop_duplicates(["date", "ticker"], keep="last").sort_values(["ticker", "date"])
    write_parquet(cache, output_path)
    quality = _quality_report(cache, ticker_list, start_date, end_date)
    if quality_path is None:
        quality_path = Path(output_path).parent / "fetch_quality.csv"
    ensure_parent(quality_path)
    quality.to_csv(quality_path, index=False)
    if error_path is None:
        error_path = Path(output_path).parent.parent / "metadata" / "fetch_errors.csv"
    ensure_parent(error_path)
    error_columns = ["timestamp", "tickers", "error", "start", "end"]
    if errors:
        error_frame = pd.DataFrame(errors, columns=error_columns)
        error_file = Path(error_path)
        if error_file.exists():
            try:
                error_frame = pd.concat([pd.read_csv(error_file), error_frame], ignore_index=True)
            except (OSError, pd.errors.ParserError):
                pass
        error_frame.to_csv(error_file, index=False)
    elif not Path(error_path).exists():
        pd.DataFrame(columns=error_columns).to_csv(error_path, index=False)
    for _, row in quality.iterrows():
        if int(row["rows"]) == 0 or not bool(row["covers_requested_start"]) or not bool(row["covers_requested_end"]):
            LOGGER.warning("Incomplete price history: %s", row.to_dict())
    LOGGER.info("Price cache contains %s rows for %s tickers; downloaded %s incomplete tickers", len(cache), len(ticker_list), len(needs_download))
    return cache


def fetch_selected_and_indices(
    selected_path: str | Path,
    *,
    start: str | dt.date,
    end: str | dt.date,
    prices_path: str | Path,
    indices_path: str | Path,
    prices_quality_path: str | Path | None = None,
    indices_quality_path: str | Path | None = None,
    prices_error_path: str | Path | None = None,
    indices_error_path: str | Path | None = None,
    index_tickers: Iterable[str] | None = None,
    index_fallbacks: Mapping[str, Iterable[str]] | None = None,
    **kwargs: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(selected_path)
    if "ticker" not in selected:
        raise ValueError(f"Selected universe has no ticker column: {selected_path}")
    tickers = selected["ticker"].dropna().astype(str).unique().tolist()
    prices = fetch_price_data(
        tickers,
        start=start,
        end=end,
        output_path=prices_path,
        quality_path=prices_quality_path,
        error_path=prices_error_path,
        **kwargs,
    )
    requested_indices = list(dict.fromkeys(str(ticker) for ticker in (index_tickers or ["^N225", "^TPX"])))
    indices = fetch_price_data(
        requested_indices,
        start=start,
        end=end,
        output_path=indices_path,
        quality_path=indices_quality_path,
        error_path=indices_error_path,
        **kwargs,
    )
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    fallback_map: dict[str, list[str]] = {}
    for ticker, candidates in (index_fallbacks or {}).items():
        if isinstance(candidates, str):
            candidates = [candidates]
        fallback_map[str(ticker)] = [str(candidate) for candidate in candidates]
    for canonical_ticker in requested_indices:
        if _history_is_usable(indices, canonical_ticker, start_date, end_date):
            continue
        fallback_used = False
        for fallback_ticker in fallback_map.get(canonical_ticker, []):
            try:
                fallback = _download_batch(
                    [fallback_ticker],
                    start_date,
                    end_date,
                    retries=int(kwargs.get("retries", 3)),
                    timeout_seconds=int(kwargs.get("timeout_seconds", 30)),
                )
            except Exception as exc:  # noqa: BLE001 - provider exceptions vary
                LOGGER.warning("Index fallback download failed for %s (%s): %s", canonical_ticker, fallback_ticker, exc)
                continue
            if not _history_is_usable(fallback, fallback_ticker, start_date, end_date):
                LOGGER.warning("Index fallback is incomplete for %s (%s)", canonical_ticker, fallback_ticker)
                continue
            fallback = fallback.copy()
            fallback["ticker"] = canonical_ticker
            indices = pd.concat(
                [indices[indices["ticker"].ne(canonical_ticker)], fallback],
                ignore_index=True,
            )
            LOGGER.warning(
                "Using %s as a proxy for unavailable %s index history; rows are stored under %s",
                fallback_ticker,
                canonical_ticker,
                canonical_ticker,
            )
            fallback_used = True
            break
        if not fallback_used and not _history_is_usable(indices, canonical_ticker, start_date, end_date):
            LOGGER.warning("Index history remains incomplete for %s", canonical_ticker)
    indices = indices.drop_duplicates(["date", "ticker"], keep="last").sort_values(["ticker", "date"])
    write_parquet(indices, indices_path)
    if indices_quality_path is not None:
        quality = _quality_report(indices, requested_indices, start_date, end_date)
        ensure_parent(indices_quality_path)
        quality.to_csv(indices_quality_path, index=False)
    return prices, indices


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--selected", default=None)
    args = parser.parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    start = args.start or config.get("data", {}).get("start_date", "2015-01-01")
    end = args.end or runtime_as_of(config).isoformat()
    data_config = config.get("data", {})
    fetch_config = config.get("fetch", {})
    fetch_selected_and_indices(
        args.selected or config_path(config, "selected_universe"),
        start=start,
        end=end,
        prices_path=config_path(config, "raw_prices"),
        indices_path=config_path(config, "raw_indices"),
        prices_quality_path=config_path(config, "raw_prices").parent / "prices_quality.csv",
        indices_quality_path=config_path(config, "raw_indices").parent / "indices_quality.csv",
        prices_error_path=config_path(config, "raw_prices").parent.parent / "metadata" / "fetch_errors_prices.csv",
        indices_error_path=config_path(config, "raw_indices").parent.parent / "metadata" / "fetch_errors_indices.csv",
        retries=int(fetch_config.get("retries", 3)),
        timeout_seconds=int(fetch_config.get("timeout_seconds", 30)),
        batch_size=int(fetch_config.get("batch_size", 50)),
        index_tickers=data_config.get("index_tickers", ["^N225", "^TPX"]),
        index_fallbacks=data_config.get("index_fallbacks", {"^TPX": ["1306.T"]}),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
