"""Stock-master loading and Yahoo Finance metadata resolution."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

MASTER_COLUMNS = [
    "ticker",
    "company_name",
    "sector_id",
    "sector_name",
    "yahoo_sector",
    "exchange",
    "security_type",
    "listing_date",
    "delisted_from",
    "delisted_to",
    "continuous_listing_verified",
    "shares_outstanding",
    "market_cap_jpy",
    "market_cap_as_of",
    "market_cap_retrieved_at",
    "foreign_company",
    "is_fund",
    "is_reit",
    "is_preferred",
    "is_infrastructure_fund",
]


def ticker_key(ticker: Any) -> str:
    """Normalize a ticker for matching overrides and metadata rows."""

    value = str(ticker).strip()
    value = re.sub(r"^(\d+)\.0$", r"\1", value)
    value = value.removesuffix(".T")
    return value.upper()


def normalize_ticker(ticker: Any) -> str:
    value = str(ticker).strip().upper()
    value = re.sub(r"^(\d+)\.0$", r"\1", value)
    if value.startswith("^") or "." in value:
        return value
    return f"{value}.T"


def _as_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t", "verified", "確認済み"}:
        return True
    if text in {"0", "false", "no", "n", "f", "unverified", "未確認"}:
        return False
    return None


def _clean_date(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        return None
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError):
        return None


def load_csv(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        if columns is None:
            raise FileNotFoundError(path)
        return pd.DataFrame(columns=columns)
    try:
        frame = pd.read_csv(path)
    except UnicodeDecodeError:
        # JPX exports are commonly Shift-JIS/CP932 encoded.
        frame = pd.read_csv(path, encoding="cp932")
    if columns:
        for column in columns:
            if column not in frame:
                frame[column] = np.nan
        frame = frame[columns + [c for c in frame.columns if c not in columns]]
    return frame


JPX_COLUMN_ALIASES = {
    "コード": "ticker",
    "銘柄コード": "ticker",
    "銘柄名": "company_name",
    "会社名": "company_name",
    "市場・商品区分": "exchange",
    "市場区分": "exchange",
    "上場日": "listing_date",
    "33業種区分": "sector_name",
    "業種": "sector_name",
}


def read_jpx_master(path: str | Path) -> pd.DataFrame:
    """Convert a JPX stock-list export into the project's master schema.

    JPX does not by itself prove continuous listing since 2015 or provide the
    current market cap. Those fields intentionally remain blank and must be
    verified/enriched before universe selection.
    """

    source = load_csv(path)
    source = source.rename(columns={key: value for key, value in JPX_COLUMN_ALIASES.items() if key in source.columns})
    if "security_type" not in source:
        exchange_values = source["exchange"] if "exchange" in source else pd.Series("", index=source.index)
        source["security_type"] = exchange_values.astype(str).map(
            lambda value: "common stock" if "内国株式" in value or "普通株" in value else value
        )
    common_mask = source["security_type"].astype(str).str.contains("common stock|普通株", case=False, regex=True, na=False)
    for column in ("foreign_company", "is_fund", "is_reit", "is_preferred", "is_infrastructure_fund"):
        if column not in source:
            source[column] = np.nan
        source.loc[common_mask & source[column].isna(), column] = False
    source["ticker"] = source["ticker"].map(normalize_ticker)
    for column in MASTER_COLUMNS:
        if column not in source:
            source[column] = np.nan
    return source[MASTER_COLUMNS]


def write_jpx_master(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Create a normalized manual-review master CSV from a JPX export."""

    frame = read_jpx_master(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def load_stock_master(path: str | Path) -> pd.DataFrame:
    frame = load_csv(path, MASTER_COLUMNS)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].map(normalize_ticker)
    for column in ("listing_date", "delisted_from", "delisted_to", "market_cap_as_of"):
        frame[column] = frame[column].map(_clean_date)
    for column in (
        "continuous_listing_verified",
        "foreign_company",
        "is_fund",
        "is_reit",
        "is_preferred",
        "is_infrastructure_fund",
    ):
        frame[column] = frame[column].map(_as_bool)
    for column in ("shares_outstanding", "market_cap_jpy"):
        frame[column] = pd.to_numeric(frame[column].astype(str).str.replace(",", "", regex=False), errors="coerce")
    return frame


def load_sector_overrides(path: str | Path) -> dict[str, dict[str, str]]:
    frame = load_csv(path, ["ticker", "sector_id", "sector_name", "reason"])
    if frame.empty:
        return {}
    return {
        ticker_key(row.ticker): {
            "sector_id": str(row.sector_id).strip(),
            "sector_name": str(row.sector_name).strip() if pd.notna(row.sector_name) else "",
            "reason": str(row.reason).strip() if pd.notna(row.reason) else "",
        }
        for row in frame.itertuples(index=False)
        if pd.notna(row.ticker) and pd.notna(row.sector_id)
    }


def apply_listing_overrides(frame: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    overrides = load_csv(path, ["ticker", "listing_date", "continuous_listing_verified", "reason"])
    if overrides.empty or frame.empty:
        return frame
    result = frame.copy()
    for row in overrides.itertuples(index=False):
        key = ticker_key(row.ticker)
        mask = result["ticker"].map(ticker_key).eq(key)
        if not mask.any():
            continue
        if pd.notna(row.listing_date) and str(row.listing_date).strip():
            result.loc[mask, "listing_date"] = _clean_date(row.listing_date)
        verified = _as_bool(row.continuous_listing_verified)
        if verified is not None:
            result.loc[mask, "continuous_listing_verified"] = verified
        result.loc[mask, "listing_override_reason"] = (
            str(row.reason) if pd.notna(row.reason) else "manual override"
        )
    return result


class YahooMetadataResolver:
    """Resolve missing market-cap metadata with bounded retries.

    This resolver is deliberately called only for candidates that have already
    passed the static master-file checks. Price history itself is fetched in a
    batch by :mod:`stock_sim.fetch_data`.
    """

    def __init__(self, retries: int = 3, timeout_seconds: int = 30, cache_path: str | Path | None = None) -> None:
        self.retries = max(1, retries)
        self.timeout_seconds = timeout_seconds
        self.cache_path = Path(cache_path) if cache_path else None
        self._yf = None
        self._cache: pd.DataFrame | None = None

    def _load_cache(self) -> pd.DataFrame:
        if self._cache is not None:
            return self._cache
        if self.cache_path is None or not self.cache_path.exists():
            self._cache = pd.DataFrame(columns=["ticker", "as_of", "fetched_at", "raw_info_json"])
            return self._cache
        try:
            self._cache = pd.read_parquet(self.cache_path)
            for column in ("ticker", "as_of", "fetched_at", "raw_info_json"):
                if column not in self._cache:
                    self._cache[column] = np.nan
        except (ImportError, OSError, ValueError) as exc:
            LOGGER.warning("Could not read metadata cache %s: %s", self.cache_path, exc)
            self._cache = pd.DataFrame(columns=["ticker", "as_of", "fetched_at", "raw_info_json"])
        return self._cache

    def _cached_info(self, ticker: str, as_of: str) -> dict[str, Any] | None:
        cache = self._load_cache()
        rows = cache[(cache["ticker"].astype(str).map(ticker_key) == ticker_key(ticker)) & cache["as_of"].astype(str).eq(as_of)]
        if rows.empty:
            return None
        try:
            value = json.loads(str(rows.iloc[-1]["raw_info_json"]))
            return value if isinstance(value, dict) else None
        except (TypeError, json.JSONDecodeError):
            return None

    def _save_info(self, ticker: str, as_of: str, info: dict[str, Any]) -> None:
        if self.cache_path is None:
            return
        cache = self._load_cache()
        row = pd.DataFrame(
            [{"ticker": ticker, "as_of": as_of, "fetched_at": pd.Timestamp.utcnow().isoformat(), "raw_info_json": json.dumps(info, default=str)}]
        )
        cache = pd.concat([cache, row], ignore_index=True).drop_duplicates(["ticker", "as_of"], keep="last")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(self.cache_path, index=False)

    def _module(self):
        if self._yf is None:
            try:
                import yfinance as yf
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise RuntimeError(
                    "yfinance is required to resolve missing market-cap metadata; "
                    "fill market_cap_jpy in the stock master instead"
                ) from exc
            self._yf = yf
        return self._yf

    def _info(self, ticker: str) -> dict[str, Any]:
        yf = self._module()
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                security = yf.Ticker(ticker)
                info = security.get_info() if hasattr(security, "get_info") else security.info
                return dict(info or {})
            except Exception as exc:  # noqa: BLE001 - external provider errors vary
                last_error = exc
                LOGGER.warning("Yahoo metadata attempt %s/%s failed for %s: %s", attempt + 1, self.retries, ticker, exc)
                if attempt + 1 < self.retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Could not resolve Yahoo metadata for {ticker}: {last_error}")

    def _price_on_date(self, ticker: str, as_of: str) -> float | None:
        """Read the last close on or before an arbitrary as-of date."""

        yf = self._module()
        date = dt.date.fromisoformat(as_of[:10])
        start = date - dt.timedelta(days=7)
        end = date + dt.timedelta(days=1)
        security = yf.Ticker(ticker)
        try:
            history = security.history(
                start=start.isoformat(),
                end=end.isoformat(),
                auto_adjust=False,
                timeout=self.timeout_seconds,
            )
        except TypeError:  # older yfinance does not expose timeout here
            history = security.history(start=start.isoformat(), end=end.isoformat(), auto_adjust=False)
        if history is None or history.empty or "Close" not in history:
            return None
        history = history.loc[pd.to_datetime(history.index).date <= date]
        if history.empty:
            return None
        value = pd.to_numeric(history["Close"], errors="coerce").dropna()
        return float(value.iloc[-1]) if not value.empty else None

    def resolve(self, frame: pd.DataFrame, as_of: str) -> pd.DataFrame:
        result = frame.copy()
        # Pandas 3 keeps all-NA columns as ``float64`` when they come from a
        # CSV.  These fields are populated with provider metadata below, so
        # declare them as object columns before assigning strings.
        metadata_text_columns = (
            "company_name",
            "yahoo_sector",
            "market_cap_as_of",
            "market_cap_retrieved_at",
            "market_cap_calculation_method",
            "market_cap_source",
            "market_cap_currency",
        )
        for column in metadata_text_columns:
            if column not in result.columns:
                result[column] = pd.Series(pd.NA, index=result.index, dtype="object")
            else:
                result[column] = result[column].astype("object")
        for index, row in result.iterrows():
            if pd.notna(row.get("market_cap_jpy")) and float(row["market_cap_jpy"]) > 0:
                continue
            ticker = str(row["ticker"])
            try:
                info = self._cached_info(ticker, as_of) or self._info(ticker)
                try:
                    self._save_info(ticker, as_of, info)
                except Exception as exc:  # noqa: BLE001 - cache is an optimization
                    LOGGER.warning("Could not write metadata cache for %s: %s", ticker, exc)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Metadata unavailable for %s: %s", ticker, exc)
                continue
            if not str(row.get("company_name", "")).strip() and info.get("longName"):
                result.at[index, "company_name"] = info["longName"]
            if not str(row.get("yahoo_sector", "")).strip() and info.get("sector"):
                result.at[index, "yahoo_sector"] = info["sector"]
            shares = info.get("sharesOutstanding")
            market_cap = info.get("marketCap")
            currency = str(info.get("currency") or "JPY").upper()
            requested_date = dt.date.fromisoformat(as_of[:10])
            use_current_market_cap = requested_date >= dt.datetime.now(dt.timezone.utc).date()
            if market_cap is not None and currency == "JPY" and use_current_market_cap:
                result.at[index, "market_cap_jpy"] = float(market_cap)
                result.at[index, "market_cap_calculation_method"] = "Yahoo Finance marketCap"
                result.at[index, "market_cap_source"] = "yahoo_finance"
                result.at[index, "market_cap_currency"] = currency
            elif shares is not None and currency == "JPY":
                # ``regularMarketPrice`` is a quote in the same currency as the
                # share count. We refuse an unknown FX conversion rather than
                # silently mixing currencies.
                price = self._price_on_date(ticker, as_of) if not use_current_market_cap else (info.get("regularMarketPrice") or info.get("currentPrice"))
                if price is not None and currency == "JPY":
                    result.at[index, "shares_outstanding"] = float(shares)
                    result.at[index, "market_cap_jpy"] = float(price) * float(shares)
                    result.at[index, "market_cap_calculation_method"] = (
                        "historical close at as-of * shares outstanding" if not use_current_market_cap else "latest price * shares outstanding"
                    )
                    result.at[index, "market_cap_source"] = "yahoo_finance"
                    result.at[index, "market_cap_currency"] = currency
            result.at[index, "market_cap_as_of"] = as_of
            if pd.notna(result.at[index, "market_cap_jpy"]):
                result.at[index, "market_cap_retrieved_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Normalize a JPX CSV into stock_master.csv")
    parser.add_argument("--jpx-input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    write_jpx_master(args.jpx_input, args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
