"""Eligibility filtering and sector-wise top-N universe selection."""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import ensure_parent
from .constants import SECTOR_DEFINITIONS, SECTOR_IDS, SECTOR_NAMES
from .metadata import (
    YahooMetadataResolver,
    apply_listing_overrides,
    load_sector_overrides,
    load_stock_master,
    normalize_ticker,
    ticker_key,
)

LOGGER = logging.getLogger(__name__)


def _mapping(path: str | Path | None) -> dict[str, str]:
    if path is None or not Path(path).exists():
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    result: dict[str, str] = {}
    for section in ("yahoo_to_sector_id", "gics_to_sector_id", "jpx_to_sector_id"):
        result.update({str(k): str(v) for k, v in (raw.get(section) or {}).items()})
    return result


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == ""


def _bool_or_none(value: Any) -> bool | None:
    if _missing(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "verified", "確認済み"}:
        return True
    if text in {"0", "false", "no", "n", "unverified", "未確認"}:
        return False
    return None


def _is_japanese_exchange(row: pd.Series) -> bool:
    exchange = str(row.get("exchange", "")).strip().lower()
    if not exchange:
        # A .T ticker is a useful fallback for manually prepared masters, but
        # an explicit non-Japanese exchange is never accepted.
        return str(row.get("ticker", "")).upper().endswith(".T")
    return any(
        token in exchange
        for token in ("tse", "tokyo", "jpx", "東証", "prime", "standard", "growth", "プライム", "スタンダード", "グロース", "内国株式")
    )


def _exclusion_reason(row: pd.Series) -> str | None:
    if not _is_japanese_exchange(row):
        return "not_a_japanese_exchange"
    bool_columns = {
        "foreign_company": "foreign_company",
        "is_fund": "fund",
        "is_reit": "reit",
        "is_preferred": "preferred_stock",
        "is_infrastructure_fund": "infrastructure_fund",
    }
    for column in bool_columns:
        flag = _bool_or_none(row.get(column))
        if flag is True:
            return "fund_etf_reit_preferred_or_infrastructure"
        if flag is None:
            return f"{column}_unverified"
    security = str(row.get("security_type", "")).strip().lower()
    if not security:
        return "security_type_unknown"
    name = str(row.get("company_name", "")).strip().lower()
    exclusion_tokens = (
        "etf",
        "etn",
        "reit",
        "リート",
        "投資信託",
        "インフラファンド",
        "優先株",
        "preferred",
        "infrastructure fund",
    )
    if any(token in security or token in name for token in exclusion_tokens):
        return "fund_etf_reit_preferred_or_infrastructure"
    if security and not any(token in security for token in ("common", "ordinary", "普通", "株式")):
        return "not_common_stock"
    return None


def _date(value: Any) -> dt.date | None:
    if _missing(value):
        return None
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError):
        return None


def _eligibility(row: pd.Series, start_date: dt.date, as_of: dt.date) -> str | None:
    reason = _exclusion_reason(row)
    if reason:
        return reason
    listing_date = _date(row.get("listing_date"))
    if listing_date is None:
        return "listing_date_unknown"
    if listing_date > start_date:
        return "listed_after_training_start"
    verified = _bool_or_none(row.get("continuous_listing_verified"))
    if verified is not True:
        return "continuous_listing_not_verified"
    delisted_from = _date(row.get("delisted_from"))
    delisted_to = _date(row.get("delisted_to"))
    if delisted_from is not None and delisted_from <= as_of:
        # A delisting before the as-of date is a failed continuity check. The
        # end date is retained for audit but does not make the gap disappear.
        return "delisted_during_observation_period"
    if delisted_to is not None and delisted_to >= start_date:
        return "had_delisting_period"
    return None


def _resolve_sector(
    row: pd.Series,
    overrides: dict[str, dict[str, str]],
    mapping: dict[str, str],
) -> tuple[str | None, str | None, str]:
    key = ticker_key(row.get("ticker", ""))
    if key in overrides:
        override = overrides[key]
        sector_id = override.get("sector_id")
        if sector_id in SECTOR_IDS:
            return sector_id, SECTOR_NAMES[sector_id], "manual_override"
    sector_id = str(row.get("sector_id", "")).strip()
    if sector_id in SECTOR_IDS:
        return sector_id, SECTOR_NAMES[sector_id], "stock_master"
    for column in ("yahoo_sector", "sector_name"):
        value = str(row.get(column, "")).strip()
        if value in mapping and mapping[value] in SECTOR_IDS:
            sector_id = mapping[value]
            return sector_id, SECTOR_NAMES[sector_id], f"mapping:{column}"
    return None, None, "unclassified"


def _rejection_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = ["ticker", "company_name", "reason", "sector_id", "as_of"]
    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows)
    for column in columns:
        if column not in result:
            result[column] = ""
    return result[columns]


def select_universe(
    master_path: str | Path,
    *,
    as_of: str | dt.date,
    start_date: str | dt.date = "2015-01-01",
    sector_mapping_path: str | Path | None = None,
    sector_overrides_path: str | Path | None = None,
    listing_overrides_path: str | Path | None = None,
    output_path: str | Path = "data/selected_universe.csv",
    rejection_path: str | Path = "data/metadata/universe_rejections.csv",
    max_per_sector: int = 3,
    metadata_resolver: YahooMetadataResolver | None = None,
) -> pd.DataFrame:
    """Select at most ``max_per_sector`` verified common stocks per sector.

    The function is intentionally injectable: tests and offline runs can pass a
    fully populated master without touching Yahoo Finance.
    """

    as_of_date = pd.Timestamp(as_of).date()
    start = pd.Timestamp(start_date).date()
    master = load_stock_master(master_path)
    if master.empty:
        raise ValueError(
            f"Stock master is empty: {master_path}. Populate it from a JPX export or a manual CSV."
        )
    if listing_overrides_path is not None:
        master = apply_listing_overrides(master, listing_overrides_path)
    overrides = load_sector_overrides(sector_overrides_path) if sector_overrides_path else {}
    mapping = _mapping(sector_mapping_path)

    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for _, original in master.iterrows():
        row = original.copy()
        row["ticker"] = normalize_ticker(row["ticker"])
        sector_id, sector_name, sector_source = _resolve_sector(row, overrides, mapping)
        if sector_id is None:
            rejected.append(
                {
                    "ticker": row["ticker"],
                    "company_name": row.get("company_name", ""),
                    "reason": "sector_unclassified",
                    "sector_id": "",
                    "as_of": as_of_date.isoformat(),
                }
            )
            continue
        reason = _eligibility(row, start, as_of_date)
        if reason:
            rejected.append(
                {
                    "ticker": row["ticker"],
                    "company_name": row.get("company_name", ""),
                    "reason": reason,
                    "sector_id": sector_id,
                    "as_of": as_of_date.isoformat(),
                }
            )
            continue
        item = row.to_dict()
        item.update({"ticker": row["ticker"], "sector_id": sector_id, "sector_name": sector_name, "sector_source": sector_source})
        candidates.append(item)

    candidate_frame = pd.DataFrame(candidates)
    if not candidate_frame.empty:
        missing_cap = candidate_frame["market_cap_jpy"].isna() | (candidate_frame["market_cap_jpy"] <= 0)
        if missing_cap.any() and metadata_resolver is not None:
            candidate_frame = metadata_resolver.resolve(candidate_frame, as_of_date.isoformat())
        for index, row in candidate_frame.iterrows():
            cap = pd.to_numeric(pd.Series([row.get("market_cap_jpy")]), errors="coerce").iloc[0]
            if pd.isna(cap) or cap <= 0:
                rejected.append(
                    {
                        "ticker": row.get("ticker", ""),
                        "company_name": row.get("company_name", ""),
                        "reason": "market_cap_unavailable",
                        "sector_id": row.get("sector_id", ""),
                        "as_of": as_of_date.isoformat(),
                    }
                )
                candidate_frame.at[index, "_invalid"] = True
            else:
                candidate_frame.at[index, "market_cap_jpy"] = float(cap)
        if "_invalid" in candidate_frame:
            candidate_frame = candidate_frame[candidate_frame["_invalid"] != True].copy()
            candidate_frame = candidate_frame.drop(columns=["_invalid"])

    selected: list[dict[str, Any]] = []
    if not candidate_frame.empty:
        candidate_frame["market_cap_jpy"] = pd.to_numeric(candidate_frame["market_cap_jpy"], errors="coerce")
        candidate_frame["market_cap_as_of"] = candidate_frame["market_cap_as_of"].fillna(as_of_date.isoformat())
        for sector_id, sector_name in SECTOR_DEFINITIONS:
            sector_rows = candidate_frame[candidate_frame["sector_id"].eq(sector_id)].copy()
            sector_rows = sector_rows.sort_values(["market_cap_jpy", "ticker"], ascending=[False, True])
            if len(sector_rows) < max_per_sector:
                LOGGER.warning("Sector %s has only %s eligible stocks", sector_id, len(sector_rows))
            for rank, (_, row) in enumerate(sector_rows.head(max_per_sector).iterrows(), start=1):
                item = row.to_dict()
                item["market_cap_rank_in_sector"] = rank
                item["continuous_listing_verified"] = True
                item["listing_date"] = _date(item.get("listing_date")).isoformat()
                item["market_cap_as_of"] = _date(item.get("market_cap_as_of")).isoformat() if _date(item.get("market_cap_as_of")) else as_of_date.isoformat()
                if _missing(item.get("market_cap_source")):
                    item["market_cap_source"] = "stock_master"
                if _missing(item.get("market_cap_currency")):
                    item["market_cap_currency"] = "JPY"
                if _missing(item.get("market_cap_calculation_method")):
                    item["market_cap_calculation_method"] = "provided market_cap_jpy"
                if _missing(item.get("market_cap_retrieved_at")):
                    item["market_cap_retrieved_at"] = pd.Timestamp.now(tz="UTC").isoformat()
                item["metadata_source"] = item.get("market_cap_source", "stock_master")
                selected.append(item)

    output_columns = [
        "sector_id",
        "sector_name",
        "ticker",
        "company_name",
        "market_cap_jpy",
        "market_cap_as_of",
        "listing_date",
        "continuous_listing_verified",
        "market_cap_rank_in_sector",
        "sector_source",
        "metadata_source",
        "market_cap_currency",
        "market_cap_calculation_method",
        "market_cap_retrieved_at",
    ]
    selected_frame = pd.DataFrame(selected)
    for column in output_columns:
        if column not in selected_frame:
            selected_frame[column] = ""
    selected_frame = selected_frame[output_columns]
    selected_frame = selected_frame.sort_values(["sector_id", "market_cap_rank_in_sector"]).reset_index(drop=True)
    ensure_parent(output_path)
    selected_frame.to_csv(output_path, index=False)
    ensure_parent(rejection_path)
    _rejection_frame(rejected).to_csv(rejection_path, index=False)
    LOGGER.info("Selected %s stocks across %s sectors", len(selected_frame), selected_frame["sector_id"].nunique())
    return selected_frame
