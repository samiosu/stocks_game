from pathlib import Path

import pandas as pd

from stock_sim.constants import SECTOR_DEFINITIONS, SECTOR_IDS
from stock_sim.universe import select_universe


def _master_rows():
    rows = []
    for sector_index, (sector_id, sector_name) in enumerate(SECTOR_DEFINITIONS):
        for rank in range(3):
            code = str(1000 + sector_index * 10 + rank)
            rows.append(
                {
                    "ticker": code,
                    "company_name": f"Company {code}",
                    "sector_id": sector_id,
                    "sector_name": sector_name,
                    "exchange": "TSE",
                    "security_type": "common stock",
                    "listing_date": "2010-01-01",
                    "continuous_listing_verified": True,
                    "market_cap_jpy": float(10_000_000_000 - rank * 100_000_000),
                    "market_cap_as_of": "2026-08-18",
                    "foreign_company": False,
                    "is_fund": False,
                    "is_reit": False,
                    "is_preferred": False,
                    "is_infrastructure_fund": False,
                }
            )
    rows.extend(
        [
            {
                "ticker": "9990",
                "company_name": "An ETF",
                "sector_id": "energy",
                "exchange": "TSE",
                "security_type": "ETF",
                "listing_date": "2010-01-01",
                "continuous_listing_verified": True,
                "market_cap_jpy": 99_000_000_000,
                "foreign_company": False,
                "is_fund": True,
                "is_reit": False,
                "is_preferred": False,
                "is_infrastructure_fund": False,
            },
            {
                "ticker": "9991",
                "company_name": "Unknown listing",
                "sector_id": "energy",
                "exchange": "TSE",
                "security_type": "common stock",
                "continuous_listing_verified": True,
                "market_cap_jpy": 98_000_000_000,
                "foreign_company": False,
                "is_fund": False,
                "is_reit": False,
                "is_preferred": False,
                "is_infrastructure_fund": False,
            },
            {
                "ticker": "9992",
                "company_name": "A REIT",
                "sector_id": "real_estate",
                "exchange": "TSE",
                "security_type": "REIT",
                "listing_date": "2010-01-01",
                "continuous_listing_verified": True,
                "market_cap_jpy": 97_000_000_000,
                "foreign_company": False,
                "is_fund": False,
                "is_reit": True,
                "is_preferred": False,
                "is_infrastructure_fund": False,
            },
        ]
    )
    return rows


def test_sector_definition_is_exactly_eleven():
    assert len(SECTOR_DEFINITIONS) == 11
    assert len(SECTOR_IDS) == 11
    assert len(set(SECTOR_IDS)) == 11


def test_selects_three_per_sector_and_excludes_unverified(tmp_path: Path):
    master_path = tmp_path / "master.csv"
    pd.DataFrame(_master_rows()).to_csv(master_path, index=False)
    output_path = tmp_path / "selected.csv"
    rejection_path = tmp_path / "rejections.csv"
    selected = select_universe(
        master_path,
        as_of="2026-08-18",
        output_path=output_path,
        rejection_path=rejection_path,
    )
    assert len(selected) == 33
    assert selected.groupby("sector_id").size().eq(3).all()
    assert selected.groupby("sector_id")["market_cap_jpy"].apply(lambda values: values.is_monotonic_decreasing).all()
    rejected = pd.read_csv(rejection_path)
    assert "fund_etf_reit_preferred_or_infrastructure" in set(rejected["reason"])
    assert "listing_date_unknown" in set(rejected["reason"])
    assert not selected["ticker"].str.contains("9990").any()
