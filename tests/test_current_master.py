from pathlib import Path

import pandas as pd

from stock_sim.constants import SECTOR_IDS


def test_bundled_current_master_has_three_fixed_tickers_per_sector():
    path = Path(__file__).parents[1] / "data" / "metadata" / "stock_master.csv"
    master = pd.read_csv(path)
    assert len(master) == 33
    assert set(master["sector_id"]) == set(SECTOR_IDS)
    assert master.groupby("sector_id").size().eq(3).all()
    assert master["ticker"].is_unique
    assert master["continuous_listing_verified"].astype(str).str.lower().eq("true").all()

