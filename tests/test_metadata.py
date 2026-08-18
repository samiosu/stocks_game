import numpy as np
import pandas as pd

from stock_sim.metadata import YahooMetadataResolver


def test_yahoo_metadata_resolver_handles_all_na_text_columns():
    frame = pd.DataFrame(
        {
            "ticker": ["1605.T"],
            "company_name": [np.nan],
            "yahoo_sector": [np.nan],
            "market_cap_jpy": [np.nan],
            "market_cap_as_of": [np.nan],
            "market_cap_retrieved_at": [np.nan],
            "market_cap_calculation_method": [np.nan],
            "market_cap_source": [np.nan],
            "market_cap_currency": [np.nan],
        }
    )
    resolver = YahooMetadataResolver(retries=1)
    resolver._info = lambda _ticker: {
        "longName": "Test Energy",
        "sector": "Energy",
        "marketCap": 1_000_000_000_000,
        "currency": "JPY",
    }

    result = resolver.resolve(frame, as_of="2026-08-18")

    assert result.loc[0, "market_cap_jpy"] == 1_000_000_000_000
    assert result.loc[0, "market_cap_source"] == "yahoo_finance"
    assert result.loc[0, "market_cap_currency"] == "JPY"
    assert isinstance(result.loc[0, "market_cap_retrieved_at"], str)
