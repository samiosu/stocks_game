import numpy as np
import pandas as pd

from stock_sim.fetch_data import PRICE_COLUMNS, fetch_selected_and_indices


def _history(ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    values = 100.0 + np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": values,
            "high": values + 1.0,
            "low": values - 1.0,
            "close": values + 0.5,
            "adj_close": values + 0.5,
            "volume": np.full(len(dates), 1_000_000.0),
        },
        columns=PRICE_COLUMNS,
    )


def test_index_fallback_replaces_incomplete_primary_history(tmp_path, monkeypatch):
    dates = pd.bdate_range("2020-01-01", periods=30)

    def fake_download(tickers, start, end, *, retries, timeout_seconds):
        del start, end, retries, timeout_seconds
        ticker = tickers[0]
        history = _history(ticker, dates)
        return history.iloc[:1].copy() if ticker == "^TPX" else history

    monkeypatch.setattr("stock_sim.fetch_data._download_batch", fake_download)
    selected_path = tmp_path / "selected.csv"
    pd.DataFrame({"ticker": ["1000.T"]}).to_csv(selected_path, index=False)

    _, indices = fetch_selected_and_indices(
        selected_path,
        start=dates[0].date().isoformat(),
        end=dates[-1].date().isoformat(),
        prices_path=tmp_path / "prices.parquet",
        indices_path=tmp_path / "indices.parquet",
        prices_quality_path=tmp_path / "prices_quality.csv",
        indices_quality_path=tmp_path / "indices_quality.csv",
        prices_error_path=tmp_path / "prices_errors.csv",
        indices_error_path=tmp_path / "indices_errors.csv",
        index_fallbacks={"^TPX": ["1306.T"]},
        retries=1,
        timeout_seconds=1,
        batch_size=50,
    )

    assert set(indices["ticker"]) == {"^N225", "^TPX"}
    assert len(indices[indices["ticker"].eq("^TPX")]) == len(dates)
    assert indices[indices["ticker"].eq("^TPX")]["close"].notna().all()
