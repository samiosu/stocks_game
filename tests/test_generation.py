import numpy as np
import pandas as pd
import pytest

from stock_sim.constants import OHLCV_FIELDS, SECTOR_IDS
from stock_sim.generate import (
    _repair_ohlcv_bar,
    event_vector,
    ohlc_frame,
    ohlcv_frame,
    prices_from_returns,
    save_candlestick_chart,
)


def test_price_generation_is_finite_and_seed_independent_for_deterministic_returns():
    returns = np.full((5, 11), 0.01)
    prices_a = prices_from_returns(returns, initial_price=100)
    prices_b = prices_from_returns(returns, initial_price=100)
    assert prices_a.shape == (5, 11)
    assert np.isfinite(prices_a).all()
    assert np.array_equal(prices_a, prices_b)


def test_zero_returns_keep_price_unchanged():
    returns = np.zeros((120, 11))
    prices = prices_from_returns(returns, initial_price=100)
    assert np.array_equal(prices, np.full((120, 11), 100.0))


def test_positive_returns_are_compounded_as_log_returns():
    returns = np.full((120, 11), 0.01)
    prices = prices_from_returns(returns, initial_price=100)
    expected = 100.0 * np.exp(np.arange(1, 121) * 0.01)
    np.testing.assert_allclose(prices[:, 0], expected)
    np.testing.assert_allclose(prices[-1], 100.0 * np.exp(1.2))


def test_negative_returns_are_compounded_as_log_returns():
    returns = np.full((120, 11), -0.01)
    prices = prices_from_returns(returns, initial_price=100)
    expected = 100.0 * np.exp(-np.arange(1, 121) * 0.01)
    np.testing.assert_allclose(prices[:, 0], expected)
    np.testing.assert_allclose(prices[-1], 100.0 * np.exp(-1.2))


def test_ohlc_invariants_and_close_alignment():
    returns = np.full((30, 11), 0.002)
    closes = prices_from_returns(returns, initial_price=100)
    frame = ohlc_frame(pd.bdate_range("2026-01-01", periods=30), returns, closes, seed=9)

    for sector_id in frame.columns[1:12]:
        open_values = frame[f"{sector_id}__open"].to_numpy()
        high_values = frame[f"{sector_id}__high"].to_numpy()
        low_values = frame[f"{sector_id}__low"].to_numpy()
        close_values = frame[f"{sector_id}__close"].to_numpy()
        np.testing.assert_allclose(frame[sector_id].to_numpy(), close_values)
        assert np.isfinite(np.column_stack([open_values, high_values, low_values, close_values])).all()
        assert (high_values >= np.maximum(open_values, close_values)).all()
        assert (low_values <= np.minimum(open_values, close_values)).all()
        assert (low_values > 0).all()


def test_direct_ohlcv_frame_contains_model_output_fields():
    bars = np.empty((4, len(SECTOR_IDS), len(OHLCV_FIELDS)), dtype=float)
    bars[:, :, 0] = 100.0
    bars[:, :, 1] = 105.0
    bars[:, :, 2] = 95.0
    bars[:, :, 3] = 102.0
    bars[:, :, 4] = 1_000_000.0
    frame = ohlcv_frame(pd.bdate_range("2026-01-01", periods=4), bars)
    assert all(f"{SECTOR_IDS[0]}__{field}" in frame for field in OHLCV_FIELDS)
    np.testing.assert_allclose(frame[SECTOR_IDS[0]], frame[f"{SECTOR_IDS[0]}__close"])
    assert (frame[f"{SECTOR_IDS[0]}__volume"] > 0).all()


def test_ohlcv_repair_enforces_positive_values_and_candle_bounds():
    bad = np.full((len(SECTOR_IDS), len(OHLCV_FIELDS)), -1.0)
    repaired = _repair_ohlcv_bar(bad)
    assert (repaired > 0).all()
    assert (repaired[:, 1] >= np.maximum(repaired[:, 0], repaired[:, 3])).all()
    assert (repaired[:, 2] <= np.minimum(repaired[:, 0], repaired[:, 3])).all()


def test_mplfinance_saves_generated_candlestick_image(tmp_path):
    pytest.importorskip("mplfinance")
    returns = np.full((20, 11), 0.001)
    closes = prices_from_returns(returns, initial_price=100)
    frame = ohlc_frame(pd.bdate_range("2026-01-01", periods=20), returns, closes, seed=12)
    output = save_candlestick_chart(frame, tmp_path / "generated_candlestick.png", sector_id="energy")
    assert output.exists()
    assert output.stat().st_size > 0


def test_event_vector_encodes_decay_and_affected_sector():
    columns = ["event__market_shock", "event__market_shock__energy", "event__market_shock__financials"]
    schedule = [{"start_date": "2026-01-01", "event_type": "market_shock", "intensity": 2, "duration": 10, "affected_sectors": ["energy"], "decay_rate": 0.1}]
    first = event_vector("2026-01-01", schedule, columns)
    later = event_vector("2026-01-02", schedule, columns)
    assert first[0] == 2
    assert first[1] == 2
    assert first[2] == 0
    assert later[0] < first[0]
    assert later[1] < first[1]
