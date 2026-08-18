import numpy as np

from stock_sim.generate import event_vector, prices_from_returns


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
