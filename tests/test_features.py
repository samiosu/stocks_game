import numpy as np
import pandas as pd

from stock_sim.constants import OHLCV_FIELDS, SECTOR_DEFINITIONS
from stock_sim.dataset import make_ohlcv_window_arrays, make_window_arrays
from stock_sim.features import build_feature_frame
from stock_sim.preprocessing import FeatureScaler


def _synthetic_inputs(days=90):
    dates = pd.bdate_range("2015-01-01", periods=days)
    prices = []
    selected = []
    for index, (sector_id, sector_name) in enumerate(SECTOR_DEFINITIONS):
        ticker = f"{7000 + index}.T"
        selected.append({"ticker": ticker, "sector_id": sector_id, "sector_name": sector_name, "market_cap_jpy": 1e10})
        close = 100 + index + np.arange(days) * (0.03 + index / 1000)
        for date, value in zip(dates, close):
            prices.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": value * 0.99,
                    "high": value * 1.01,
                    "low": value * 0.98,
                    "close": value,
                    "adj_close": value,
                    "volume": 100000 + np.arange(days)[len([row for row in prices if row["ticker"] == ticker])],
                }
            )
    indices = []
    for ticker in ("^N225", "^TPX"):
        for day, date in enumerate(dates):
            value = 1000 + day * 0.5
            indices.append({"date": date, "ticker": ticker, "open": value, "high": value * 1.01, "low": value * 0.99, "close": value, "adj_close": value, "volume": 1e6})
    return pd.DataFrame(prices), pd.DataFrame(indices), pd.DataFrame(selected)


def test_feature_frame_has_no_nan_and_expected_sector_features():
    prices, indices, selected = _synthetic_inputs()
    sector, stocks, schema = build_feature_frame(prices, indices, selected)
    assert len(sector) > 0
    assert len(stocks) > 0
    assert len(schema["return_columns"]) == 11
    assert len(schema["ohlcv_columns"]) == 11 * len(OHLCV_FIELDS)
    assert np.isfinite(sector[schema["feature_columns"]].to_numpy()).all()
    assert np.isfinite(sector[schema["ohlcv_columns"]].to_numpy()).all()
    assert np.isfinite(sector[schema["return_columns"]].to_numpy()).all()
    assert any(column.startswith("event__") for column in schema["event_columns"])


def test_windows_do_not_include_future_rows_in_input():
    frame = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=12), "feature": np.arange(12, dtype=float), "return": np.arange(12, dtype=float) / 100, "vol": np.full(12, 0.1)})
    scaler = FeatureScaler(["feature"]).fit(frame.loc[:7, ["feature"]])
    first = make_window_arrays(frame, feature_columns=["feature"], return_columns=["return"], volatility_columns=["vol"], sequence_length=3, horizon=1, scaler=scaler)
    changed = frame.copy()
    changed.loc[11, "feature"] = 999999.0
    second = make_window_arrays(changed, feature_columns=["feature"], return_columns=["return"], volatility_columns=["vol"], sequence_length=3, horizon=1, scaler=scaler)
    assert np.array_equal(first.features[:-1], second.features[:-1])


def test_ohlcv_windows_use_the_same_columns_for_input_and_target():
    prices, indices, selected = _synthetic_inputs(days=100)
    sector, _, schema = build_feature_frame(prices, indices, selected)
    columns = schema["ohlcv_columns"]
    scaler = FeatureScaler(columns).fit(sector.loc[:49, columns])
    arrays = make_ohlcv_window_arrays(
        sector,
        ohlcv_columns=columns,
        sequence_length=5,
        horizon=1,
        scaler=scaler,
        volume_lookback=5,
    )
    assert arrays.features.shape[1:] == (5, 11 * len(OHLCV_FIELDS))
    assert arrays.targets.shape[1] == 11 * len(OHLCV_FIELDS)
    assert arrays.teacher_features is not None
    assert arrays.teacher_features.shape[1:] == (1, 11 * len(OHLCV_FIELDS))
    assert np.isfinite(arrays.features).all()
    assert np.isfinite(arrays.targets).all()
