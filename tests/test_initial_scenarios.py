import json

import numpy as np
import pandas as pd
import pytest

from stock_sim.constants import OHLCV_FIELDS, SECTOR_IDS
from stock_sim.initial_scenarios import REGIMES, export_initial_scenarios, measure_windows


@pytest.fixture
def scenario_history(tmp_path):
    sectors = list(reversed(SECTOR_IDS))
    columns = [f"{sector}__{field}" for sector in sectors for field in OHLCV_FIELDS]
    noise = 0.0005 * np.sin(np.arange(120))
    returns = np.concatenate(
        [
            0.0025 + noise,
            -0.0025 + noise,
            0.001 * np.sin(np.arange(120)),
            np.tile([0.04, -0.04], 60),
        ]
    )
    close = np.exp(np.cumsum(returns))[:, None] * np.arange(100.0, 111.0)[None, :]
    bars = np.stack(
        [
            close * 0.999,
            close * 1.01,
            close * 0.99,
            close,
            np.full_like(close, 10000),
        ],
        axis=-1,
    )
    frame = pd.DataFrame(bars.reshape(-1, 55), columns=columns)
    frame["date"] = pd.bdate_range("2020-01-01", periods=len(frame))
    metadata = {
        "inputType": "OHLCV",
        "sequenceLength": 60,
        "featureSize": 55,
        "ohlcvColumns": columns,
        "sectorIds": sectors,
        "ohlcvFields": list(OHLCV_FIELDS),
        "onnx": {"inputShape": [1, 60, 55]},
    }
    data_path = tmp_path / "sector_data.parquet"
    metadata_path = tmp_path / "model.metadata.json"
    calendar_path = tmp_path / "prices.parquet"
    frame.iloc[::-1, ::-1].to_parquet(data_path, index=False)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    frame[["date"]].to_parquet(calendar_path, index=False)
    return frame, metadata, data_path, metadata_path, calendar_path


def test_export_all_regimes_preserves_raw_contiguous_nonoverlapping_windows(
    scenario_history, tmp_path
):
    frame, metadata, data_path, metadata_path, calendar_path = scenario_history
    output_dir = tmp_path / "scenarios"
    catalog_path = export_initial_scenarios(
        data_path,
        metadata_path,
        calendar_path,
        output_dir,
        per_regime=1,
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert catalog["windowCount"] == 4
    assert [item["regime"] for item in catalog["scenarios"]] == list(REGIMES)
    all_dates = set()
    for item in catalog["scenarios"]:
        payload = json.loads((output_dir / item["file"]).read_text())
        expected = frame.loc[frame.date.between(item["startDate"], item["endDate"])]
        assert len(expected) == 60
        assert payload["dates"] == expected.date.dt.strftime("%Y-%m-%d").tolist()
        assert not all_dates.intersection(payload["dates"])
        all_dates.update(payload["dates"])
        assert payload["ohlcvColumns"] == metadata["ohlcvColumns"]
        np.testing.assert_array_equal(
            np.asarray(payload["values"], dtype=np.float32).reshape(60, 55),
            expected[metadata["ohlcvColumns"]].to_numpy(np.float32),
        )
        metric = item["metrics"]
        if item["regime"] in ("bull", "bear"):
            direction = 1 if item["regime"] == "bull" else -1
            assert direction * metric["totalReturn"] >= 0.05
            assert metric["trendRSquared"] >= 0.5
        elif item["regime"] == "sideways":
            assert abs(metric["totalReturn"]) <= 0.02
            assert metric["priceRange"] <= 0.08
        else:
            assert (
                metric["dailyVolatility"]
                >= catalog["selection"]["criteria"]["volatileDailyVolatilityMinimum"]
            )


def test_metrics_do_not_depend_on_sector_price_levels(scenario_history):
    frame, metadata, *_ = scenario_history
    _, baseline, _ = measure_windows(frame, metadata, frame.date)
    scaled = frame.copy()
    for index, sector in enumerate(metadata["sectorIds"]):
        columns = [f"{sector}__{field}" for field in OHLCV_FIELDS[:4]]
        scaled[columns] *= 10.0 ** (index % 5 - 2)
    _, actual, _ = measure_windows(scaled, metadata, frame.date)
    metrics = ["totalReturn", "dailyVolatility", "priceRange", "trendRSquared"]
    np.testing.assert_allclose(actual[metrics], baseline[metrics], rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, 0.0, -1.0])
def test_excludes_bad_ohlcv_and_missing_trading_days_without_stitching(scenario_history, bad_value):
    frame, metadata, *_ = scenario_history
    original = frame.copy()
    bad_date = frame.date.iloc[125]
    gap_date = frame.date.iloc[250]
    frame = frame.copy()
    frame.loc[125, metadata["ohlcvColumns"][0]] = bad_value
    frame = frame.drop(index=250)
    _, windows, quality = measure_windows(frame, metadata, original.date)
    assert quality["invalidOhlcvRows"] == 1
    assert quality["excludedInvalidOhlcvWindows"] == 60
    assert quality["excludedNonconsecutiveWindows"] == 59
    for date in (bad_date, gap_date):
        assert not (
            (pd.to_datetime(windows.startDate) <= date) & (pd.to_datetime(windows.endDate) >= date)
        ).any()
    assert quality["possibleWindows"] == (
        quality["validWindows"]
        + quality["excludedInvalidOhlcvWindows"]
        + quality["excludedNonconsecutiveWindows"]
    )


def test_excludes_invalid_candle_bounds(scenario_history):
    frame, metadata, *_ = scenario_history
    frame = frame.copy()
    frame.loc[125, metadata["ohlcvColumns"][2]] = 1e9
    _, _, quality = measure_windows(frame, metadata, frame.date)
    assert quality["invalidOhlcvRows"] == 1
    assert quality["excludedInvalidOhlcvWindows"] == 60


def test_missing_calendar_date_is_not_assumed_to_be_a_holiday(scenario_history):
    frame, metadata, *_ = scenario_history
    with pytest.raises(ValueError, match="calendar"):
        measure_windows(frame, metadata, frame.date.drop(index=100))


def test_catalog_and_payloads_are_reproducible(scenario_history, tmp_path):
    _, _, data_path, metadata_path, calendar_path = scenario_history
    first = tmp_path / "first"
    second = tmp_path / "second"
    for output_dir in (first, second):
        export_initial_scenarios(
            data_path,
            metadata_path,
            calendar_path,
            output_dir,
            per_regime=1,
        )
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()


@pytest.mark.parametrize("per_regime", [0, -1, 100])
def test_insufficient_scenarios_fail_before_writing(scenario_history, tmp_path, per_regime):
    _, _, data_path, metadata_path, calendar_path = scenario_history
    output_dir = tmp_path / "scenarios"
    with pytest.raises(ValueError):
        export_initial_scenarios(
            data_path,
            metadata_path,
            calendar_path,
            output_dir,
            per_regime=per_regime,
        )
    assert not output_dir.exists()
