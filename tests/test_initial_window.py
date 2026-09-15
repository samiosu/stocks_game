import json

import numpy as np
import pandas as pd
import pytest

from stock_sim.constants import OHLCV_FIELDS, SECTOR_IDS
from stock_sim.initial_window import export_initial_window, main


@pytest.fixture
def window_files(tmp_path):
    # Deliberately differ from the project's usual sector order and source column order.
    sectors = list(reversed(SECTOR_IDS))
    columns = [f"{sector}__{field}" for sector in sectors for field in OHLCV_FIELDS]
    bars = np.arange(62 * 11, dtype=np.float64).reshape(62, 11, 1) + 100.123456789
    bars = bars + np.array([0, 3, -2, 1, 10000])
    frame = pd.DataFrame(bars.reshape(62, 55), columns=columns)
    frame["date"] = pd.bdate_range("2026-01-01", periods=62)
    data_path = tmp_path / "sector_data.parquet"
    frame.iloc[::-1, ::-1].to_parquet(data_path, index=False)
    metadata = {
        "inputType": "OHLCV",
        "sequenceLength": 60,
        "featureSize": 55,
        "ohlcvColumns": columns,
        "sectorIds": sectors,
        "ohlcvFields": list(OHLCV_FIELDS),
        "onnx": {"inputShape": [1, 60, 55]},
    }
    metadata_path = tmp_path / "model.metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return data_path, metadata_path, tmp_path / "export" / "initial.json", frame, metadata


def test_export_preserves_raw_values_in_metadata_order(window_files):
    data_path, metadata_path, output_path, frame, metadata = window_files
    assert export_initial_window(data_path, metadata_path, output_path) == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    expected = frame.tail(60)[metadata["ohlcvColumns"]].to_numpy(dtype=np.float32)
    assert (payload["sequenceLength"], payload["featureSize"]) == (60, 55)
    assert payload["ohlcvColumns"] == metadata["ohlcvColumns"]
    assert payload["dates"] == frame.tail(60)["date"].dt.strftime("%Y-%m-%d").tolist()
    np.testing.assert_array_equal(np.asarray(payload["values"]).reshape(60, 55), expected)


@pytest.mark.parametrize(
    "problem", ["short", "duplicate_date", "nan", "inf", "zero", "high", "low", "missing_column"]
)
def test_export_rejects_invalid_window_without_writing(window_files, problem):
    data_path, metadata_path, output_path, frame, metadata = window_files
    columns = metadata["ohlcvColumns"]
    if problem == "short":
        frame = frame.tail(59)
    elif problem == "duplicate_date":
        frame.loc[61, "date"] = frame.loc[60, "date"]
    elif problem == "missing_column":
        frame = frame.drop(columns=columns[0])
    else:
        column, value = {
            "nan": (columns[0], np.nan),
            "inf": (columns[0], np.inf),
            "zero": (columns[4], 0),
            "high": (columns[1], 1),
            "low": (columns[2], 1e9),
        }[problem]
        frame.loc[61, column] = value
    frame.to_parquet(data_path, index=False)
    with pytest.raises(ValueError):
        export_initial_window(data_path, metadata_path, output_path)
    assert not output_path.exists()


def test_export_rejects_inconsistent_input_shape(window_files):
    data_path, metadata_path, output_path, _, metadata = window_files
    metadata["onnx"]["inputShape"] = [1, 60, 215]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        export_initial_window(data_path, metadata_path, output_path)
    assert not output_path.exists()


def test_cli_defaults_to_model_metadata_and_processed_directory(window_files):
    data_path = window_files[0]
    config_path = data_path.parent / "config.yaml"
    config_path.write_text(
        f"paths:\n  sector_data: {data_path}\n  checkpoint: {data_path.parent / 'model.pt'}\n",
        encoding="utf-8",
    )
    main(["--config", str(config_path)])
    output = json.loads(data_path.with_name("initial_ohlcv_window.json").read_text())
    assert len(output["values"]) == 3300
