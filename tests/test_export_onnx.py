import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from stock_sim.export_onnx import export_onnx_model
from stock_sim.constants import OHLCV_FIELDS, SECTOR_IDS
from stock_sim.model import MarketLSTM
from stock_sim.ohlcv import relative_column_names
from stock_sim.preprocessing import FeatureScaler


def test_exported_onnx_can_be_loaded(tmp_path: Path):
    feature_count = len(SECTOR_IDS) * len(OHLCV_FIELDS)
    ohlcv_columns = [f"{sector_id}__{field}" for sector_id in SECTOR_IDS for field in OHLCV_FIELDS]
    relative_columns = relative_column_names()
    model = MarketLSTM(feature_size=feature_count, output_size=feature_count, hidden_size=8)
    checkpoint = {
        "model_type": "lstm_relative_ohlcv",
        "input_type": "ohlcv",
        "target_type": "relative_ohlcv",
        "state_dict": model.state_dict(),
        "model_kwargs": {"feature_size": feature_count, "output_size": feature_count, "hidden_size": 8, "sector_count": 11},
        "sequence_length": 60,
        "volume_lookback": 20,
        "ohlcv_columns": ohlcv_columns,
        "relative_columns": relative_columns,
        "relative_scaler_metadata": {
            "columns": relative_columns,
            "mean": [0.0] * feature_count,
            "scale": [1.0] * feature_count,
        },
        "relative_noise_scale": [0.5] * len(OHLCV_FIELDS),
        "generation_config": {
            "relative_clip": [0.12, 0.12, 0.08, 0.08, 1.0],
            "volume_stochastic_scale": 0.25,
        },
    }
    checkpoint_path = tmp_path / "model.pt"
    scaler_path = tmp_path / "scaler.pkl"
    output_path = tmp_path / "model.onnx"
    metadata_path = tmp_path / "model.metadata.json"
    torch.save(checkpoint, checkpoint_path)
    FeatureScaler(checkpoint["ohlcv_columns"]).fit(torch.zeros((4, feature_count)).numpy()).save(scaler_path)
    export_onnx_model(checkpoint_path, scaler_path, output_path, metadata_path=metadata_path)
    import onnx

    graph = onnx.load(output_path)
    onnx.checker.check_model(graph)
    operators = {node.op_type for node in graph.graph.node}
    assert "LSTM" in operators
    assert "GRU" not in operators
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["modelType"] == "LSTM_RELATIVE_OHLCV"
    assert metadata["inputType"] == "OHLCV"
    assert metadata["onnx"]["inputShape"] == [1, 60, feature_count]
    assert metadata["onnx"]["outputShape"] == [1, 11, 5]
    assert metadata["ohlcvFields"] == list(OHLCV_FIELDS)
    assert metadata["relativeFields"] == [
        "gap_log_return",
        "body_log_return",
        "upper_wick_log_range",
        "lower_wick_log_range",
        "log_volume_ratio",
    ]
    assert len(metadata["scaler"]["mean"]) == feature_count
    assert "reconstructed" in metadata["outputSemantics"]["output"]
