import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from stock_sim.export_onnx import export_onnx_model
from stock_sim.constants import OHLCV_FIELDS, SECTOR_IDS
from stock_sim.model import MarketLSTM
from stock_sim.preprocessing import FeatureScaler


def test_exported_onnx_can_be_loaded(tmp_path: Path):
    feature_count = len(SECTOR_IDS) * len(OHLCV_FIELDS)
    ohlcv_columns = [f"{sector_id}__{field}" for sector_id in SECTOR_IDS for field in OHLCV_FIELDS]
    model = MarketLSTM(feature_size=feature_count, output_size=feature_count, hidden_size=8)
    checkpoint = {
        "model_type": "lstm_ohlcv",
        "input_type": "ohlcv",
        "state_dict": model.state_dict(),
        "model_kwargs": {"feature_size": feature_count, "output_size": feature_count, "hidden_size": 8, "sector_count": 11},
        "sequence_length": 60,
        "ohlcv_columns": ohlcv_columns,
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
    assert metadata["modelType"] == "LSTM_OHLCV"
    assert metadata["inputType"] == "OHLCV"
    assert metadata["onnx"]["inputShape"] == [1, 60, feature_count]
    assert metadata["onnx"]["outputShape"] == [1, 11, 5]
    assert metadata["ohlcvFields"] == list(OHLCV_FIELDS)
    assert len(metadata["scaler"]["mean"]) == feature_count
    assert "inverseTransform" in metadata["outputSemantics"]
