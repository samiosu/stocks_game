import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from stock_sim.export_onnx import export_onnx_model
from stock_sim.model import MarketGRU
from stock_sim.preprocessing import FeatureScaler


def test_exported_onnx_can_be_loaded(tmp_path: Path):
    feature_count = 7
    model = MarketGRU(feature_size=feature_count, hidden_size=8, factor_mode=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_kwargs": {"feature_size": feature_count, "hidden_size": 8, "factor_mode": True, "sector_count": 11},
        "sequence_length": 60,
        "feature_columns": [f"f{i}" for i in range(feature_count)],
    }
    checkpoint_path = tmp_path / "model.pt"
    scaler_path = tmp_path / "scaler.pkl"
    output_path = tmp_path / "model.onnx"
    metadata_path = tmp_path / "model.metadata.json"
    torch.save(checkpoint, checkpoint_path)
    FeatureScaler(checkpoint["feature_columns"]).fit(torch.zeros((4, feature_count)).numpy()).save(scaler_path)
    export_onnx_model(checkpoint_path, scaler_path, output_path, metadata_path=metadata_path)
    import onnx

    onnx.checker.check_model(onnx.load(output_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["onnx"]["inputShape"] == [1, 60, feature_count]
    assert metadata["onnx"]["outputShape"] == [1, 11, 2]
    assert len(metadata["scaler"]["mean"]) == feature_count
    assert metadata["outputSemantics"]["sigmaFormula"] == "softplus(logSigma) + 1e-5"
