"""Checkpoint loading must never execute pickle globals from model files."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from stock_sim.constants import OHLCV_FIELDS, SECTOR_IDS
from stock_sim.generate import _load_checkpoint
from stock_sim.model import MarketLSTM
from stock_sim.ohlcv import relative_column_names


def test_loads_a_valid_tensor_checkpoint(tmp_path: Path):
    feature_count = len(SECTOR_IDS) * len(OHLCV_FIELDS)
    kwargs = {"feature_size": feature_count, "output_size": feature_count, "hidden_size": 8}
    checkpoint = {
        "model_type": "lstm_relative_ohlcv",
        "input_type": "ohlcv",
        "target_type": "relative_ohlcv",
        "ohlcv_columns": [f"{sector}__{field}" for sector in SECTOR_IDS for field in OHLCV_FIELDS],
        "relative_columns": relative_column_names(),
        "model_kwargs": kwargs,
        "state_dict": MarketLSTM(**kwargs).state_dict(),
    }
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    model, loaded = _load_checkpoint(path, "cpu")
    assert isinstance(model, MarketLSTM)
    assert loaded["model_type"] == checkpoint["model_type"]


def test_rejects_checkpoint_with_executable_pickle_payload(tmp_path: Path):
    marker = tmp_path / "executed"

    class Malicious:
        def __reduce__(self):
            return (eval, ("__import__('pathlib').Path(" + repr(str(marker)) + ").touch()",))

    path = tmp_path / "untrusted.pt"
    torch.save({"payload": Malicious()}, path)
    with pytest.raises(Exception, match="Weights only load failed|Unsupported global"):
        _load_checkpoint(path, "cpu")
    assert not marker.exists()


def test_rejects_vulnerable_torch_versions_before_loading(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.9.1")
    with pytest.raises(RuntimeError, match="2.10.0 or newer"):
        _load_checkpoint(tmp_path / "untrusted.pt", "cpu")
