import json
import pickle

import numpy as np
import pytest

from stock_sim.constants import OHLCV_FIELDS, SECTOR_IDS
from stock_sim.generate import _load_checkpoint
from stock_sim.migrate_scalers import migrate_scalers
from stock_sim.model import MarketLSTM
from stock_sim.ohlcv import relative_column_names
from stock_sim.preprocessing import FeatureScaler


UNPICKLED = []


def mark_unpickled():
    UNPICKLED.append(True)


class UnsafePayload:
    def __reduce__(self):
        return mark_unpickled, ()


def checkpoint():
    pytest.importorskip("torch")
    columns = [f"{sector}__{field}" for sector in SECTOR_IDS for field in OHLCV_FIELDS]
    relative = relative_column_names()
    kwargs = {"feature_size": len(columns), "output_size": len(columns), "hidden_size": 4}
    model = MarketLSTM(**kwargs)
    return model, {
        "model_type": "lstm_relative_ohlcv", "input_type": "ohlcv",
        "target_type": "relative_ohlcv", "model_kwargs": kwargs,
        "state_dict": model.state_dict(), "ohlcv_columns": columns,
        "relative_columns": relative,
        "scaler_metadata": {"columns": columns, "mean": [2.0] * len(columns), "scale": [3.0] * len(columns)},
        "relative_scaler_metadata": {"columns": relative, "mean": [0.0] * len(relative), "scale": [1.0] * len(relative)},
    }


def test_json_scaler_round_trip_preserves_parameters_and_transforms(tmp_path):
    values = np.array([[1.0, -2.0], [3.0, 4.0], [5.0, 8.0]])
    original = FeatureScaler(["価格", "volume"]).fit(values)
    path = tmp_path / "scaler.json"
    original.save(path)
    assert json.loads(path.read_text())["schema_version"] == 1
    restored = FeatureScaler.load(path)
    assert restored.columns == original.columns
    np.testing.assert_array_equal(restored.mean_, original.mean_)
    np.testing.assert_array_equal(restored.scale_, original.scale_)
    np.testing.assert_allclose(restored.transform(values), original.transform(values))
    np.testing.assert_allclose(restored.inverse_transform(restored.transform(values)), values)


@pytest.mark.parametrize("changes", [
    {"schema_version": 2}, {"schema_version": True}, {"columns": []},
    {"columns": ["a", "a"]}, {"columns": [1]}, {"mean": []},
    {"mean": [[1.0]]}, {"mean": [True]}, {"mean": ["1"]},
    {"mean": [float("nan")]}, {"scale": [float("inf")]},
    {"scale": [0.0]}, {"scale": [-1.0]},
])
def test_invalid_scaler_metadata_is_rejected(tmp_path, changes):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"schema_version": 1, "columns": ["a"], "mean": [0.0], "scale": [1.0], **changes}))
    with pytest.raises(ValueError):
        FeatureScaler.load(path)


@pytest.mark.parametrize("protocol", [0, pickle.HIGHEST_PROTOCOL])
def test_pickle_scaler_is_rejected_without_executing_it(tmp_path, protocol):
    UNPICKLED.clear()
    path = tmp_path / "legacy.pkl"
    path.write_bytes(pickle.dumps(UnsafePayload(), protocol=protocol))
    with pytest.raises(ValueError, match="Scaler must be JSON"):
        FeatureScaler.load(path)
    assert UNPICKLED == []


def test_weights_only_checkpoint_preserves_model_predictions(tmp_path):
    torch = pytest.importorskip("torch")
    original, metadata = checkpoint()
    original.eval()
    path = tmp_path / "model.pt"
    torch.save(metadata, path)
    restored, _ = _load_checkpoint(path, "cpu")
    sample = torch.zeros(1, 3, len(metadata["ohlcv_columns"]))
    with torch.no_grad():
        torch.testing.assert_close(restored(sample), original(sample))


def test_checkpoint_payload_is_rejected_without_executing_it(tmp_path):
    torch = pytest.importorskip("torch")
    UNPICKLED.clear()
    path = tmp_path / "unsafe.pt"
    torch.save({"payload": UnsafePayload()}, path)
    with pytest.raises(pickle.UnpicklingError):
        _load_checkpoint(path, "cpu")
    assert UNPICKLED == []


def test_checkpoint_typeerror_does_not_trigger_an_unsafe_retry(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []

    def unsupported(*args, **kwargs):
        calls.append(kwargs)
        raise TypeError("unsupported checkpoint")

    monkeypatch.setattr(torch, "load", unsupported)
    with pytest.raises(TypeError, match="unsupported checkpoint"):
        _load_checkpoint(tmp_path / "model.pt", "cpu")
    assert len(calls) == 1
    assert calls[0]["weights_only"] is True


def test_migration_recovers_scalers_without_opening_old_pickle_files(tmp_path):
    torch = pytest.importorskip("torch")
    _, metadata = checkpoint()
    torch.save(metadata, tmp_path / "model.pt")
    (tmp_path / "features.pkl").write_bytes(pickle.dumps(UnsafePayload()))
    UNPICKLED.clear()
    config = {"_project_root": str(tmp_path), "paths": {
        "checkpoint": "model.pt", "scaler": "features.json", "relative_scaler": "relative.json",
    }}
    outputs = migrate_scalers(config)
    assert len(outputs) == 2
    np.testing.assert_array_equal(FeatureScaler.load(outputs[0]).mean_, metadata["scaler_metadata"]["mean"])
    assert FeatureScaler.load(outputs[1]).columns == metadata["relative_columns"]
    assert UNPICKLED == []
    before = outputs[0].read_bytes()
    with pytest.raises(FileExistsError):
        migrate_scalers(config)
    assert outputs[0].read_bytes() == before


def test_migration_validates_both_scalers_before_writing(tmp_path):
    torch = pytest.importorskip("torch")
    _, metadata = checkpoint()
    metadata["relative_scaler_metadata"]["scale"][0] = 0.0
    torch.save(metadata, tmp_path / "model.pt")
    config = {"_project_root": str(tmp_path), "paths": {
        "checkpoint": "model.pt", "scaler": "features.json", "relative_scaler": "relative.json",
    }}
    with pytest.raises(ValueError):
        migrate_scalers(config)
    assert not (tmp_path / "features.json").exists()
    assert not (tmp_path / "relative.json").exists()
