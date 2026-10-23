import numpy as np
import pandas as pd
import pytest

from stock_sim.preprocessing import FeatureScaler


def test_scaler_transform_can_replace_integer_feature_columns():
    frame = pd.DataFrame({"signal": [0.0, 1.0, 2.0], "event": [0, 0, 0]})
    scaler = FeatureScaler(["signal", "event"]).fit(frame)

    transformed = scaler.transform(frame)

    assert transformed["event"].dtype.kind == "f"
    assert np.isfinite(transformed.to_numpy(dtype=float)).all()


def test_scaler_round_trip_restores_values():
    values = np.array([[1.0, -2.0], [3.0, 4.0], [5.0, 8.0]])
    scaler = FeatureScaler(["a", "b"]).fit(values)
    transformed = scaler.transform(values)
    restored = scaler.inverse_transform(transformed)
    np.testing.assert_allclose(restored, values)


def test_scaler_json_round_trip_and_rejects_invalid_values(tmp_path):
    scaler = FeatureScaler(["a", "b"]).fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    path = tmp_path / "scaler.json"
    scaler.save(path)
    loaded = FeatureScaler.load(path)
    assert loaded.columns == ["a", "b"]
    np.testing.assert_allclose(loaded.transform(np.array([[2.0, 3.0]])), [[0.0, 0.0]])

    path.write_text('{"version":1,"columns":["a"],"mean":[0],"scale":[0]}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid FeatureScaler values"):
        FeatureScaler.load(path)


def test_scaler_does_not_unpickle_legacy_file(tmp_path):
    import pickle

    class Malicious:
        def __reduce__(self):
            return (eval, ("__import__('pathlib').Path(" + repr(str(tmp_path / "executed")) + ").touch()",))

    path = tmp_path / "scaler.pkl"
    path.write_bytes(pickle.dumps(Malicious()))
    with pytest.raises((UnicodeError, ValueError)):
        FeatureScaler.load(path)
    assert not (tmp_path / "executed").exists()
