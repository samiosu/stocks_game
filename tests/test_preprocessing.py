import numpy as np
import pandas as pd

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
