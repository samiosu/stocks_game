import pytest

torch = pytest.importorskip("torch")

from stock_sim.model import MarketLSTM, ohlcv_loss


def test_lstm_direct_ohlcv_output_shape_and_loss():
    model = MarketLSTM(feature_size=55, output_size=55, ohlcv_features=5, hidden_size=16)
    x = torch.zeros((2, 60, 55))
    prediction = model(x)
    assert tuple(prediction.shape) == (2, 11, 5)
    target = torch.zeros((2, 11, 5))
    loss = ohlcv_loss(prediction, target)
    assert torch.isfinite(loss)
