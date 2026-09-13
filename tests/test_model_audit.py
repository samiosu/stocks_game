import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
from stock_sim.constants import SECTOR_IDS, OHLCV_FIELDS
from stock_sim.dataset import make_ohlcv_split_datasets
from stock_sim.evaluate import _real_returns, heldout_starts
from stock_sim.generate import sample_residual_path
from stock_sim.inference import OHLCVInference
from stock_sim.model import MarketLSTM
from stock_sim.ohlcv import relative_column_names
from stock_sim.preprocessing import FeatureScaler


def fixture():
    columns = [f'{s}__{f}' for s in SECTOR_IDS for f in OHLCV_FIELDS]
    base = np.tile([100., 103., 98., 101., 1e6], 11)
    raw = base[None] * np.linspace(.9, 1.1, 80)[:, None]
    scaler = FeatureScaler(columns).fit(raw[:40])
    target = FeatureScaler(relative_column_names()).fit(np.zeros((2,55)))
    target.scale_ = np.tile([.01,.01,.005,.005,.2], 11)
    model = MarketLSTM(55, hidden_size=8, input_mean=scaler.mean_.tolist(), input_scale=scaler.scale_.tolist())
    return raw, scaler, target, model


def test_window_normalization_preserves_prediction_across_absolute_levels():
    raw, scaler, _, model = fixture()
    model.eval()
    shifted = raw.copy().reshape(-1,11,5)
    shifted[..., :4] *= 10
    shifted[..., 4] *= 100
    a = torch.tensor(scaler.transform(raw[-60:])[None], dtype=torch.float32)
    b = torch.tensor(scaler.transform(shifted.reshape(-1,55)[-60:])[None], dtype=torch.float32)
    torch.testing.assert_close(model(a), model(b), rtol=1e-4, atol=1e-5)


def test_rollout_has_gradient_through_predicted_bar():
    raw, scaler, target, model = fixture()
    inference = OHLCVInference(model, scaler, target)
    x = torch.tensor(scaler.transform(raw[-60:])[None], dtype=torch.float32)
    first = model(x)
    first.retain_grad()
    second = model(inference.advance(x, first))
    second.sum().backward()
    assert first.grad is not None and first.grad.abs().sum() > 0


def test_residual_sampling_keeps_market_day_and_block_dependencies():
    bank = np.arange(30.)[:,None,None] * np.ones((30,11,5))
    checkpoint = {'residual_bank': bank.tolist(), 'generation_config': {'residual_block_length': 5}}
    a = sample_residual_path(checkpoint, 100, 42, 1., .25)
    np.testing.assert_array_equal(a, sample_residual_path(checkpoint, 100, 42, 1., .25))
    assert not np.array_equal(a, sample_residual_path(checkpoint,100,43,1.,.25))
    np.testing.assert_array_equal(a[:,0,:4], a[:,10,:4])
    np.testing.assert_allclose(a[:,:,4], a[:,:,0]*.25)
    assert np.all((np.diff(a[0:5,0,0]) % 30) == 1)


def test_splits_purge_targets_crossing_boundaries_and_evaluation_uses_ohlcv():
    raw, scaler, target, _ = fixture()
    frame = pd.DataFrame(raw, columns=scaler.columns)
    frame['date'] = pd.bdate_range('2020-01-01', periods=len(frame))
    train_end = str(frame.date.iloc[39].date())
    validation_end = str(frame.date.iloc[59].date())
    splits = make_ohlcv_split_datasets(frame, ohlcv_columns=scaler.columns, scaler=scaler,
                                      sector_count=11,field_count=5,sequence_length=20,
                                      forecast_steps=5, target_scaler=target,
                                      train_end=train_end,validation_end=validation_end)
    assert len(splits['train']) == 16
    assert len(splits['validation']) == 16
    assert len(splits['test']) == 16
    _, returns = _real_returns(frame)
    np.testing.assert_allclose(returns[1:,0], np.diff(np.log(frame['energy__close'])))
    starts = heldout_starts(frame,20,10,validation_end)
    assert starts[0] == 60 and starts[-1] == 70


def test_onnx_multistep_matches_python_with_same_residual(tmp_path):
    ort = pytest.importorskip('onnxruntime')
    from stock_sim.export_onnx import export_onnx_model
    raw, scaler, target, model = fixture()
    kwargs = dict(feature_size=55,hidden_size=8,input_mean=scaler.mean_.tolist(),input_scale=scaler.scale_.tolist())
    checkpoint = dict(model_type='lstm_relative_ohlcv',input_type='ohlcv',target_type='relative_ohlcv',
                      state_dict=model.state_dict(),model_kwargs=kwargs,sequence_length=60,volume_lookback=20,
                      ohlcv_columns=scaler.columns,relative_columns=target.columns,
                      volume_anchor=[1e6]*11,
                      relative_scaler_metadata=dict(columns=target.columns,mean=target.mean_.tolist(),scale=target.scale_.tolist()))
    torch.save(checkpoint,tmp_path/'model.pt'); scaler.save(tmp_path/'scaler.pkl')
    export_onnx_model(tmp_path/'model.pt',tmp_path/'scaler.pkl',tmp_path/'model.onnx')
    session = ort.InferenceSession(str(tmp_path/'model.stochastic.onnx'),providers=['CPUExecutionProvider'])
    inference = OHLCVInference(model,scaler,target,volume_anchor=[1e6]*11).eval()
    # Price far outside the training scaler's +/-6 range exercises old clipping bug.
    window = (raw[-60:]*10).copy()
    expected_window = window.copy()
    rng = np.random.default_rng(8)
    for _ in range(5):
        noise = rng.normal(0,.1,(1,11,5)).astype('float32')
        x = scaler.transform(window)[None].astype('float32')
        actual = session.run(None, {'features':x,'residual':noise})[0]
        with torch.no_grad():
            expected = inference(torch.tensor(scaler.transform(expected_window)[None],dtype=torch.float32),torch.tensor(noise)).numpy()
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=.01)
        window = np.concatenate([window[1:],actual.reshape(1,55)])
        expected_window = np.concatenate([expected_window[1:],expected.reshape(1,55)])


def test_volume_returns_toward_training_anchor_and_daily_limit():
    raw, scaler, target, model = fixture()
    raw = raw[-60:].copy().reshape(60,11,5)
    raw[..., 4] = 1e7
    inference = OHLCVInference(model, scaler, target, volume_anchor=[1e6]*11).eval()
    x = torch.tensor(scaler.transform(raw.reshape(60,55))[None], dtype=torch.float32)
    prediction = torch.zeros(1,11,5)
    with torch.no_grad():
        for _ in range(400):
            previous = (x[:,-1] * inference.input_scale + inference.input_mean).reshape(1,11,5)
            bar = inference.reconstruct(x, prediction)
            assert (bar[...,4] >= previous[...,4]/3-.1).all()
            assert (bar[...,4] <= previous[...,4]*3+.1).all()
            x = inference.advance(x, prediction)
    assert torch.max(torch.abs(torch.log(bar[...,4]/1e6))) < .1
