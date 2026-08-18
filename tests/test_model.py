import pytest

torch = pytest.importorskip("torch")

from stock_sim.model import MarketGRU, probabilistic_loss, sample_returns, sigma_from_log_sigma


def test_gru_output_shape_and_sampling():
    model = MarketGRU(feature_size=7, hidden_size=16, factor_mode=True)
    x = torch.zeros((2, 60, 7))
    params = model(x)
    assert tuple(params.shape) == (2, 11, 2)
    target = torch.zeros((2, 11))
    loss = probabilistic_loss(params, target, torch.zeros((2, 11)))
    assert torch.isfinite(loss)
    generator = torch.Generator().manual_seed(42)
    one = sample_returns(params, generator=generator)
    generator = torch.Generator().manual_seed(42)
    two = sample_returns(params, generator=generator)
    assert torch.equal(one, two)


def test_sigma_parameterization_is_positive_and_continuous():
    values = torch.tensor([-10.0, 0.0, 10.0])
    sigma = sigma_from_log_sigma(values)
    assert torch.all(sigma > 0)
    assert sigma[0] < sigma[1] < sigma[2]


def test_factor_sampling_creates_positive_cross_sector_dependence():
    batch = 3000
    params = torch.zeros((batch, 11, 2))
    params[..., 1] = torch.log(torch.expm1(torch.tensor(0.02)))
    loadings = torch.ones((11, 3))
    sampled = sample_returns(
        params,
        generator=torch.Generator().manual_seed(7),
        factor_loadings=loadings,
        common_noise_weight=0.8,
    )
    correlation = torch.corrcoef(sampled.T)
    off_diagonal = correlation[~torch.eye(11, dtype=torch.bool)]
    assert float(off_diagonal.mean()) > 0.5
