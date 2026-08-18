import numpy as np

from stock_sim.factors import estimate_factor_structure


def test_factor_structure_has_three_named_factors_and_normalized_loadings():
    rng = np.random.default_rng(4)
    common = rng.normal(size=(500, 1))
    returns = common @ np.full((1, 11), 0.7) + rng.normal(scale=0.4, size=(500, 11))
    metadata = estimate_factor_structure(returns)

    assert metadata["factor_names"] == ["market", "rotation", "volatility"]
    loadings = np.asarray(metadata["factor_loadings"])
    assert loadings.shape == (11, 3)
    np.testing.assert_allclose(np.linalg.norm(loadings, axis=1), 1.0)
    assert 0.0 <= metadata["common_noise_weight"] <= 0.98
