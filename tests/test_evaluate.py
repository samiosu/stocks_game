import numpy as np
import pytest

from stock_sim.evaluate import _save_figures


def test_evaluation_figures_only_include_close_path_comparison(tmp_path):
    pytest.importorskip("matplotlib")
    real = np.zeros((30, 11), dtype=float)
    generated = np.full((30, 11), 0.001, dtype=float)

    paths = _save_figures(real, generated, tmp_path)

    assert [path.name for path in paths] == ["real_vs_generated.png"]
    assert (tmp_path / "real_vs_generated.png").exists()
    assert not (tmp_path / "sector_correlation.png").exists()
    assert not (tmp_path / "return_distribution.png").exists()
