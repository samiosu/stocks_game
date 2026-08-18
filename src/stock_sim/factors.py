"""Small, serializable factor-structure utilities for the sector generator."""

from __future__ import annotations

from typing import Any

import numpy as np

FACTOR_NAMES = ("market", "rotation", "volatility")


def _off_diagonal_mean(matrix: np.ndarray) -> float:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(matrix[mask].mean())


def estimate_factor_structure(
    returns: np.ndarray,
    *,
    factor_names: tuple[str, ...] = FACTOR_NAMES,
) -> dict[str, Any]:
    """Estimate a compact PCA factor structure from sector log returns.

    The returned loading rows are normalized so that they can be used as
    correlation loadings in ``r = mu + sigma * noise``.  The first principal
    component is the broad market factor; the next two capture rotation and
    volatility-regime co-movement.  Their signs are arbitrary statistically,
    so they are made deterministic for reproducible checkpoints.
    """

    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < len(factor_names):
        raise ValueError("returns must be a 2D matrix with one column per sector")
    if len(factor_names) != 3:
        raise ValueError("The generator currently requires market, rotation, and volatility factors")
    if values.shape[0] < 3 or not np.isfinite(values).all():
        raise ValueError("returns must contain at least three finite observations")

    centered = values - values.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False, ddof=0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    factor_count = len(factor_names)
    selected_values = eigenvalues[:factor_count]
    raw_loadings = eigenvectors[:, :factor_count] * np.sqrt(selected_values)[None, :]

    # Eigenvector signs are not identifiable.  Stabilize them so checkpoints
    # produced on different BLAS implementations have the same orientation.
    for index in range(factor_count):
        anchor = int(np.argmax(np.abs(raw_loadings[:, index])))
        if raw_loadings[anchor, index] < 0:
            raw_loadings[:, index] *= -1.0
    row_norm = np.linalg.norm(raw_loadings, axis=1, keepdims=True)
    normalized_loadings = raw_loadings / np.maximum(row_norm, 1e-12)

    target_correlation = np.corrcoef(values, rowvar=False)
    factor_correlation = normalized_loadings @ normalized_loadings.T
    target_mean = _off_diagonal_mean(target_correlation)
    factor_mean = _off_diagonal_mean(factor_correlation)
    if factor_mean > 1e-8 and target_mean > 0:
        common_noise_weight = float(np.sqrt(target_mean / factor_mean))
    else:
        common_noise_weight = 0.75
    common_noise_weight = float(np.clip(common_noise_weight, 0.0, 0.98))

    total_variance = float(eigenvalues.sum())
    explained = selected_values / max(total_variance, 1e-12)
    return {
        "factor_names": list(factor_names),
        "factor_loadings": normalized_loadings.tolist(),
        "raw_factor_loadings": raw_loadings.tolist(),
        "eigenvalues": selected_values.tolist(),
        "explained_variance_ratio": explained.tolist(),
        "target_offdiag_correlation_mean": target_mean,
        "factor_offdiag_correlation_mean": _off_diagonal_mean(factor_correlation),
        "common_noise_weight": common_noise_weight,
    }
