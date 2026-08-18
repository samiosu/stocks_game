"""A shared probabilistic GRU for all eleven sectors."""

from __future__ import annotations

import math
from typing import Any

try:  # Keep feature/universe CLIs usable before torch is installed.
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - project installation supplies torch
    torch = None
    nn = None
    F = None


SECTOR_COUNT = 11


if nn is not None:

    class MarketGRU(nn.Module):
        """GRU whose output is ``[batch, sectors, (mu, log_sigma)]``.

        ``factor_mode`` adds a learned sector loading to a shared market factor
        and a shared rotation factor. The direct head remains present so the
        model can represent sector-specific effects and can later be replaced
        with a richer factor decoder without changing the ONNX interface.
        """

        def __init__(
            self,
            feature_size: int,
            *,
            hidden_size: int = 128,
            num_layers: int = 1,
            dropout: float = 0.0,
            bidirectional: bool = False,
            sector_count: int = SECTOR_COUNT,
            factor_mode: bool = True,
            mu_scale: float = 0.002,
            factor_loadings: list[list[float]] | None = None,
            factor_return_scales: list[float] | tuple[float, ...] = (0.006, 0.003, 0.002),
            common_noise_weight: float = 0.0,
            max_sigma: float = 0.04,
        ) -> None:
            super().__init__()
            if bidirectional:
                raise ValueError("bidirectional GRU is intentionally unsupported for Unity compatibility")
            if num_layers < 1:
                raise ValueError("num_layers must be positive")
            self.feature_size = int(feature_size)
            self.hidden_size = int(hidden_size)
            self.num_layers = int(num_layers)
            self.sector_count = int(sector_count)
            self.factor_mode = bool(factor_mode)
            if mu_scale <= 0:
                raise ValueError("mu_scale must be positive")
            self.mu_scale = float(mu_scale)
            if max_sigma <= 1e-5:
                raise ValueError("max_sigma must be greater than the sigma floor")
            if not 0.0 <= common_noise_weight <= 1.0:
                raise ValueError("common_noise_weight must be between 0 and 1")
            scales = torch.as_tensor(factor_return_scales, dtype=torch.float32)
            if scales.shape != (3,) or not torch.isfinite(scales).all() or (scales < 0).any():
                raise ValueError("factor_return_scales must contain three non-negative values")
            if factor_loadings is None:
                default_loadings = torch.ones((self.sector_count, 3), dtype=torch.float32)
                default_loadings[:, 1] = torch.linspace(-1.0, 1.0, self.sector_count)
                default_loadings[:, 2] = torch.linspace(1.0, -1.0, self.sector_count)
                factor_loadings_tensor = default_loadings
            else:
                factor_loadings_tensor = torch.as_tensor(factor_loadings, dtype=torch.float32)
            if factor_loadings_tensor.shape != (self.sector_count, 3):
                raise ValueError(f"factor_loadings must have shape [{self.sector_count}, 3]")
            if not torch.isfinite(factor_loadings_tensor).all():
                raise ValueError("factor_loadings must be finite")
            factor_loadings_tensor = factor_loadings_tensor / factor_loadings_tensor.norm(dim=1, keepdim=True).clamp_min(1e-8)
            self.register_buffer("factor_loadings", factor_loadings_tensor)
            self.register_buffer("factor_return_scales", scales)
            self.common_noise_weight = float(common_noise_weight)
            self.max_sigma = float(max_sigma)
            self.max_log_sigma = float(math.log(math.expm1(self.max_sigma - 1e-5)))
            self.gru = nn.GRU(
                input_size=self.feature_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=False,
                batch_first=True,
            )
            self.mu_head = nn.Linear(self.hidden_size, self.sector_count)
            self.log_sigma_head = nn.Linear(self.hidden_size, self.sector_count)
            self.factor_head = nn.Linear(self.hidden_size, 3) if self.factor_mode else None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 3 or x.shape[-1] != self.feature_size:
                raise ValueError(f"Expected [batch, sequence, {self.feature_size}], got {tuple(x.shape)}")
            output, _ = self.gru(x)
            hidden = output[:, -1, :]
            mu = self.mu_scale * F.softsign(self.mu_head(hidden))
            if self.factor_mode and self.factor_head is not None:
                # The GRU predicts a smooth market/rotation/volatility regime.
                # Sector returns are then decoded through fixed, data-estimated
                # loadings instead of eleven independent directional logits.
                factors = torch.tanh(self.factor_head(hidden))
                mu = mu + (
                    factors.unsqueeze(1)
                    * self.factor_loadings.unsqueeze(0)
                    * self.factor_return_scales.view(1, 1, 3)
                ).sum(dim=-1)
            raw_log_sigma = self.log_sigma_head(hidden)
            if self.factor_mode and self.factor_head is not None:
                raw_log_sigma = raw_log_sigma + 0.35 * factors[:, 2:3]
            # Smoothly cap sigma at a realistic daily upper bound.  Unlike a
            # return clip, this does not create repeated sampled values.
            log_sigma = self.max_log_sigma - F.softplus(self.max_log_sigma - raw_log_sigma)
            return torch.stack((mu, log_sigma), dim=-1)


else:

    class MarketGRU:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for MarketGRU")


def require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for model training, generation, and ONNX export")


def sigma_from_log_sigma(log_sigma: torch.Tensor, floor: float = 1e-5) -> torch.Tensor:
    """Map the unconstrained scale head to a positive standard deviation."""

    require_torch()
    return F.softplus(log_sigma) + float(floor)


def probabilistic_loss(
    params: torch.Tensor,
    target_returns: torch.Tensor,
    target_log_volatility: torch.Tensor | None = None,
    *,
    volatility_loss_weight: float = 0.1,
    mean_loss_weight: float = 0.25,
) -> torch.Tensor:
    """Stable Gaussian likelihood plus optional mean and volatility targets."""

    require_torch()
    mu = params[..., 0]
    sigma = sigma_from_log_sigma(params[..., 1])
    log_sigma = torch.log(sigma)
    z = (target_returns - mu) / sigma
    nll = 0.5 * (z.square() + 2.0 * log_sigma)
    loss = nll.mean()
    if mean_loss_weight > 0:
        loss = loss + float(mean_loss_weight) * F.mse_loss(mu, target_returns)
    if target_log_volatility is not None and volatility_loss_weight > 0:
        loss = loss + float(volatility_loss_weight) * F.mse_loss(log_sigma, target_log_volatility)
    return loss


def sample_returns(
    params: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    volatility_scale: float = 1.0,
    epsilon: torch.Tensor | None = None,
    factor_loadings: torch.Tensor | None = None,
    common_noise_weight: float = 0.0,
    factor_epsilon: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample sector returns from model parameters."""

    require_torch()
    if epsilon is None:
        epsilon = torch.randn(params.shape[:-1], generator=generator, device=params.device, dtype=params.dtype)
    noise = epsilon
    if factor_loadings is not None and common_noise_weight > 0:
        loadings = torch.as_tensor(factor_loadings, device=params.device, dtype=params.dtype)
        if loadings.ndim != 2 or loadings.shape[0] != params.shape[-2]:
            raise ValueError("factor_loadings must have shape [sectors, factors]")
        loadings = loadings / loadings.norm(dim=1, keepdim=True).clamp_min(1e-8)
        factor_shape = (*params.shape[:-2], loadings.shape[1])
        if factor_epsilon is None:
            factor_epsilon = torch.randn(factor_shape, generator=generator, device=params.device, dtype=params.dtype)
        if tuple(factor_epsilon.shape) != factor_shape:
            raise ValueError(f"factor_epsilon must have shape {factor_shape}")
        common = torch.einsum("...k,sk->...s", factor_epsilon, loadings)
        weight = float(common_noise_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("common_noise_weight must be between 0 and 1")
        noise = weight * common + math.sqrt(max(0.0, 1.0 - weight * weight)) * epsilon
    return params[..., 0] + sigma_from_log_sigma(params[..., 1]) * float(volatility_scale) * noise


def model_hyperparameters(
    config: dict[str, Any],
    feature_size: int,
    *,
    factor_loadings: list[list[float]] | None = None,
    common_noise_weight: float | None = None,
) -> dict[str, Any]:
    model_config = config.get("model", {})
    return {
        "feature_size": feature_size,
        "hidden_size": int(model_config.get("hidden_size", 128)),
        "num_layers": int(model_config.get("num_layers", 1)),
        "dropout": float(model_config.get("dropout", 0.0)),
        "bidirectional": bool(model_config.get("bidirectional", False)),
        "factor_mode": bool(model_config.get("factor_mode", True)),
        "mu_scale": float(model_config.get("mu_scale", 0.002)),
        "factor_loadings": factor_loadings,
        "factor_return_scales": list(model_config.get("factor_return_scales", [0.006, 0.003, 0.002])),
        "common_noise_weight": float(
            model_config.get("common_noise_weight", 0.0)
            if common_noise_weight is None
            else common_noise_weight
        ),
        "max_sigma": float(model_config.get("max_sigma", 0.04)),
        "sector_count": SECTOR_COUNT,
    }
