"""A shared LSTM that predicts the next OHLCV bar for all sectors."""

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
OHLCV_FIELD_COUNT = 5


if nn is not None:

    class MarketLSTM(nn.Module):
        """LSTM whose output is the next ``[batch, sectors, OHLCV]`` bar.

        Inputs and targets are standardized raw sector OHLCV values. The
        scaler is kept outside the model so the ONNX graph remains a simple
        LSTM plus a linear output head.
        """

        def __init__(
            self,
            feature_size: int,
            *,
            output_size: int | None = None,
            ohlcv_features: int = OHLCV_FIELD_COUNT,
            hidden_size: int = 128,
            num_layers: int = 1,
            dropout: float = 0.0,
            bidirectional: bool = False,
            sector_count: int = SECTOR_COUNT,
        ) -> None:
            super().__init__()
            if bidirectional:
                raise ValueError("bidirectional LSTM is intentionally unsupported for Unity compatibility")
            if num_layers < 1:
                raise ValueError("num_layers must be positive")
            if ohlcv_features < 1:
                raise ValueError("ohlcv_features must be positive")
            self.feature_size = int(feature_size)
            self.hidden_size = int(hidden_size)
            self.num_layers = int(num_layers)
            self.sector_count = int(sector_count)
            self.ohlcv_features = int(ohlcv_features)
            expected_output_size = self.sector_count * self.ohlcv_features
            self.output_size = expected_output_size if output_size is None else int(output_size)
            if self.output_size != expected_output_size:
                raise ValueError(
                    f"output_size must equal sector_count * ohlcv_features ({expected_output_size})"
                )
            self.lstm = nn.LSTM(
                input_size=self.feature_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=False,
                batch_first=True,
            )
            self.ohlcv_head = nn.Linear(self.hidden_size, self.output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 3 or x.shape[-1] != self.feature_size:
                raise ValueError(f"Expected [batch, sequence, {self.feature_size}], got {tuple(x.shape)}")
            output, _ = self.lstm(x)
            hidden = output[:, -1, :]
            return self.ohlcv_head(hidden).reshape(-1, self.sector_count, self.ohlcv_features)


else:

    class MarketLSTM:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for MarketLSTM")


# Keep the old import name usable for downstream callers. Existing GRU and
# probabilistic-LSTM checkpoints are not compatible with the OHLCV model.
MarketGRU = MarketLSTM


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


def ohlcv_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute a robust loss on standardized next-bar OHLCV values."""

    require_torch()
    if predicted.shape != target.shape:
        raise ValueError(f"predicted and target shapes must match, got {predicted.shape} and {target.shape}")
    if predicted.ndim != 3 or predicted.shape[-1] != OHLCV_FIELD_COUNT:
        raise ValueError("OHLCV tensors must have shape [batch, sectors, 5]")
    return F.smooth_l1_loss(predicted, target)


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
        "output_size": SECTOR_COUNT * OHLCV_FIELD_COUNT,
        "ohlcv_features": OHLCV_FIELD_COUNT,
        "hidden_size": int(model_config.get("hidden_size", 128)),
        "num_layers": int(model_config.get("num_layers", 1)),
        "dropout": float(model_config.get("dropout", 0.0)),
        "bidirectional": bool(model_config.get("bidirectional", False)),
        "sector_count": SECTOR_COUNT,
    }
