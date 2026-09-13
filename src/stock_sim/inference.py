"""Shared differentiable OHLCV reconstruction for rollout and ONNX."""

import numpy as np
import torch
from torch import nn


class OHLCVInference(nn.Module):
    def __init__(self, model, scaler, target_scaler, volume_lookback=20, relative_clip=None,
                 feature_z_clip=6.0, volume_anchor=None, volume_anchor_strength=0.1,
                 volume_daily_limit=3.0):
        super().__init__()
        self.model = model
        self.volume_lookback = int(volume_lookback)
        self.feature_z_clip = feature_z_clip
        self.volume_anchor_strength = float(volume_anchor_strength) if volume_anchor is not None else 0.0
        self.volume_daily_limit = float(volume_daily_limit)
        if not 0 <= self.volume_anchor_strength <= 1 or not np.isfinite(self.volume_daily_limit) or self.volume_daily_limit < 1:
            raise ValueError("Invalid volume mean-reversion or daily limit")
        anchor = np.ones(11) if volume_anchor is None else np.asarray(volume_anchor)
        if anchor.shape != (11,) or not np.isfinite(anchor).all() or (anchor <= 0).any():
            raise ValueError("Invalid sector volume anchors")
        self.register_buffer("volume_anchor", torch.as_tensor(anchor, dtype=torch.float32))
        limits = np.asarray(relative_clip if relative_clip is not None else [0.12, 0.12, 0.08, 0.08, 1.0])
        if limits.shape != (5,) or not np.isfinite(limits).all() or (limits <= 0).any():
            raise ValueError("relative_clip must contain five finite positive limits")
        for name, value in [("input_mean", scaler.mean_), ("input_scale", scaler.scale_),
                            ("target_mean", target_scaler.mean_), ("target_scale", target_scaler.scale_),
                            ("limits", limits)]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))

    def predict_relative(self, x):
        if not self.model.normalize_window and self.feature_z_clip is not None:
            return self.model(x.clamp(-self.feature_z_clip, self.feature_z_clip))
        return self.model(x)

    def reconstruct(self, x, prediction):
        relative = (prediction.flatten(1) * self.target_scale + self.target_mean).reshape(-1, 11, 5)
        relative = torch.maximum(torch.minimum(relative, self.limits), -self.limits)
        raw = (x * self.input_scale + self.input_mean).reshape(x.shape[0], x.shape[1], 11, 5)
        previous = raw[:, -1].clamp_min(1e-6)
        log_reference = torch.log(raw[:, -self.volume_lookback:, :, 4].clamp_min(1e-6)).mean(dim=1)
        reference = torch.exp((1-self.volume_anchor_strength)*log_reference
                              + self.volume_anchor_strength*torch.log(self.volume_anchor))
        op = previous[..., 3] * torch.exp(relative[..., 0])
        cl = op * torch.exp(relative[..., 1])
        hi = torch.maximum(op, cl) * torch.exp(relative[..., 2].clamp_min(0))
        lo = torch.minimum(op, cl) * torch.exp(-relative[..., 3].clamp_min(0))
        vol = reference * torch.exp(relative[..., 4])
        vol = torch.maximum(torch.minimum(vol, previous[..., 4]*self.volume_daily_limit),
                            previous[..., 4]/self.volume_daily_limit)
        return torch.stack([op, hi, lo, cl, vol], dim=-1)

    def advance(self, x, prediction):
        bar = self.reconstruct(x, prediction).flatten(1)
        next_features = (bar - self.input_mean) / self.input_scale
        return torch.cat([x[:, 1:], next_features[:, None]], dim=1)

    def forward(self, x, residual=None):
        prediction = self.predict_relative(x)
        if residual is not None:
            prediction = prediction + residual
        return self.reconstruct(x, prediction)
