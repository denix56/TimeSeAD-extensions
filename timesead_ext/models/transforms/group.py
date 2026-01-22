"""Learnable group-wise temporal transforms."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _softplus_inverse(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value))


def _make_base_grid(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, length, device=device, dtype=dtype).unsqueeze(0)


def _grid_sample_1d(x: torch.Tensor, grid_t: torch.Tensor) -> torch.Tensor:
    x_2d = x.unsqueeze(-1)
    n, _, length = x.shape
    grid = torch.zeros((n, length, 1, 2), device=x.device, dtype=x.dtype)
    grid[..., 1] = grid_t.unsqueeze(-1)
    return F.grid_sample(x_2d, grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze(-1)


def _apply_scale_bias(
    x: torch.Tensor,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    if scale is not None:
        x = x * scale
    if bias is not None:
        x = x + bias
    return x


class GroupTransform(nn.Module):
    """Apply differentiable group-wise time shift, scale/bias, and monotone warp."""

    def __init__(self, group_cfg: Dict):
        super().__init__()
        self.k_group = int(group_cfg.get("k_group", 1))
        self.enable_shift = bool(group_cfg.get("enable_shift", True))
        self.enable_scale_bias = bool(group_cfg.get("enable_scale_bias", True))
        self.enable_warp = bool(group_cfg.get("enable_warp", True))

        self.warp_knots = int(group_cfg.get("warp_knots", 8))
        self.max_shift_frac = float(group_cfg.get("max_shift_frac", 0.0))
        self.scale_range = float(group_cfg.get("scale_range", 0.0))
        self.bias_range = float(group_cfg.get("bias_range", 0.0))
        self.warp_strength = float(group_cfg.get("warp_strength", 0.0))

        if self.enable_shift:
            self.shift = nn.Parameter(torch.zeros(self.k_group))
        else:
            self.register_parameter("shift", None)

        if self.enable_scale_bias:
            init_scale = _softplus_inverse(torch.tensor(1.0))
            self.scale_raw = nn.Parameter(init_scale.repeat(self.k_group))
            self.bias_raw = nn.Parameter(torch.zeros(self.k_group))
        else:
            self.register_parameter("scale_raw", None)
            self.register_parameter("bias_raw", None)

        if self.enable_warp:
            init_knots = _softplus_inverse(torch.ones(self.warp_knots))
            self.warp_raw = nn.Parameter(init_knots.repeat(self.k_group, 1))
        else:
            self.register_parameter("warp_raw", None)

    def _expand_group_params(self, params: torch.Tensor, batch_size: int) -> torch.Tensor:
        return params.repeat(batch_size)

    def _get_scale_bias(self, batch_size: int, channels: int, length: int) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.enable_scale_bias:
            return None, None

        scale = F.softplus(self.scale_raw)
        scale = 1.0 + self.scale_range * (scale - 1.0)
        bias = self.bias_range * torch.tanh(self.bias_raw)

        scale = self._expand_group_params(scale, batch_size).view(batch_size * self.k_group, 1, 1)
        bias = self._expand_group_params(bias, batch_size).view(batch_size * self.k_group, 1, 1)
        scale = scale.expand(-1, channels, length)
        bias = bias.expand(-1, channels, length)
        return scale, bias

    def _get_shift_grid(self, batch_size: int, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        base = _make_base_grid(length, device, dtype)
        shift = self._expand_group_params(self.shift, batch_size)
        shift = torch.tanh(shift) * self.max_shift_frac
        shift = shift * 2.0
        return base + shift.unsqueeze(-1)

    def _get_warp_grid(self, batch_size: int, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        knots = F.softplus(self.warp_raw)
        knots = torch.cumsum(knots, dim=-1)
        knots = knots / knots[..., -1:].clamp_min(1e-6)

        knots = knots.unsqueeze(1)
        warp = F.interpolate(knots, size=length, mode="linear", align_corners=True).squeeze(1)

        identity = torch.linspace(0.0, 1.0, length, device=device, dtype=dtype).unsqueeze(0)
        warp = identity + self.warp_strength * (warp - identity)
        warp = warp.clamp(0.0, 1.0)
        warp = self._expand_group_params(warp, batch_size)
        return warp.view(batch_size * self.k_group, length) * 2.0 - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, length).

        Returns:
            Tensor of shape (batch, k_group, channels, length).
        """
        batch_size, channels, length = x.shape
        x_grouped = x.unsqueeze(1).expand(batch_size, self.k_group, channels, length)
        x_grouped = x_grouped.reshape(batch_size * self.k_group, channels, length)

        if self.enable_shift and self.max_shift_frac > 0:
            grid_t = self._get_shift_grid(batch_size, length, x.device, x.dtype)
            x_grouped = _grid_sample_1d(x_grouped, grid_t)

        if self.enable_warp and self.warp_strength > 0:
            grid_t = self._get_warp_grid(batch_size, length, x.device, x.dtype)
            x_grouped = _grid_sample_1d(x_grouped, grid_t)

        scale, bias = self._get_scale_bias(batch_size, channels, length)
        x_grouped = _apply_scale_bias(x_grouped, scale, bias)

        return x_grouped.view(batch_size, self.k_group, channels, length)
