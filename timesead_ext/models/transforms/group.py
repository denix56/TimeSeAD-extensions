from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Transform


def group_cfg(
    k_group: int = 1,
    enable_shift: bool = True,
    enable_scale_bias: bool = True,
    enable_warp: bool = True,
    warp_knots: int = 8,
    max_shift_frac: float = 0.1,
    scale_range: float | Tuple[float, float] = 0.2,
    bias_range: float | Tuple[float, float] = 0.1,
    warp_strength: float = 1.0,
) -> Dict[str, object]:
    return {
        "k_group": k_group,
        "enable_shift": enable_shift,
        "enable_scale_bias": enable_scale_bias,
        "enable_warp": enable_warp,
        "warp_knots": warp_knots,
        "max_shift_frac": max_shift_frac,
        "scale_range": scale_range,
        "bias_range": bias_range,
        "warp_strength": warp_strength,
    }


def _inverse_softplus(value: float) -> float:
    return float(torch.log(torch.expm1(torch.tensor(value))).item())


def _range_bounds(range_value: float | Iterable[float], center: float) -> Tuple[float, float]:
    if isinstance(range_value, (tuple, list)):
        low, high = range_value
        return float(low), float(high)
    span = float(range_value)
    if center == 1.0:
        if span <= 1.0:
            return max(1.0 - span, 1e-4), 1.0 + span
        return 1.0 / span, span
    return center - span, center + span


class GroupTransform(Transform):
    def __init__(self, channels: int, cfg: Dict[str, object]):
        super().__init__()
        self.channels = channels
        self.enable_shift = bool(cfg.get("enable_shift", True))
        self.enable_scale_bias = bool(cfg.get("enable_scale_bias", True))
        self.enable_warp = bool(cfg.get("enable_warp", True))
        self.warp_knots = int(cfg.get("warp_knots", 8))
        self.max_shift_frac = float(cfg.get("max_shift_frac", 0.1))
        self.scale_range = cfg.get("scale_range", 0.2)
        self.bias_range = cfg.get("bias_range", 0.1)
        self.warp_strength = float(cfg.get("warp_strength", 1.0))

        if self.enable_shift:
            self.raw_shift = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("raw_shift", None)

        if self.enable_scale_bias:
            self.raw_scale = nn.Parameter(torch.zeros(channels))
            self.raw_bias = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter("raw_scale", None)
            self.register_parameter("raw_bias", None)

        if self.enable_warp:
            if self.warp_knots < 2:
                raise ValueError("warp_knots must be >= 2")
            knot_init = _inverse_softplus(1.0)
            self.raw_warp = nn.Parameter(torch.full((self.warp_knots,), knot_init))
        else:
            self.register_parameter("raw_warp", None)

        self._scale_offset = _inverse_softplus(1.0)

    def _base_coords(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.linspace(-1.0, 1.0, length, device=device, dtype=dtype)

    def _warp_coords(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        base = self._base_coords(length, device, dtype)
        if not self.enable_warp:
            return base
        deltas = F.softplus(self.raw_warp)
        cumulative = torch.cumsum(deltas, dim=0)
        cumulative = (cumulative - cumulative[0]) / (cumulative[-1] - cumulative[0])
        warp = cumulative * 2.0 - 1.0
        warp = warp.view(1, 1, -1)
        warp = F.interpolate(warp, size=length, mode="linear", align_corners=True)
        warp = warp.view(-1)
        strength = max(0.0, min(self.warp_strength, 1.0))
        return base + strength * (warp - base)

    def _shift_coords(self, coords: torch.Tensor) -> torch.Tensor:
        if not self.enable_shift:
            return coords
        shift = self.max_shift_frac * torch.tanh(self.raw_shift)[0]
        return coords + shift * 2.0

    def _apply_time_transform(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.enable_shift or self.enable_warp):
            return x
        length = x.shape[-1]
        coords = self._warp_coords(length, x.device, x.dtype)
        coords = self._shift_coords(coords)
        grid = torch.zeros((x.shape[0], 1, length, 2), device=x.device, dtype=x.dtype)
        grid[..., 0] = coords.view(1, 1, -1).expand(x.shape[0], 1, -1)
        return F.grid_sample(
            x.unsqueeze(2),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(2)

    def _apply_scale_bias(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enable_scale_bias:
            return x
        scale = F.softplus(self.raw_scale + self._scale_offset)
        min_scale, max_scale = _range_bounds(self.scale_range, 1.0)
        if min_scale != max_scale:
            scale = min_scale + (max_scale - min_scale) * torch.sigmoid(scale - 1.0)
        bias_min, bias_max = _range_bounds(self.bias_range, 0.0)
        bias = bias_min + (bias_max - bias_min) * torch.sigmoid(self.raw_bias)
        return x * scale.view(1, -1, 1) + bias.view(1, -1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_time_transform(x)
        return self._apply_scale_bias(x)


def make_group_family(channels: int, cfg: Dict[str, object]) -> List[GroupTransform]:
    k_group = int(cfg.get("k_group", 1))
    if k_group < 1:
        raise ValueError("k_group must be >= 1")
    return [GroupTransform(channels, cfg) for _ in range(k_group)]
