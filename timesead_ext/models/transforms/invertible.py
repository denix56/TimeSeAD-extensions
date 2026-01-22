from __future__ import annotations

from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn

from .base import Transform


def invertible_cfg(
    num_flows: int = 4,
    hidden: int = 32,
    kernel_size: int = 3,
    clamp: float = 2.0,
    share_across_k: bool = False,
    k_invertible: int = 1,
) -> Dict[str, object]:
    return {
        "num_flows": num_flows,
        "hidden": hidden,
        "kernel_size": kernel_size,
        "clamp": clamp,
        "share_across_k": share_across_k,
        "k_invertible": k_invertible,
    }


class AffineCoupling(nn.Module):
    def __init__(self, channels: int, hidden: int, kernel_size: int, clamp: float, swap: bool = False):
        super().__init__()
        self.clamp = clamp
        self.swap = swap
        c1 = channels // 2
        c2 = channels - c1
        self.c1 = c1
        self.c2 = c2
        if swap:
            cond_channels = c2
            target_channels = c1
        else:
            cond_channels = c1
            target_channels = c2
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(cond_channels, hidden, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, target_channels * 2, kernel_size, padding=padding),
        )

    def _split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x.split([self.c1, self.c2], dim=1)
        return x1, x2

    def _condition(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale_shift = self.net(cond)
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = self.clamp * torch.tanh(scale)
        return scale, shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self._split(x)
        if self.swap:
            scale, shift = self._condition(x2)
            y1 = x1 * torch.exp(scale) + shift
            return torch.cat([y1, x2], dim=1)
        scale, shift = self._condition(x1)
        y2 = x2 * torch.exp(scale) + shift
        return torch.cat([x1, y2], dim=1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y1, y2 = self._split(y)
        if self.swap:
            scale, shift = self._condition(y2)
            x1 = (y1 - shift) * torch.exp(-scale)
            return torch.cat([x1, y2], dim=1)
        scale, shift = self._condition(y1)
        x2 = (y2 - shift) * torch.exp(-scale)
        return torch.cat([y1, x2], dim=1)


class InvertibleFlow(Transform):
    def __init__(self, channels: int, num_flows: int, hidden: int, kernel_size: int, clamp: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AffineCoupling(
                    channels=channels,
                    hidden=hidden,
                    kernel_size=kernel_size,
                    clamp=clamp,
                    swap=bool(idx % 2),
                )
                for idx in range(num_flows)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y


def make_invertible_family(channels: int, cfg: Dict[str, Any]) -> List[InvertibleFlow]:
    num_flows = int(cfg["num_flows"])
    hidden = int(cfg["hidden"])
    kernel_size = int(cfg["kernel_size"])
    clamp = float(cfg["clamp"])
    k_invertible = int(cfg.get("k_invertible", 1))
    share_across_k = bool(cfg.get("share_across_k", False))

    if k_invertible < 1:
        raise ValueError("k_invertible must be >= 1")

    if share_across_k:
        flow = InvertibleFlow(channels, num_flows, hidden, kernel_size, clamp)
        return [flow for _ in range(k_invertible)]

    return [InvertibleFlow(channels, num_flows, hidden, kernel_size, clamp) for _ in range(k_invertible)]
