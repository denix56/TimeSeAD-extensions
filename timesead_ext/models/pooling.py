from __future__ import annotations

import math

import torch
from torch import nn


class MeanPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class MeanMaxPool(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return self.proj(pooled)


class AttnPool(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(x, self.query) * self.scale
        weights = torch.softmax(scores, dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)
