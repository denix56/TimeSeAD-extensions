from __future__ import annotations

import torch
from torch import nn


class LinearHead(nn.Module):
    def __init__(self, d_model: int, proj_dim: int, **kwargs: object) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, proj_dim, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MLPHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        proj_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = d_model if hidden_dim is None else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
