from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import TransformerBlock


def _pool_tokens(x: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "mean":
        return x.mean(dim=1)
    if pooling == "max":
        return x.max(dim=1).values
    if pooling == "first":
        return x[:, 0]
    raise ValueError(f"Unknown pooling: {pooling}")


def _pad_or_trim(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    current_len = x.shape[-1]
    if current_len == seq_len:
        return x
    if current_len < seq_len:
        pad = seq_len - current_len
        return F.pad(x, (0, pad))
    return x[..., -seq_len:].contiguous()


class ITransformerEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        norm_first: bool = True,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pooling = pooling

        self.variate_proj = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    norm_first=norm_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = _pad_or_trim(x, self.seq_len)
        tokens = self.variate_proj(x)
        tokens = self.dropout(tokens)
        for block in self.blocks:
            tokens, _ = block(tokens)
        if self.pooling == "base":
            return tokens
        return _pool_tokens(tokens, self.pooling)
