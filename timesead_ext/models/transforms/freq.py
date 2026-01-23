from __future__ import annotations

import math
from typing import Dict, List, Any

import torch
import torch.nn as nn

from .base import Transform


def freq_cfg(
    k_freq: int = 1,
    mode: str = "channel",
    freq_block: int = 8,
    init_identity: bool = True,
    eps: float = 1e-5,
) -> Dict[str, Any]:
    return {
        "k_freq": k_freq,
        "mode": mode,
        "freq_block": freq_block,
        "init_identity": init_identity,
        "eps": eps,
    }


def _cayley(skew: torch.Tensor, eps: float) -> torch.Tensor:
    eye = torch.eye(skew.shape[-1], device=skew.device, dtype=skew.dtype)
    eye = eye.expand(skew.shape[:-2] + eye.shape)
    a = eye + skew
    b = eye - skew
    if eps > 0:
        b = b + eps * eye
    return torch.linalg.solve(b, a)


class FreqTransform(Transform):
    def __init__(self, channels: int, seq_len: int, cfg: Dict[str, Any]):
        super().__init__()
        self.channels = channels
        self.seq_len = int(seq_len)
        self.mode = str(cfg.get("mode", "channel")).lower()
        self.freq_block = int(cfg.get("freq_block", 8))
        self.init_identity = bool(cfg.get("init_identity", True))
        self.eps = float(cfg.get("eps", 1e-5))

        if self.seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if self.freq_block < 1:
            raise ValueError("freq_block must be >= 1")
        if self.mode not in {"channel", "freq"}:
            raise ValueError("mode must be 'channel' or 'freq'")

        self.raw_skew: nn.Parameter | None = None
        self._freq_bins: int | None = None
        freq_bins = self._freq_bins_from_seq_len(self.seq_len)
        self._init_params(freq_bins)

    @staticmethod
    def _freq_bins_from_seq_len(seq_len: int) -> int:
        return seq_len // 2 + 1

    def _init_params(self, freq_bins: int) -> None:
        if self.mode == "channel":
            shape = (freq_bins, self.channels, self.channels)
        else:
            block_count = math.ceil(freq_bins / self.freq_block)
            shape = (self.channels, block_count, self.freq_block, self.freq_block)

        if self.init_identity:
            param = torch.zeros(shape)
        else:
            param = 0.01 * torch.randn(shape)

        self.raw_skew = nn.Parameter(param)
        self._freq_bins = freq_bins

    def _mix_channels(self, x_freq: torch.Tensor) -> torch.Tensor:
        assert self.raw_skew is not None
        skew = self.raw_skew - self.raw_skew.transpose(-1, -2)
        ortho = _cayley(skew, self.eps).to(dtype=x_freq.dtype)
        freq_t = x_freq.permute(0, 2, 1)
        mixed = torch.einsum("fij,bfj->bfi", ortho, freq_t)
        return mixed.permute(0, 2, 1)

    def _mix_freq_blocks(self, x_freq: torch.Tensor) -> torch.Tensor:
        assert self.raw_skew is not None
        skew = self.raw_skew - self.raw_skew.transpose(-1, -2)
        ortho = _cayley(skew, self.eps).to(dtype=x_freq.dtype)

        batch, channels, freq_bins = x_freq.shape
        block_count = ortho.shape[1]
        output = x_freq.clone()

        for block_idx in range(block_count):
            start = block_idx * self.freq_block
            if start >= freq_bins:
                break
            end = min(start + self.freq_block, freq_bins)
            block = x_freq[:, :, start:end]
            if end - start < self.freq_block:
                pad = self.freq_block - (end - start)
                block = torch.nn.functional.pad(block, (0, pad))
            mixed = torch.einsum("cij,bcj->bci", ortho[:, block_idx], block)
            output[:, :, start:end] = mixed[:, :, : end - start]

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = torch.fft.rfft(x.contiguous())
        torch._check(
            freq.shape[-1] == self._freq_bins,
            lambda: f"Input frequency bins changed. Recreate the transform for a new input length.",
        )
        if self.mode == "channel":
            mixed = self._mix_channels(freq)
        else:
            mixed = self._mix_freq_blocks(freq)

        return torch.fft.irfft(mixed.contiguous(), n=x.shape[-1])


def make_freq_family(channels: int, seq_len: int, cfg: Dict[str, Any]) -> List[FreqTransform]:
    k_freq = int(cfg.get("k_freq", 1))
    if k_freq < 1:
        raise ValueError("k_freq must be >= 1")
    return [FreqTransform(channels, seq_len, cfg) for _ in range(k_freq)]
