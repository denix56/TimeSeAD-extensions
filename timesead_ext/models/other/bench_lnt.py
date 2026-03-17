from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any, Dict

import torch

try:
    from timesead_ext.models.other.lnt import LNT, LNTLoss
except ModuleNotFoundError as exc:
    if exc.name != "timesead_ext":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from timesead_ext.models.other.lnt import LNT, LNTLoss


def _build_model(encoder_type: str, channels: int, length: int) -> LNT:
    common = dict(
        ts_channels=channels,
        seq_len=length,
        n_transformations=6,
        hidden_dim=64,
        latent_dim=32,
        transform_cfg={"hidden_dim": 64, "dropout": 0.0},
    )
    if encoder_type == "patchtst_time":
        return LNT(
            encoder_type=encoder_type,
            encoder_cfg={
                "patch_len": 16,
                "patch_stride": 8,
                "d_model": 64,
                "num_heads": 4,
                "num_layers": 2,
                "d_ff": 128,
                "dropout": 0.0,
            },
            **common,
        )
    if encoder_type == "bosch_cpc":
        if length < 64:
            encoder_cfg = {
                "enc_hidden": 48,
                "gru_hidden": 48,
                "strides": [3, 2],
                "filters": [5, 3],
                "padding": [2, 1],
            }
        else:
            encoder_cfg = {
                "enc_hidden": 48,
                "gru_hidden": 48,
            }
        return LNT(
            encoder_type=encoder_type,
            encoder_cfg=encoder_cfg,
            **common,
        )
    return LNT(encoder_type="cnn", **common)


def _step_benchmark(encoder_type: str, batch: int, channels: int, length: int, steps: int) -> Dict[str, Any]:
    torch.manual_seed(0)
    model = _build_model(encoder_type, channels, length)
    loss_fn = LNTLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(batch, length, channels)

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn((model((x,)),))
        loss.backward()
        optimizer.step()

    timings = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        combined = model((x,))
        loss = loss_fn((combined,))
        loss.backward()
        optimizer.step()
        timings.append(time.perf_counter() - t0)

    return {
        "encoder_type": encoder_type,
        "avg_step_s": sum(timings) / len(timings),
        "output_shape": list(combined.shape),
        "loss": float(loss.detach()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run small CPU LNT benchmarks")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    results = [
        _step_benchmark("cnn", args.batch, args.channels, args.length, args.steps),
        _step_benchmark("patchtst_time", args.batch, args.channels, args.length, args.steps),
        _step_benchmark("bosch_cpc", args.batch, args.channels, args.length, args.steps),
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
