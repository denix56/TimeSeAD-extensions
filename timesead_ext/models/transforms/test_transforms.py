import torch

from timesead_ext.models.transforms.freq import FreqTransform, freq_cfg
from timesead_ext.models.transforms.group import GroupTransform, group_cfg
from timesead_ext.models.transforms.invertible import InvertibleFlow, invertible_cfg


def _make_input(batch: int = 2, channels: int = 3, length: int = 16) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, channels, length)


def test_output_shapes_and_gradients() -> None:
    x = _make_input().requires_grad_(True)

    flow_cfg = invertible_cfg(num_flows=2, hidden=8, kernel_size=3, clamp=2.0)
    transforms = [
        GroupTransform(channels=3, cfg=group_cfg()),
        FreqTransform(channels=3, cfg=freq_cfg(mode="channel", freq_block=4)),
        InvertibleFlow(
            channels=3,
            num_flows=int(flow_cfg["num_flows"]),
            hidden=int(flow_cfg["hidden"]),
            kernel_size=int(flow_cfg["kernel_size"]),
            clamp=float(flow_cfg["clamp"]),
        ),
    ]

    for transform in transforms:
        y = transform(x)
        assert y.shape == x.shape
        loss = y.pow(2).mean()
        loss.backward(retain_graph=True)

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        param_grads = [p.grad for p in transform.parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() for g in param_grads)

        x.grad.zero_()
        transform.zero_grad(set_to_none=True)


def test_invertible_round_trip() -> None:
    x = _make_input()
    flow_cfg = invertible_cfg(num_flows=3, hidden=16, kernel_size=3, clamp=2.0)
    flow = InvertibleFlow(
        channels=3,
        num_flows=int(flow_cfg["num_flows"]),
        hidden=int(flow_cfg["hidden"]),
        kernel_size=int(flow_cfg["kernel_size"]),
        clamp=float(flow_cfg["clamp"]),
    )

    y = flow(x)
    x_hat = flow.inverse(y)

    assert torch.allclose(x, x_hat, rtol=1e-4, atol=1e-4)


def test_freq_transform_preserves_energy() -> None:
    x = _make_input(batch=4, channels=3, length=32)
    transform = FreqTransform(channels=3, cfg=freq_cfg(mode="channel", freq_block=4))

    y = transform(x)

    energy_x = x.pow(2).sum(dim=(-1, -2))
    energy_y = y.pow(2).sum(dim=(-1, -2))

    assert torch.allclose(energy_x, energy_y, rtol=1e-3, atol=1e-4)
