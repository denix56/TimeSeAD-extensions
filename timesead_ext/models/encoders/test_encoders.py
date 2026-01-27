import torch

from timesead_ext.models.encoders import ITransformerEncoder, PatchTSTEncoder


def _assert_forward_and_backward(encoder: torch.nn.Module, batch: int, channels: int, length: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(batch, channels, length, requires_grad=True)

    y = encoder(x)

    assert y.shape == (batch, encoder.d_model)

    loss = y.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    param_grads = [p.grad for p in encoder.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in param_grads)


def test_patchtst_encoder_forward_backward() -> None:
    batch, channels, length = 2, 3, 12
    encoder = PatchTSTEncoder(
        seq_len=length,
        patch_len=4,
        patch_stride=4,
        d_model=16,
        num_heads=2,
        num_layers=1,
        d_ff=32,
        dropout=0.0,
        pooling="mean",
    )

    _assert_forward_and_backward(encoder, batch, channels, length)


def test_itransformer_encoder_forward_backward() -> None:
    batch, channels, length = 2, 4, 10
    encoder = ITransformerEncoder(
        seq_len=length,
        d_model=24,
        num_heads=2,
        num_layers=1,
        d_ff=48,
        dropout=0.0,
        pooling="mean",
    )

    _assert_forward_and_backward(encoder, batch, channels, length)
