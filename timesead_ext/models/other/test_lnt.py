from __future__ import annotations

import torch
import torch.nn.functional as F

from timesead_ext.models.other.lnt import (
    BoschCPCEncoder,
    LNT,
    LNTAnomalyDetector,
    LNTLoss,
    ModernTCNTimeEncoder,
    NeuralTransformationLearner,
    SensorformerTimeEncoder,
    VectorizedTransformationBank,
)


def _assert_forward_and_backward(model: LNT, batch: int, channels: int, length: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(batch, length, channels, requires_grad=True)
    combined = model((x,))

    assert combined.shape == (batch, length, model.n_transformations + 1, model.latent_dim)

    loss = LNTLoss()((combined,))
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    param_grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in param_grads)


def _legacy_lnt_dcl_score(combined: torch.Tensor, temperature: float, eval_mode: bool) -> torch.Tensor:
    batch_size, seq_len, n_all, latent_dim = combined.shape
    combined_flat = combined.reshape(-1, n_all, latent_dim)
    combined_norm = F.normalize(combined_flat, p=2, dim=-1)
    sim = torch.matmul(combined_norm, combined_norm.transpose(-2, -1)) / temperature
    sim_pos = sim[:, 1:, 0]

    diag_mask = torch.eye(n_all, device=combined.device, dtype=torch.bool)
    sim_masked = sim.masked_fill(diag_mask, float("-inf"))
    normalization_const = torch.logsumexp(sim_masked[:, 1:, :], dim=-1)
    loss_per_timestep = (-sim_pos + normalization_const).sum(dim=-1)
    loss_per_sample = loss_per_timestep.reshape(batch_size, seq_len)
    if eval_mode:
        return loss_per_sample.mean(dim=1)
    return loss_per_sample.mean()


def _make_sine_batch(batch: int, length: int, channels: int) -> torch.Tensor:
    time = torch.linspace(0, 4 * torch.pi, steps=length)
    samples = []
    for batch_idx in range(batch):
        channel_series = []
        phase_shift = 0.2 * batch_idx
        for channel_idx in range(channels):
            freq = channel_idx + 1
            channel_series.append(torch.sin(freq * time + phase_shift) + 0.1 * torch.cos((freq + 1) * time))
        samples.append(torch.stack(channel_series, dim=-1))
    return torch.stack(samples, dim=0)


def test_lnt_forward_backward_all_backbones() -> None:
    batch, channels, length = 2, 3, 48
    common = dict(
        ts_channels=channels,
        seq_len=length,
        n_transformations=4,
        hidden_dim=32,
        latent_dim=16,
        transform_cfg={"hidden_dim": 24, "dropout": 0.0},
    )
    models = [
        LNT(encoder_type="cnn", **common),
        LNT(
            encoder_type="modern_tcn",
            encoder_cfg={
                "d_model": 24,
                "num_layers": 2,
                "kernel_sizes": [5, 7],
                "dilations": [1, 2],
                "ff_mult": 2,
                "dropout": 0.0,
            },
            **common,
        ),
        LNT(
            encoder_type="sensorformer_time",
            encoder_cfg={
                "patch_len": 8,
                "patch_stride": 4,
                "d_model": 16,
                "num_heads": 2,
                "num_layers": 1,
                "d_ff": 32,
                "dropout": 0.0,
                "global_patches": 6,
                "attention_chunk_size": None,
            },
            **common,
        ),
        LNT(
            encoder_type="bosch_cpc",
            encoder_cfg={
                "enc_hidden": 16,
                "gru_hidden": 16,
                "strides": [3, 2],
                "filters": [5, 3],
                "padding": [2, 1],
            },
            **common,
        ),
    ]

    for model in models:
        _assert_forward_and_backward(model, batch, channels, length)


def test_modern_tcn_encoder_preserves_sequence_length() -> None:
    encoder = ModernTCNTimeEncoder(
        input_channels=3,
        latent_dim=12,
        d_model=16,
        num_layers=2,
        kernel_sizes=(7, 11),
        dilations=(1, 3),
        ff_mult=2,
        dropout=0.0,
    )

    x = torch.randn(2, 3, 37, requires_grad=True)
    y = encoder(x)
    assert y.shape == (2, 37, 12)

    y.pow(2).mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_sensorformer_time_encoder_preserves_sequence_length() -> None:
    encoder = SensorformerTimeEncoder(
        input_channels=3,
        seq_len=64,
        latent_dim=12,
        patch_len=8,
        patch_stride=5,
        d_model=16,
        num_heads=2,
        num_layers=1,
        d_ff=32,
        dropout=0.0,
        global_patches=6,
        attention_chunk_size=None,
    )

    x = torch.randn(2, 3, 37, requires_grad=True)
    y = encoder(x)
    assert y.shape == (2, 37, 12)

    y.pow(2).mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_bosch_cpc_encoder_preserves_sequence_length() -> None:
    encoder = BoschCPCEncoder(
        input_channels=4,
        latent_dim=14,
        enc_hidden=16,
        gru_hidden=12,
        strides=(4, 2),
        filters=(5, 3),
        padding=(2, 1),
    )

    x = torch.randn(2, 4, 53, requires_grad=True)
    y = encoder(x)
    assert y.shape == (2, 53, 14)

    y.pow(2).mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_vectorized_transform_bank_matches_legacy_modules() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 7)

    for transformation_type in ("residual", "multiplicative", "feedforward"):
        legacy = NeuralTransformationLearner(
            latent_dim=7,
            hidden_dim=11,
            n_transformations=3,
            transformation_type=transformation_type,
        )
        vectorized = VectorizedTransformationBank.from_legacy(legacy, dropout=0.0)

        expected = legacy(x)
        actual = vectorized(x)
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_optimized_lnt_loss_matches_legacy_formula() -> None:
    torch.manual_seed(0)
    combined = torch.randn(3, 9, 5, 8)
    loss = LNTLoss(temperature=0.3)

    actual_train = loss((combined,))
    actual_eval = loss((combined,), eval=True)
    expected_train = _legacy_lnt_dcl_score(combined, temperature=0.3, eval_mode=False)
    expected_eval = _legacy_lnt_dcl_score(combined, temperature=0.3, eval_mode=True)

    assert torch.allclose(actual_train, expected_train, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_eval, expected_eval, atol=1e-6, rtol=1e-6)


def test_training_loss_still_averages_over_all_timesteps() -> None:
    torch.manual_seed(0)
    combined = torch.randn(4, 11, 5, 7)
    loss = LNTLoss(temperature=0.4)

    score_map = loss.per_timestep_scores((combined,))
    expected = score_map.mean()
    last_only = score_map[:, -1].mean()
    actual = loss((combined,))

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert not torch.isclose(actual, last_only, atol=1e-4, rtol=1e-4)


def test_detector_returns_last_timestep_score() -> None:
    torch.manual_seed(0)
    model = LNT(
        ts_channels=3,
        seq_len=24,
        n_transformations=4,
        hidden_dim=16,
        latent_dim=12,
        encoder_type="modern_tcn",
        encoder_cfg={
            "d_model": 16,
            "num_layers": 2,
            "kernel_sizes": [5, 7],
            "dilations": [1, 2],
            "ff_mult": 2,
            "dropout": 0.0,
        },
        transform_cfg={"hidden_dim": 16, "dropout": 0.0},
    )
    detector = LNTAnomalyDetector(model)
    x = torch.randn(5, 24, 3)

    with torch.no_grad():
        combined = model((x,))
        expected = detector.loss.per_timestep_scores((combined,))[:, -1]

    actual = detector.compute_online_anomaly_score((x,))
    assert actual.shape == (5,)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_lnt_last_step_detector_smoke() -> None:
    torch.manual_seed(0)
    batch, channels, length = 8, 3, 48
    clean = _make_sine_batch(batch, length, channels)
    anomalous = clean.clone()
    anomalous[:, -1, :] = anomalous[:, -1, :] + 3.5

    model = LNT(
        ts_channels=channels,
        seq_len=length,
        n_transformations=4,
        hidden_dim=24,
        latent_dim=12,
        encoder_type="modern_tcn",
        encoder_cfg={
            "d_model": 16,
            "num_layers": 2,
            "kernel_sizes": [5, 7],
            "dilations": [1, 2],
            "ff_mult": 2,
            "dropout": 0.0,
        },
        transform_cfg={"hidden_dim": 16, "dropout": 0.0},
    )
    loss = LNTLoss()
    detector = LNTAnomalyDetector(model, loss=loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    model.train()
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        combined = model((clean,))
        train_loss = loss((combined,))
        train_loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        clean_scores = detector.compute_online_anomaly_score((clean,))
        anomaly_scores = detector.compute_online_anomaly_score((anomalous,))

    assert anomaly_scores.mean() > clean_scores.mean()
