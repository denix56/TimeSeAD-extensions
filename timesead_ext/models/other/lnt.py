from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timesead.models.common import AnomalyDetector
    from timesead.models import BaseModel
    from timesead.optim.loss import Loss
    from timesead.utils.utils import pack_tuple
except ModuleNotFoundError as exc:
    if exc.name is None or not exc.name.startswith("timesead"):
        raise

    class BaseModel(nn.Module):
        pass

    class AnomalyDetector(nn.Module):
        pass

    class Loss(nn.Module):
        pass

    def pack_tuple(x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return (x,)

from timesead_ext.models.encoders.patchtst import (
    PatchTSTEncoder,
    _compute_num_patches,
    _compute_padded_len,
)


class SimpleEncoder(nn.Module):
    """Legacy CNN encoder that keeps one latent vector per timestep."""

    def __init__(self, input_channels: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return self.conv3(x)


class CNNTimeEncoder(nn.Module):
    """Compatibility wrapper that returns timestep-aligned latents."""

    def __init__(self, input_channels: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = SimpleEncoder(input_channels, hidden_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).transpose(1, 2).contiguous()


class PatchTSTTimeEncoder(nn.Module):
    """PatchTST backbone adapted back to per-timestep latent vectors."""

    def __init__(
        self,
        input_channels: int,
        seq_len: int,
        latent_dim: int,
        patch_len: int = 16,
        patch_stride: int = 8,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        token_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.patch_encoder = PatchTSTEncoder(
            seq_len=seq_len,
            patch_len=patch_len,
            patch_stride=patch_stride,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            pooling="base",
            token_chunk_size=token_chunk_size,
        )
        self.patch_to_time = nn.Linear(d_model, patch_len * d_model)
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_fuse = nn.Linear(input_channels * d_model, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, actual_len = x.shape
        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {channels}")

        num_patches = _compute_num_patches(actual_len, self.patch_len, self.patch_stride)
        max_patches = self.patch_encoder.pos_embed.shape[1]
        if num_patches > max_patches:
            raise ValueError(
                f"Input length {actual_len} requires {num_patches} patches, "
                f"but encoder was initialized for at most {max_patches} patches"
            )

        patch_tokens = self.patch_encoder(x)
        patch_tokens = patch_tokens.reshape(batch_size, channels, num_patches, self.d_model)

        patch_values = self.patch_to_time(patch_tokens)
        padded_len = _compute_padded_len(actual_len, self.patch_len, self.patch_stride)
        patch_values = patch_values.reshape(batch_size * channels, num_patches, self.d_model * self.patch_len)
        patch_values = patch_values.transpose(1, 2).contiguous()

        time_tokens = F.fold(
            patch_values,
            output_size=(1, padded_len),
            kernel_size=(1, self.patch_len),
            stride=(1, self.patch_stride),
        )
        counts = self._overlap_counts(num_patches, padded_len, x.device, x.dtype)
        time_tokens = time_tokens / counts
        time_tokens = time_tokens.squeeze(2)[..., :actual_len].transpose(1, 2).contiguous()
        time_tokens = time_tokens.reshape(batch_size, channels, actual_len, self.d_model)
        time_tokens = self.channel_norm(time_tokens)
        time_tokens = time_tokens.permute(0, 2, 1, 3).reshape(batch_size, actual_len, channels * self.d_model)
        return self.output_norm(self.channel_fuse(time_tokens))

    def _overlap_counts(
        self,
        num_patches: int,
        padded_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ones = torch.ones(1, self.patch_len, num_patches, device=device, dtype=dtype)
        counts = F.fold(
            ones,
            output_size=(1, padded_len),
            kernel_size=(1, self.patch_len),
            stride=(1, self.patch_stride),
        )
        return counts.clamp_min_(1.0)


def _make_strided_conv_encoder(
    input_channels: int,
    hidden_dim: int,
    strides: Sequence[int],
    filters: Sequence[int],
    padding: Sequence[int],
) -> nn.Sequential:
    if len(strides) != len(filters) or len(strides) != len(padding):
        raise ValueError("strides, filters, and padding must have the same length")

    layers = []
    in_channels = input_channels
    for idx, (stride, kernel_size, pad) in enumerate(zip(strides, filters, padding)):
        layers.append(
            nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad,
                ),
                nn.ReLU(),
            )
        )
        in_channels = hidden_dim

    return nn.Sequential(*layers)


class BoschCPCEncoder(nn.Module):
    """Bosch-style strided conv + GRU backbone, adapted to timestep outputs."""

    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        enc_hidden: int = 128,
        gru_hidden: int = 128,
        strides: Sequence[int] = (5, 4, 2, 2, 2),
        filters: Sequence[int] = (10, 8, 4, 4, 4),
        padding: Sequence[int] = (2, 2, 2, 2, 1),
        upsampler: str = "linear",
        upsample_chunk_size: int = 8,
        upsample_chunk_threshold: int = 1024,
    ) -> None:
        super().__init__()
        if upsampler != "linear":
            raise ValueError(f"Unsupported upsampler: {upsampler}")

        self.latent_dim = latent_dim
        self.upsampler = upsampler
        self.upsample_chunk_size = upsample_chunk_size
        self.upsample_chunk_threshold = upsample_chunk_threshold

        self.encoder = _make_strided_conv_encoder(input_channels, enc_hidden, strides, filters, padding)
        self.context = nn.GRU(enc_hidden, gru_hidden, batch_first=True)
        self.proj = nn.Linear(enc_hidden + gru_hidden, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_len = x.shape[-1]
        z_local = self.encoder(x).transpose(1, 2).contiguous()
        z_context, _ = self.context(z_local)
        z = self.output_norm(self.proj(torch.cat([z_local, z_context], dim=-1)))
        return self._upsample(z, target_len)

    def _upsample(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        if z.shape[1] == target_len:
            return z

        z = z.transpose(1, 2).contiguous()
        if target_len > self.upsample_chunk_threshold and z.shape[0] > self.upsample_chunk_size:
            parts = []
            for chunk in torch.split(z, self.upsample_chunk_size, dim=0):
                parts.append(F.interpolate(chunk, size=target_len, mode=self.upsampler, align_corners=False))
            z = torch.cat(parts, dim=0)
        else:
            z = F.interpolate(z, size=target_len, mode=self.upsampler, align_corners=False)

        return z.transpose(1, 2).contiguous()


class TransformationNetwork(nn.Module):
    """Legacy single-transformation MLP used for compatibility tests."""

    def __init__(self, latent_dim: int, hidden_dim: int, transformation_type: str = "residual"):
        super().__init__()
        self.transformation_type = transformation_type
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        if transformation_type == "residual":
            self.output_activation: nn.Module = nn.Tanh()
        elif transformation_type == "multiplicative":
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.output_activation(self.net(z))
        if self.transformation_type == "residual":
            return z + out
        if self.transformation_type == "multiplicative":
            return z * out
        return out


class NeuralTransformationLearner(nn.Module):
    """Legacy looped transform bank kept for compatibility and parity tests."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        n_transformations: int,
        transformation_type: str = "residual",
    ):
        super().__init__()
        self.n_transformations = n_transformations
        self.transformations = nn.ModuleList(
            [TransformationNetwork(latent_dim, hidden_dim, transformation_type) for _ in range(n_transformations)]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.stack([trans(z) for trans in self.transformations], dim=-2)


class VectorizedTransformationBank(nn.Module):
    """Vectorized per-transformation MLP bank without Python loops."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        n_transformations: int,
        transformation_type: str = "residual",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_transformations = n_transformations
        self.transformation_type = transformation_type
        self.dropout = nn.Dropout(dropout)

        self.weight1 = nn.Parameter(torch.empty(n_transformations, hidden_dim, latent_dim))
        self.bias1 = nn.Parameter(torch.empty(n_transformations, hidden_dim))
        self.weight2 = nn.Parameter(torch.empty(n_transformations, hidden_dim, hidden_dim))
        self.bias2 = nn.Parameter(torch.empty(n_transformations, hidden_dim))
        self.weight3 = nn.Parameter(torch.empty(n_transformations, latent_dim, hidden_dim))
        self.bias3 = nn.Parameter(torch.empty(n_transformations, latent_dim))
        self.activation = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self._reset_linear(self.weight1, self.bias1)
        self._reset_linear(self.weight2, self.bias2)
        self._reset_linear(self.weight3, self.bias3)

    @staticmethod
    def _reset_linear(weight: torch.Tensor, bias: torch.Tensor) -> None:
        for idx in range(weight.shape[0]):
            nn.init.kaiming_uniform_(weight[idx], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight[idx])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias[idx], -bound, bound)

    @classmethod
    def from_legacy(
        cls,
        legacy: NeuralTransformationLearner,
        dropout: float = 0.0,
    ) -> "VectorizedTransformationBank":
        if not legacy.transformations:
            raise ValueError("legacy transform bank must contain at least one transformation")

        first = legacy.transformations[0]
        linear1, _, linear2, _, linear3 = first.net
        bank = cls(
            latent_dim=linear1.in_features,
            hidden_dim=linear1.out_features,
            n_transformations=len(legacy.transformations),
            transformation_type=first.transformation_type,
            dropout=dropout,
        )

        with torch.no_grad():
            for idx, transformation in enumerate(legacy.transformations):
                l1, _, l2, _, l3 = transformation.net
                bank.weight1[idx].copy_(l1.weight)
                bank.bias1[idx].copy_(l1.bias)
                bank.weight2[idx].copy_(l2.weight)
                bank.bias2[idx].copy_(l2.bias)
                bank.weight3[idx].copy_(l3.weight)
                bank.bias3[idx].copy_(l3.bias)

        return bank

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = torch.einsum("...d,nhd->...nh", z, self.weight1) + self.bias1
        hidden = self.dropout(self.activation(hidden))
        hidden = torch.einsum("...ni,noi->...no", hidden, self.weight2) + self.bias2
        hidden = self.dropout(self.activation(hidden))
        out = torch.einsum("...nh,ndh->...nd", hidden, self.weight3) + self.bias3
        base = z.unsqueeze(-2)

        if self.transformation_type == "residual":
            return base + torch.tanh(out)
        if self.transformation_type == "multiplicative":
            return base * torch.sigmoid(out)
        return out


class LNT(BaseModel):
    """Static LNT variant with timestep-aligned encoder backbones."""

    def __init__(
        self,
        ts_channels: int,
        seq_len: int,
        n_transformations: int = 10,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        transformation_type: str = "residual",
        encoder_type: str = "patchtst_time",
        encoder_cfg: Optional[Dict[str, Any]] = None,
        transform_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.ts_channels = ts_channels
        self.seq_len = seq_len
        self.n_transformations = n_transformations
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type

        encoder_cfg = dict(encoder_cfg or {})
        transform_cfg = dict(transform_cfg or {})

        if encoder_type == "cnn":
            encoder_hidden = int(encoder_cfg.get("hidden_dim", hidden_dim))
            self.encoder: nn.Module = CNNTimeEncoder(ts_channels, encoder_hidden, latent_dim)
        elif encoder_type == "patchtst_time":
            cfg = {
                "patch_len": 16,
                "patch_stride": 8,
                "d_model": 128,
                "num_heads": 4,
                "num_layers": 3,
                "d_ff": 256,
                "dropout": 0.1,
                **encoder_cfg,
            }
            self.encoder = PatchTSTTimeEncoder(
                input_channels=ts_channels,
                seq_len=seq_len,
                latent_dim=latent_dim,
                patch_len=int(cfg["patch_len"]),
                patch_stride=int(cfg["patch_stride"]),
                d_model=int(cfg["d_model"]),
                num_heads=int(cfg["num_heads"]),
                num_layers=int(cfg["num_layers"]),
                d_ff=int(cfg["d_ff"]),
                dropout=float(cfg["dropout"]),
                token_chunk_size=int(cfg["token_chunk_size"]) if "token_chunk_size" in cfg else None,
            )
        elif encoder_type == "bosch_cpc":
            cfg = {
                "enc_hidden": 128,
                "gru_hidden": 128,
                "strides": [5, 4, 2, 2, 2],
                "filters": [10, 8, 4, 4, 4],
                "padding": [2, 2, 2, 2, 1],
                "upsampler": "linear",
                **encoder_cfg,
            }
            self.encoder = BoschCPCEncoder(
                input_channels=ts_channels,
                latent_dim=latent_dim,
                enc_hidden=int(cfg["enc_hidden"]),
                gru_hidden=int(cfg["gru_hidden"]),
                strides=tuple(int(v) for v in cfg["strides"]),
                filters=tuple(int(v) for v in cfg["filters"]),
                padding=tuple(int(v) for v in cfg["padding"]),
                upsampler=str(cfg["upsampler"]),
                upsample_chunk_size=int(cfg.get("upsample_chunk_size", 8)),
                upsample_chunk_threshold=int(cfg.get("upsample_chunk_threshold", 1024)),
            )
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        transform_hidden = int(transform_cfg.get("hidden_dim", max(hidden_dim, latent_dim * 2)))
        transform_dropout = float(transform_cfg.get("dropout", 0.1))
        self.transformation_learner = VectorizedTransformationBank(
            latent_dim=latent_dim,
            hidden_dim=transform_hidden,
            n_transformations=n_transformations,
            transformation_type=transformation_type,
            dropout=transform_dropout,
        )

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        x, = inputs
        if x.shape[-1] != self.ts_channels:
            raise ValueError(f"Expected input with {self.ts_channels} channels, got {x.shape[-1]}")

        x = x.permute(0, 2, 1).contiguous()
        z = self.encoder(x)
        if z.shape[1] != x.shape[-1]:
            raise RuntimeError(f"Encoder {self.encoder_type} did not preserve the time dimension")

        z_transformed = self.transformation_learner(z)
        return torch.cat([z.unsqueeze(2), z_transformed], dim=2)


def _lnt_dcl_score(
    combined: torch.Tensor,
    temperature: float,
    eval_mode: bool,
    trans_diag_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, seq_len, n_all, latent_dim = combined.shape
    n_trans = n_all - 1

    combined_flat = combined.contiguous().reshape(-1, n_all, latent_dim)
    combined_norm = F.normalize(combined_flat, p=2, dim=-1)
    z_ori = combined_norm[:, :1, :]
    z_trans = combined_norm[:, 1:, :]
    sim_pos = torch.sum(z_trans * z_ori, dim=-1) / temperature

    logits = torch.matmul(z_trans, combined_norm.transpose(-2, -1)) / temperature
    if trans_diag_mask is None or trans_diag_mask.device != combined.device or trans_diag_mask.shape != (n_trans, n_trans):
        trans_diag_mask = torch.eye(n_trans, device=combined.device, dtype=torch.bool)

    logits = logits.clone()
    logits[..., 1:].masked_fill_(trans_diag_mask.unsqueeze(0), float("-inf"))
    loss_per_position = -sim_pos + torch.logsumexp(logits, dim=-1)
    loss_per_sample = loss_per_position.sum(dim=-1).reshape(batch_size, seq_len)

    if eval_mode:
        return loss_per_sample.mean(dim=1)
    return loss_per_sample.mean()


class LNTLoss(Loss):
    """Deterministic Contrastive Loss for LNT."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.register_buffer("_trans_diag_mask", torch.empty(0, 0, dtype=torch.bool), persistent=False)

    def _get_trans_diag_mask(self, n_trans: int, device: torch.device) -> torch.Tensor:
        if self._trans_diag_mask.device != device or self._trans_diag_mask.shape != (n_trans, n_trans):
            self._trans_diag_mask = torch.eye(n_trans, device=device, dtype=torch.bool)
        return self._trans_diag_mask

    def forward(
        self,
        predictions: Tuple[torch.Tensor, ...],
        targets: Tuple[torch.Tensor, ...] = None,
        eval: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        combined, = predictions
        n_trans = combined.shape[2] - 1
        diag_mask = self._get_trans_diag_mask(n_trans, combined.device)
        return _lnt_dcl_score(combined, self.temperature, eval_mode=eval, trans_diag_mask=diag_mask)


class LNTAnomalyDetector(AnomalyDetector):
    """Anomaly detector wrapper for LNT."""

    def __init__(self, model: LNT, loss: Optional[LNTLoss] = None, temperature: float = 0.5):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.loss = loss if loss is not None else LNTLoss(temperature=temperature)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        with torch.inference_mode():
            predictions = pack_tuple(self.model(inputs))

        return self.loss(predictions, eval=True)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, dataset, **kwargs) -> None:
        pass

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        label, = targets
        return label[:, -1]
