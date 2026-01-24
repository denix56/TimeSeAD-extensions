# Local Neural Transformations (LNT) - a self-supervised method for
# anomalous region detection in time series
# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from timesead.models import BaseModel
from timesead.models.common import AnomalyDetector
from timesead.optim.loss import Loss


def _get_attr(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def activation_from_name(name: str) -> nn.Module:
    if not hasattr(nn, name):
        raise ValueError(f"Activation {name} not found in torch.nn")
    return getattr(nn, name)


class ResidualBlock(nn.Module):
    """Residual blocks as used in Neural Transformation Learning."""

    def __init__(
        self,
        in_channels: int,
        n_filters: Sequence[int],
        filter_sizes: Sequence[int],
        strides: Sequence[int],
        paddings: Sequence[int],
        activation: nn.Module = nn.ReLU,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if not (len(n_filters) == len(filter_sizes) == len(strides) == len(paddings)):
            raise ValueError("ResidualBlock configuration sequences must have the same length.")

        convs: List[nn.Conv1d] = []
        hidden_size = in_channels
        for n, f_size, f_stride, f_pad in zip(n_filters, filter_sizes, strides, paddings):
            convs.append(
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=n,
                    kernel_size=f_size,
                    stride=f_stride,
                    padding=f_pad,
                    bias=bias,
                )
            )
            hidden_size = n
        self.convs = nn.ModuleList(convs)
        self.activation = activation(inplace=True)

    @classmethod
    def from_config(cls, config: Any) -> "ResidualBlock":
        return cls(
            in_channels=_get_attr(config, "in_channels"),
            n_filters=_get_attr(config, "n_filters"),
            filter_sizes=_get_attr(config, "filter_sizes"),
            strides=_get_attr(config, "strides"),
            paddings=_get_attr(config, "paddings"),
            activation=nn.ReLU,
            bias=_get_attr(config, "bias", True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.unsqueeze(1) if x.dim() == 2 else x
        for idx, conv in enumerate(self.convs):
            out = conv(out)
            if idx + 1 == len(self.convs):
                out = self.activation(out.squeeze(1) + x)
            else:
                out = self.activation(out)
        return out


class LearnableTransformation(nn.Module):
    """A learnable transformation parameterized by a network."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @staticmethod
    def get_compatible_activation() -> nn.Module:
        return nn.Identity


class FeedForwardTransformation(LearnableTransformation):
    @staticmethod
    def get_compatible_activation() -> nn.Module:
        return nn.Identity


class ResidualTransformation(LearnableTransformation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.network(x)

    @staticmethod
    def get_compatible_activation() -> nn.Module:
        return nn.Tanh


class MultiplicativeTransformation(LearnableTransformation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.network(x)

    @staticmethod
    def get_compatible_activation() -> nn.Module:
        return nn.Sigmoid


def get_transformation_layer(transformation_type: str) -> type[LearnableTransformation]:
    return {
        "residual": ResidualTransformation,
        "feedforward": FeedForwardTransformation,
        "multiplicative": MultiplicativeTransformation,
    }[transformation_type.lower()]


def create_strided_convolutional_encoder(
    input_size: int,
    hidden_size: int,
    strides: Sequence[int] | None = None,
    filters: Sequence[int] | None = None,
    padding: Sequence[int] | None = None,
) -> nn.Sequential:
    strides = strides or [5, 4, 2, 2, 2]
    filters = filters or [10, 8, 4, 4, 4]
    padding = padding or [2, 2, 2, 2, 1]
    if not (len(strides) == len(filters) == len(padding)):
        raise ValueError("Encoder configuration sequences must have the same length.")

    enc = nn.Sequential()
    for layer, (stride, kernel, pad) in enumerate(zip(strides, filters, padding)):
        enc.add_module(
            name=f"enc_layer{layer}",
            module=nn.Sequential(
                nn.Conv1d(
                    in_channels=input_size,
                    out_channels=hidden_size,
                    kernel_size=kernel,
                    stride=stride,
                    padding=pad,
                ),
                nn.ReLU(),
            ),
        )
        input_size = hidden_size
    return enc


class InfoNCE(nn.Module):
    """Noise Contrastive Estimation loss used in CPC."""

    def __init__(self, enc_emb_size: int, ar_emb_size: int, n_prediction_steps: int) -> None:
        super().__init__()
        self._enc_emb_size = enc_emb_size
        self._ar_emb_size = ar_emb_size
        self._n_prediction_steps = n_prediction_steps
        self._future_predictor = nn.Linear(
            in_features=ar_emb_size,
            out_features=enc_emb_size * n_prediction_steps,
            bias=False,
        )

    def n_prediction_steps(self) -> int:
        return self._n_prediction_steps

    def _pack_batch_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, self._enc_emb_size)

    def predict_future(self, c: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        pred = self._future_predictor(c)
        if k is None:
            return pred
        start = (k - 1) * self._enc_emb_size
        end = k * self._enc_emb_size
        return pred[:, :-k, start:end]

    def enumerate_future(self, z: torch.Tensor, c: torch.Tensor) -> Iterable[Tuple[int, torch.Tensor, torch.Tensor]]:
        Wc = self.predict_future(c)
        for k in range(1, self._n_prediction_steps + 1):
            z_k = z[:, k:, :]
            Wc_k = Wc[:, :-k, (k - 1) * self._enc_emb_size: k * self._enc_emb_size]
            yield k, z_k, Wc_k

    @staticmethod
    def _compute_f_score(z_k: torch.Tensor, Wc_k: torch.Tensor) -> torch.Tensor:
        return torch.matmul(z_k, Wc_k)

    def partial_contrastive_loss(self, z_k: torch.Tensor, Wc_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_k_flat = self._pack_batch_sequence(z_k)
        Wc_k_flat = self._pack_batch_sequence(Wc_k)
        scores = self._compute_f_score(z_k_flat, Wc_k_flat.T)
        pos_scores = scores.diagonal()
        denom = torch.logsumexp(scores, dim=(0, 1))
        loss = -((pos_scores - denom).sum() / scores.size(0))
        acc = (scores.argmax(dim=0) == torch.arange(scores.size(0), device=scores.device)).float().mean()
        return loss, acc

    def contrastive_loss(self, z: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        total_loss = 0.0
        accuracies = torch.zeros(self._n_prediction_steps, device=z.device)
        Wc = self._future_predictor(c)
        for k in range(1, self._n_prediction_steps + 1):
            z_k = z[:, k:, :]
            Wc_k = Wc[:, :-k, (k - 1) * self._enc_emb_size: k * self._enc_emb_size]
            z_k_flat = self._pack_batch_sequence(z_k)
            Wc_k_flat = self._pack_batch_sequence(Wc_k)
            scores = self._compute_f_score(z_k_flat, Wc_k_flat.T)
            pos_scores = scores.diagonal()
            denom = torch.logsumexp(scores, dim=(0, 1))
            total_loss += -((pos_scores - denom).sum() / scores.size(0))
            accuracies[k - 1] = (
                scores.argmax(dim=0)
                == torch.arange(scores.size(0), device=scores.device)
            ).float().mean()
        total_loss = total_loss / self._n_prediction_steps
        return total_loss, accuracies


class CPCNetwork(nn.Module):
    """Contrastive Predictive Coding network used by LNT."""

    def __init__(
        self,
        input_size: int,
        enc_emb_size: int,
        ar_emb_size: int,
        n_prediction_steps: int,
        encoder: Optional[nn.Module] = None,
        autoregressive: Optional[nn.Module] = None,
        encoder_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._enc_emb_size = enc_emb_size
        self._ar_emb_size = ar_emb_size
        if encoder is None:
            if encoder_config is None:
                self._encoder = create_strided_convolutional_encoder(
                    input_size=input_size, hidden_size=enc_emb_size
                )
            else:
                self._encoder = create_strided_convolutional_encoder(
                    input_size=input_size,
                    hidden_size=enc_emb_size,
                    strides=encoder_config.get("strides"),
                    filters=encoder_config.get("filter_sizes"),
                    padding=encoder_config.get("paddings"),
                )
        else:
            self._encoder = encoder

        self._autoregressive = autoregressive or nn.GRU(
            input_size=enc_emb_size, hidden_size=ar_emb_size, batch_first=True
        )
        self._infoNCE = InfoNCE(enc_emb_size, ar_emb_size, n_prediction_steps=n_prediction_steps)

    def get_embeddings(
        self, x: torch.Tensor, device: torch.device, return_global_context: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        z = self._encoder(x).permute(0, 2, 1)
        if not return_global_context:
            return z
        batch_size = x.size(0)
        h = torch.zeros((1, batch_size, self._ar_emb_size), device=device)
        c, _ = self._autoregressive(z, h)
        return z, c

    def forward(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        z, c = self.get_embeddings(x, device)
        loss, _ = self._infoNCE.contrastive_loss(z, c)
        return loss


class NeuralTransformationNetwork(nn.Module):
    """Neural Transformation Learning Network Module."""

    def __init__(
        self,
        input_size: int,
        n_transformations: int,
        transformation_type: str,
        resnet_config: Optional[Sequence[Any]] = None,
        mlp_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.n_transformations = n_transformations
        transformation = get_transformation_layer(transformation_type)

        self._transformations = nn.ModuleList()
        if resnet_config is not None:
            for _ in range(self.n_transformations):
                residual_blocks = [ResidualBlock.from_config(conf) for conf in resnet_config]
                self._transformations.append(transformation(nn.Sequential(*residual_blocks)))
        elif mlp_config is not None:
            n_hidden = mlp_config.get("n_hidden", [])
            activation = activation_from_name(mlp_config.get("activation", "ReLU"))
            use_bias = mlp_config.get("bias", True)
            for _ in range(self.n_transformations):
                layers: List[nn.Module] = []
                layer_size = input_size
                for n in n_hidden:
                    layers.append(nn.Linear(layer_size, n, bias=use_bias))
                    layers.append(activation())
                    layer_size = n
                layers.append(nn.Linear(layer_size, input_size, bias=use_bias))
                layers.append(transformation.get_compatible_activation()())
                self._transformations.append(transformation(nn.Sequential(*layers)))
        else:
            raise ValueError("Either resnet_config or mlp_config must be provided.")

    @staticmethod
    def _cosine_similarity(x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=-1)
        return torch.matmul(x_norm, x_norm.transpose(-2, -1))

    def deterministic_contrastive_loss(
        self, z: torch.Tensor, temperature: float, average: bool = True
    ) -> torch.Tensor:
        z_transformed = self(z)
        z_all = torch.cat([z.unsqueeze(1), z_transformed], dim=1)
        sim = self._cosine_similarity(z_all) / temperature
        sim_pos = sim[:, 1:, 0]
        nt = self.n_transformations + 1
        diag_mask = torch.eye(nt, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(diag_mask, float("-inf"))
        normalization = torch.logsumexp(sim[:, 1:, :], dim=-1)
        loss = -(sim_pos - normalization).sum(dim=-1)
        return loss.mean() if average else loss

    def dynamic_deterministic_contrastive_loss(
        self,
        z_tilde: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        z_transformed: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        average: bool = True,
    ) -> torch.Tensor:
        if z_transformed is None and z is None:
            raise ValueError("Either z or z_transformed must be provided.")
        if z_transformed is None:
            z_transformed = self(z)
        z_all = torch.cat([z_tilde.unsqueeze(1), z_transformed], dim=1)
        sim = self._cosine_similarity(z_all) / temperature
        sim_pos = sim[:, 1:, 0]
        nt = self.n_transformations + 1
        diag_mask = torch.eye(nt, device=z_tilde.device, dtype=torch.bool)
        sim = sim.masked_fill(diag_mask, float("-inf"))
        normalization = torch.logsumexp(sim[:, 1:, :], dim=-1)
        loss = -(sim_pos - normalization).sum(dim=-1)
        return loss.mean() if average else loss

    def forward(self, z: torch.Tensor, transformation_dim: int = 1) -> torch.Tensor:
        return torch.stack([trans(z) for trans in self._transformations], dim=transformation_dim)


class LNTNetwork(CPCNetwork):
    """Joint CPC + LNT network for local neural transformations."""

    def __init__(
        self,
        input_size: int,
        enc_emb_size: int,
        ar_emb_size: int,
        n_prediction_steps: int,
        n_transformations: int,
        transformation_type: str,
        dcl_temperature: float,
        resnet_config: Optional[Sequence[Any]] = None,
        mlp_config: Optional[Dict[str, Any]] = None,
        encoder_config: Optional[Dict[str, Any]] = None,
        detach: bool = True,
        pretrain_representations_for_epochs: Optional[int] = None,
    ) -> None:
        super().__init__(
            input_size=input_size,
            enc_emb_size=enc_emb_size,
            ar_emb_size=ar_emb_size,
            n_prediction_steps=n_prediction_steps,
            encoder_config=encoder_config,
        )
        self._neuTraL = NeuralTransformationNetwork(
            input_size=enc_emb_size,
            n_transformations=n_transformations,
            transformation_type=transformation_type,
            resnet_config=resnet_config,
            mlp_config=mlp_config,
        )
        self._temperature = dcl_temperature
        self._detach = detach
        self._pretrain_representations_for = pretrain_representations_for_epochs

    def _pack_batch_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, self._enc_emb_size)

    def score_outlierness(self, x: torch.Tensor, device: torch.device, score_backward: bool = False) -> torch.Tensor:
        z, c = self.get_embeddings(x, device)
        batch_size, seq_len, _ = z.shape
        loss_dcl = torch.zeros((batch_size, seq_len), device=device)
        for k, z_k, Wc_k in self._infoNCE.enumerate_future(z, c):
            loss_dcl_k = self._neuTraL.dynamic_deterministic_contrastive_loss(
                z=self._pack_batch_sequence(z_k),
                z_tilde=self._pack_batch_sequence(Wc_k),
                temperature=self._temperature,
                average=False,
            )
            if score_backward:
                loss_dcl[:, :-k] += loss_dcl_k.reshape(batch_size, seq_len - k)
            else:
                loss_dcl[:, k:] += loss_dcl_k.reshape(batch_size, seq_len - k)
        return loss_dcl

    def transform(self, z: torch.Tensor) -> torch.Tensor:
        return self._neuTraL(z)

    def forward(
        self, x: torch.Tensor, device: torch.device, epoch: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if epoch is None or self._pretrain_representations_for is None:
            detach = self._detach
        else:
            detach = self._detach if epoch > self._pretrain_representations_for else True
        z, c = self.get_embeddings(x, device)
        loss_cpc = 0.0
        loss_dcl = 0.0
        acc_cpc = torch.zeros(self._infoNCE.n_prediction_steps(), device=device)
        for k, z_k, Wc_k in self._infoNCE.enumerate_future(z, c):
            loss_cpc_k, acc_cpc_k = self._infoNCE.partial_contrastive_loss(z_k, Wc_k)
            batch_size, seq_len, emb_size = z_k.shape
            idx = torch.randint(0, seq_len, (batch_size, 1), device=device)
            idx_exp = idx[:, :, None].expand(-1, -1, emb_size)
            z_sample = torch.take_along_dim(z_k.detach() if detach else z_k, idx_exp, dim=1).squeeze(1)
            Wc_sample = torch.take_along_dim(Wc_k.detach() if detach else Wc_k, idx_exp, dim=1).squeeze(1)
            loss_dcl_k = self._neuTraL.dynamic_deterministic_contrastive_loss(
                z=z_sample,
                z_tilde=Wc_sample,
                temperature=self._temperature,
            )
            loss_cpc += loss_cpc_k
            loss_dcl += loss_dcl_k
            acc_cpc[k - 1] = acc_cpc_k
        loss_cpc = loss_cpc / self._infoNCE.n_prediction_steps()
        loss_dcl = loss_dcl / self._infoNCE.n_prediction_steps()
        return loss_cpc, loss_dcl, acc_cpc


class InverseLNTNetwork(CPCNetwork):
    """Inverse LNT variant that applies transformations in data space."""

    def __init__(
        self,
        input_size: int,
        enc_emb_size: int,
        ar_emb_size: int,
        n_prediction_steps: int,
        n_transformations: int,
        transformation_type: str,
        dcl_temperature: float,
        downsampling_factor: int,
        resnet_config: Optional[Sequence[Any]] = None,
        mlp_config: Optional[Dict[str, Any]] = None,
        encoder_config: Optional[Dict[str, Any]] = None,
        detach: bool = True,
        pretrain_representations_for_epochs: Optional[int] = None,
    ) -> None:
        super().__init__(
            input_size=input_size,
            enc_emb_size=enc_emb_size,
            ar_emb_size=ar_emb_size,
            n_prediction_steps=n_prediction_steps,
            encoder_config=encoder_config,
        )
        self._neuTraL = NeuralTransformationNetwork(
            input_size=input_size,
            n_transformations=n_transformations,
            transformation_type=transformation_type,
            resnet_config=resnet_config,
            mlp_config=mlp_config,
        )
        self._downsampling_factor = downsampling_factor
        self._temperature = dcl_temperature
        self._detach = detach
        self._pretrain_representations_for = pretrain_representations_for_epochs

    def _pack_batch_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x.reshape(-1, self._enc_emb_size)
        return x.transpose(1, 2).reshape(-1, self._neuTraL.n_transformations, self._enc_emb_size)

    def unfold_time_distributed(self, x: torch.Tensor) -> torch.Tensor:
        bs, nc, _ = x.shape
        x_unfold = x.unfold(-1, self._downsampling_factor, self._downsampling_factor)
        x_unfold = x_unfold.transpose(1, 2).reshape(-1, nc, self._downsampling_factor)
        return x_unfold

    def enumerate_transformed_future(
        self, z: torch.Tensor, z_trans: torch.Tensor, c: torch.Tensor
    ) -> Iterable[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
        for k, z_k, c_k in self._infoNCE.enumerate_future(z, c):
            z_trans_k = z_trans[:, :, k:, :].transpose(0, 1)
            yield k, z_k, z_trans_k, c_k

    def score_outlierness(self, x: torch.Tensor, device: torch.device, score_backward: bool = False) -> torch.Tensor:
        bs, nc, sl = x.shape
        x_unfold = self.unfold_time_distributed(x)
        x_trans = self._neuTraL(x_unfold, transformation_dim=0)
        x_trans = (
            x_trans.reshape(self._neuTraL.n_transformations, bs, -1, nc, self._downsampling_factor)
            .transpose(2, 3)
            .reshape(self._neuTraL.n_transformations, bs, nc, -1)
        )
        z, c = self.get_embeddings(x, device=device)
        z_trans = self.get_embeddings(
            x_trans.reshape(-1, nc, sl),
            device=device,
            return_global_context=False,
        )
        z_trans = z_trans.reshape(self._neuTraL.n_transformations, bs, -1, self._enc_emb_size)
        batch_size, seq_len, _ = z.shape
        loss_dcl = torch.zeros((batch_size, seq_len), device=device)
        for k, z_k, z_trans_k, Wc_k in self.enumerate_transformed_future(z, z_trans, c):
            loss_dcl_k = self._neuTraL.dynamic_deterministic_contrastive_loss(
                z_transformed=self._pack_batch_sequence(z_trans_k),
                z_tilde=self._pack_batch_sequence(Wc_k),
                temperature=self._temperature,
                average=False,
            )
            if score_backward:
                loss_dcl[:, :-k] += loss_dcl_k.reshape(batch_size, seq_len - k)
            else:
                loss_dcl[:, k:] += loss_dcl_k.reshape(batch_size, seq_len - k)
        return loss_dcl

    def forward(
        self, x: torch.Tensor, device: torch.device, epoch: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if epoch is None or self._pretrain_representations_for is None:
            detach = self._detach
        else:
            detach = self._detach if epoch > self._pretrain_representations_for else True

        bs, nc, sl = x.shape
        x_unfold = self.unfold_time_distributed(x)
        x_trans = self._neuTraL(x_unfold, transformation_dim=0)
        x_trans = (
            x_trans.reshape(self._neuTraL.n_transformations, bs, -1, nc, self._downsampling_factor)
            .transpose(2, 3)
            .reshape(self._neuTraL.n_transformations, bs, nc, -1)
        )
        z, c = self.get_embeddings(x, device=device)
        z_trans = self.get_embeddings(
            x_trans.reshape(-1, nc, sl),
            device=device,
            return_global_context=False,
        )
        z_trans = z_trans.reshape(self._neuTraL.n_transformations, bs, -1, self._enc_emb_size)
        loss_cpc = 0.0
        loss_dcl = 0.0
        acc_cpc = torch.zeros(self._infoNCE.n_prediction_steps(), device=device)
        for k, z_k, z_trans_k, Wc_k in self.enumerate_transformed_future(z, z_trans, c):
            loss_cpc_k, acc_cpc_k = self._infoNCE.partial_contrastive_loss(z_k, Wc_k)
            batch_size, seq_len, emb_size = z_k.shape
            idx = torch.randint(0, seq_len, (batch_size, 1), device=device)
            idx_exp = idx[:, :, None].expand(-1, -1, emb_size)
            trans_idx = idx_exp[:, None, :, :].expand(-1, self._neuTraL.n_transformations, -1, -1)
            z_trans_sample = torch.take_along_dim(
                z_trans_k.detach() if detach else z_trans_k, trans_idx, dim=2
            ).squeeze(2)
            Wc_sample = torch.take_along_dim(
                Wc_k.detach() if detach else Wc_k, idx_exp, dim=1
            ).squeeze(1)
            loss_dcl_k = self._neuTraL.dynamic_deterministic_contrastive_loss(
                z_transformed=z_trans_sample,
                z_tilde=Wc_sample,
                temperature=self._temperature,
            )
            loss_cpc += loss_cpc_k
            loss_dcl += loss_dcl_k
            acc_cpc[k - 1] = acc_cpc_k
        loss_cpc = loss_cpc / self._infoNCE.n_prediction_steps()
        loss_dcl = loss_dcl / self._infoNCE.n_prediction_steps()
        return loss_cpc, loss_dcl, acc_cpc


class LNT(BaseModel):
    """Local Neural Transformations model following the TimeseAD interface."""

    def __init__(
        self,
        ts_channels: int,
        seq_len: int,
        enc_emb_size: int,
        ar_emb_size: int,
        n_prediction_steps: int,
        n_transformations: int = 4,
        transformation_type: str = "residual",
        dcl_temperature: float = 1.0,
        resnet_config: Optional[Sequence[Any]] = None,
        mlp_config: Optional[Dict[str, Any]] = None,
        encoder_config: Optional[Dict[str, Any]] = None,
        inverse_order: bool = False,
        downsampling_factor: int = 160,
        detach: bool = True,
        pretrain_representations_for_epochs: Optional[int] = None,
    ) -> None:
        super().__init__()
        if inverse_order:
            self._network: LNTNetwork | InverseLNTNetwork = InverseLNTNetwork(
                input_size=ts_channels,
                enc_emb_size=enc_emb_size,
                ar_emb_size=ar_emb_size,
                n_prediction_steps=n_prediction_steps,
                n_transformations=n_transformations,
                transformation_type=transformation_type,
                dcl_temperature=dcl_temperature,
                downsampling_factor=downsampling_factor,
                resnet_config=resnet_config,
                mlp_config=mlp_config,
                encoder_config=encoder_config,
                detach=detach,
                pretrain_representations_for_epochs=pretrain_representations_for_epochs,
            )
        else:
            self._network = LNTNetwork(
                input_size=ts_channels,
                enc_emb_size=enc_emb_size,
                ar_emb_size=ar_emb_size,
                n_prediction_steps=n_prediction_steps,
                n_transformations=n_transformations,
                transformation_type=transformation_type,
                dcl_temperature=dcl_temperature,
                resnet_config=resnet_config,
                mlp_config=mlp_config,
                encoder_config=encoder_config,
                detach=detach,
                pretrain_representations_for_epochs=pretrain_representations_for_epochs,
            )
        self.seq_len = seq_len

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, = inputs
        x = x.permute(0, 2, 1).contiguous()
        device = x.device
        return self._network(x, device=device)

    def score_outlierness(self, inputs: Tuple[torch.Tensor, ...], score_backward: bool = False) -> torch.Tensor:
        x, = inputs
        x = x.permute(0, 2, 1).contiguous()
        device = x.device
        with torch.inference_mode():
            return self._network.score_outlierness(x, device=device, score_backward=score_backward)


class LNTLoss(Loss):
    """Loss for the LNT model."""

    def __init__(self, lambda_weighting: float = 1.0) -> None:
        super().__init__()
        self.lambda_weighting = lambda_weighting

    def forward(
        self,
        predictions: Tuple[torch.Tensor, ...],
        targets: Optional[Tuple[torch.Tensor, ...]] = None,
        eval: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        loss_cpc, loss_dcl, _ = predictions
        if eval:
            return loss_dcl
        return loss_cpc + self.lambda_weighting * loss_dcl


class LNTAnomalyDetector(AnomalyDetector):
    """Anomaly detector wrapper for LNT."""

    def __init__(self, model: LNT, loss: LNTLoss) -> None:
        super().__init__()
        self.model = model
        self.loss = loss

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        scores = self.model.score_outlierness(inputs)
        return scores[:, -1]

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.model.score_outlierness(inputs)

    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs: Any) -> None:
        pass

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        label, = targets
        return label[:, -1]
