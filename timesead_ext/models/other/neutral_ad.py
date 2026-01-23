# Neural Transformation Learning for Anomaly Detection (NeuTraLAD) - a self-supervised method for anomaly detection
# Copyright (c) 2022 Robert Bosch GmbH
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

from typing import Dict, List, Optional, Sequence, Tuple, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timesead.models.common import AnomalyDetector
from timesead.models import BaseModel
from timesead.optim.loss import Loss
from timesead.utils.utils import pack_tuple
from timesead_ext.models.transforms import (
    Transform,
    TransformBank,
    freq_cfg as default_freq_cfg,
    group_cfg as default_group_cfg,
    invertible_cfg as default_invertible_cfg,
    make_freq_family,
    make_group_family,
    make_invertible_family,
)
from timesead_ext.models.encoders import ITransformerEncoder, PatchTSTEncoder
from timesead_ext.models.pooling import AttnPool, MeanMaxPool, MeanPool
from timesead_ext.models.projection import LinearHead, MLPHead


class ResTrans1DBlock(torch.nn.Module):
    def __init__(self, channel: int, bias: bool = False):
        """Initialize the residual transformation block.

        Args:
            channel: Number of channels in the input/output tensor.
            bias: Whether to use bias terms in the convolution layers.
        """
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(channel, channel, 3, 1, 1, bias=bias)
        self.in1 = nn.InstanceNorm1d(channel, affine=bias)
        self.conv2 = nn.Conv1d(channel, channel, 3, 1, 1, bias=bias)
        self.in2 = nn.InstanceNorm1d(channel, affine=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual transformation block.

        Args:
            x: Input tensor of shape (batch, channels, sequence_length).

        Returns:
            Output tensor of the same shape as the input.
        """
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 dilation: int = 1, bias: bool = False):
        """Initialize a padded 1D convolution layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the 1D convolution kernel.
            stride: Convolution stride.
            dilation: Convolution dilation.
            bias: Whether to use a bias term in the convolution.
        """
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.reflection_pad = nn.ReflectionPad1d(padding)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply reflection padding followed by convolution.

        Args:
            x: Input tensor of shape (batch, channels, sequence_length).

        Returns:
            Output tensor after padding and convolution.
        """
        out = self.reflection_pad(x)
        out = self.conv1d(out)
        return out


class SeqTransformNet(Transform):
    def __init__(self, x_dim: int, hdim: int, num_layers: int):
        """Initialize a sequence-to-sequence transform network.

        Args:
            x_dim: Number of input channels.
            hdim: Hidden channel dimension.
            num_layers: Total number of layers in the network.
        """
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = ConvLayer(x_dim, hdim, 3, 1, bias=False)
        self.in1 = nn.InstanceNorm1d(hdim, affine=False)
        res_blocks = []
        for _ in range(num_layers - 2):
            res_blocks.append(ResTrans1DBlock(hdim, False))
        self.res = nn.Sequential(*res_blocks)
        self.conv2 = ConvLayer(hdim, x_dim, 3, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the input sequence.

        Args:
            x: Input tensor of shape (batch, channels, sequence_length).

        Returns:
            Transformed tensor with the same shape as the input.
        """
        out = self.relu(self.in1(self.conv1(x)))
        out = self.res(out)
        out = self.conv2(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, conv_param=None, downsample=None,
                 batchnorm: bool = False, bias: bool = False):
        """Initialize a residual block with optional downsampling.

        Args:
            in_dim: Number of input channels.
            out_dim: Number of output channels.
            conv_param: Optional tuple defining convolution parameters.
            downsample: Optional downsampling layer for the residual path.
            batchnorm: Whether to use batch normalization layers.
            bias: Whether to use bias terms in the convolutions.
        """
        super().__init__()

        self.conv1 = nn.Conv1d(in_dim, in_dim, 1, 1, 0, bias=bias)
        if conv_param is not None:
            self.conv2 = nn.Conv1d(in_dim, in_dim, conv_param[0], conv_param[1], conv_param[2], bias=bias)
        else:
            self.conv2 = nn.Conv1d(in_dim, in_dim, 3, 1, 1, bias=bias)

        self.conv3 = nn.Conv1d(in_dim, out_dim, 1, 1, 0, bias=bias)
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(in_dim)
            self.bn2 = nn.BatchNorm1d(in_dim)
            self.bn3 = nn.BatchNorm1d(out_dim)
            if downsample:
                self.bn4 = nn.BatchNorm1d(out_dim)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.batchnorm = batchnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Args:
            x: Input tensor of shape (batch, channels, sequence_length).

        Returns:
            Output tensor after residual processing.
        """
        residual = x

        out = self.conv1(x)
        if self.batchnorm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batchnorm:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.batchnorm:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            if self.batchnorm:
                residual = self.bn4(residual)

        out += residual
        out = self.relu(out)

        return out


class SeqEncoder(nn.Module):
    def __init__(self, x_dim: int, x_len: int, h_dim: int, z_dim: int, bias: bool,
                 num_layers: int, batch_norm: bool):
        """Initialize the sequence encoder.

        Args:
            x_dim: Number of input channels.
            x_len: Input sequence length.
            h_dim: Hidden channel dimension.
            z_dim: Latent dimension of the output.
            bias: Whether to use bias terms in convolutional layers.
            num_layers: Number of encoder layers.
            batch_norm: Whether to use batch normalization.
        """
        super().__init__()

        self.bias = bias
        self.batchnorm = batch_norm
        enc = [self._make_layer(x_dim, h_dim, (3, 1, 1))]
        in_dim = h_dim
        window_size = x_len
        for i in range(num_layers - 2):
            out_dim = h_dim * 2 ** i
            enc.append(self._make_layer(in_dim, out_dim, (3, 2, 1)))
            in_dim = out_dim
            window_size = math.floor((window_size + 2 - 3) / 2) + 1

        self.enc = nn.Sequential(*enc)
        self.final_layer = nn.Conv1d(in_dim, z_dim, int(window_size), 1, 0)

    def _make_layer(self, in_dim: int, out_dim: int, conv_param=None):
        """Create a residual layer for the encoder.

        Args:
            in_dim: Number of input channels.
            out_dim: Number of output channels.
            conv_param: Optional tuple defining convolution parameters.

        Returns:
            Residual block instance.
        """
        downsample = None
        if conv_param is not None:
            downsample = nn.Conv1d(in_dim, out_dim, conv_param[0], conv_param[1], conv_param[2], bias=self.bias)
        elif in_dim != out_dim:
            downsample = nn.Conv1d(in_dim, out_dim, 1, 1, 0, bias=self.bias)

        return ResBlock(in_dim, out_dim, conv_param, downsample=downsample,
                        batchnorm=self.batchnorm, bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an input sequence to a latent vector.

        Args:
            x: Input tensor of shape (batch, channels, sequence_length).

        Returns:
            Encoded tensor of shape (batch, z_dim).
        """
        z = self.enc(x)
        z = self.final_layer(z)
        return z.squeeze(-1)


def make_seq_nets(x_dim: int, config: dict):
    """Build a list of sequence encoders for each transform.

    Args:
        x_dim: Number of input channels.
        config: Configuration dictionary for encoder parameters.

    Returns:
        ModuleList containing sequence encoders.
    """
    enc_nlayers = config['enc_nlayers']
    enc_hdim = config['enc_hdim']
    z_dim = config['latent_dim']
    x_len = config['x_length']
    num_trans = config['num_trans']
    batch_norm = config['batch_norm']

    enc = nn.ModuleList([
        SeqEncoder(x_dim, x_len, enc_hdim, z_dim, config['enc_bias'], enc_nlayers, batch_norm)
        for _ in range(num_trans + 1)
    ])
    return enc


class NeutralAD(BaseModel):
    """NeutralAD model for transformation-based anomaly detection."""

    def __init__(self, ts_channels: int, seq_len: int, num_trans: int = 4, trans_type: str = 'residual',
                 enc_hdim: int = 32, enc_nlayers: int = 4, trans_nlayers: int = 4, latent_dim: int = 32,
                 batch_norm: bool = False, enc_bias: bool = False,
                 transform_families: Optional[Sequence[Sequence[Transform]]] = None,
                 use_invertible_transforms: bool = False,
                 use_group_transforms: bool = False,
                 use_freq_ortho_transforms: bool = False,
                 keep_base_transforms: bool = True,
                 invertible_cfg: Optional[Dict[str, Any]] = None,
                 group_cfg: Optional[Dict[str, Any]] = None,
                 freq_cfg: Optional[Dict[str, Any]] = None,
                 encoder_type: str = "base",
                 encoder_cfg: Optional[Dict[str, Any]] = None,
                 pooling: str = "mean",
                 proj_head: str = "linear",
                 proj_cfg: Optional[Dict[str, Any]] = None):
        """Initialize the NeutralAD model.

        Args:
            ts_channels: Number of input channels in the time series.
            seq_len: Length of the input sequence.
            num_trans: Number of base transforms to generate.
            trans_type: Type of transformation application ("forward", "mul", "residual").
            enc_hdim: Hidden dimension for the base encoder.
            enc_nlayers: Number of layers in the base encoder.
            trans_nlayers: Number of layers in the transform networks.
            latent_dim: Dimension of the latent representation.
            batch_norm: Whether to use batch normalization in the encoder.
            enc_bias: Whether to use bias terms in the encoder.
            transform_families: Optional additional transform families to include.
            use_invertible_transforms: Whether to add invertible transforms.
            use_group_transforms: Whether to add group transforms.
            use_freq_ortho_transforms: Whether to add frequency-orthogonal transforms.
            keep_base_transforms: Whether to keep the base transforms.
            invertible_cfg: Configuration for invertible transforms.
            group_cfg: Configuration for group transforms.
            freq_cfg: Configuration for frequency-orthogonal transforms.
            encoder_type: Encoder type ("base", "patchtst", "itransformer").
            encoder_cfg: Configuration for the encoder when using non-base encoders.
            pooling: Pooling strategy ("base", "mean", "meanmax", "attn").
            proj_head: Projection head type ("base", "linear", "mlp", "identity", "none").
            proj_cfg: Configuration for the projection head.
        """
        super().__init__()

        self.trans_type = trans_type
        self.z_dim = latent_dim
        self.encoder_type = encoder_type
        self.pooling = pooling
        transforms: List[Transform] = []
        if keep_base_transforms:
            if num_trans < 1:
                raise ValueError('num_trans must be >= 1 when keep_base_transforms is True')
            transforms.extend(
                [SeqTransformNet(ts_channels, ts_channels, trans_nlayers) for _ in range(num_trans)]
            )
        if use_invertible_transforms:
            invertible_cfg_data = invertible_cfg or default_invertible_cfg()
            transforms.extend(make_invertible_family(ts_channels, invertible_cfg_data))
        if use_group_transforms:
            group_cfg_data = group_cfg or default_group_cfg()
            transforms.extend(make_group_family(ts_channels, group_cfg_data))
        if use_freq_ortho_transforms:
            freq_cfg_data = freq_cfg or default_freq_cfg()
            transforms.extend(make_freq_family(ts_channels, seq_len, freq_cfg_data))
        if transform_families:
            for family in transform_families:
                transforms.extend(family)
        if not transforms:
            raise ValueError('At least one transform must be configured for NeutralAD')
        self.transform_bank = TransformBank(transforms)
        self.num_trans = len(self.transform_bank)
        self.proj = nn.Identity()
        self.pooler: Optional[nn.Module] = None
        if encoder_type == "base":
            config = dict(
                enc_nlayers=enc_nlayers,
                enc_hdim=enc_hdim,
                latent_dim=latent_dim,
                x_length=seq_len,
                num_trans=self.num_trans,
                batch_norm=batch_norm,
                enc_bias=enc_bias
            )
            self.enc = make_seq_nets(ts_channels, config)
        else:
            encoder_cfg = encoder_cfg or {}
            proj_cfg = proj_cfg or {}
            if pooling == "base":
                encoder_pooling = encoder_cfg.pop("pooling", "mean")
            else:
                encoder_pooling = "base"
            if encoder_type == "patchtst":
                encoder = PatchTSTEncoder(seq_len=seq_len, pooling=encoder_pooling, **encoder_cfg)
            elif encoder_type == "itransformer":
                encoder = ITransformerEncoder(seq_len=seq_len, pooling=encoder_pooling, **encoder_cfg)
            else:
                raise ValueError(f"Unknown encoder_type: {encoder_type}")
            self.enc = nn.ModuleList([encoder])
            if pooling != "base":
                self.pooler = self._make_pooler(pooling, encoder.d_model)
            proj_head_name = "linear" if proj_head == "base" else proj_head
            self.proj = self._make_proj_head(encoder.d_model, latent_dim, proj_head_name, proj_cfg)
            if proj_head_name in {"identity", "none"}:
                self.z_dim = encoder.d_model

    @staticmethod
    def _make_proj_head(input_dim: int, output_dim: int, proj_head: str, proj_cfg: Dict[str, Any]) -> nn.Module:
        """Build the projection head.

        Args:
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
            proj_head: Projection head name.
            proj_cfg: Configuration for the projection head.

        Returns:
            Projection head module.
        """
        if proj_head in {"identity", "none"}:
            return nn.Identity()
        if proj_head == "linear":
            return LinearHead(input_dim, output_dim, **proj_cfg)
        if proj_head == "mlp":
            hidden_dim = proj_cfg.get("hidden_dim")
            dropout = float(proj_cfg.get("dropout", 0.0))
            return MLPHead(input_dim, output_dim, hidden_dim=hidden_dim, dropout=dropout)
        raise ValueError(f"Unknown proj_head: {proj_head}")

    @staticmethod
    def _make_pooler(pooling: str, d_model: int) -> nn.Module:
        """Build the pooling module.

        Args:
            pooling: Pooling strategy name.
            d_model: Encoder model dimension for pooling.

        Returns:
            Pooling module.
        """
        if pooling == "mean":
            return MeanPool()
        if pooling == "meanmax":
            return MeanMaxPool(d_model)
        if pooling == "attn":
            return AttnPool(d_model)
        raise ValueError(f"Unknown pooling: {pooling}")

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute latent representations for transformed inputs.

        Args:
            inputs: Tuple containing a single input tensor of shape
                (batch, sequence_length, channels).

        Returns:
            Tensor of shape (batch, num_transforms + 1, latent_dim).
        """
        x, = inputs
        x = x.permute(0, 2, 1).contiguous()

        masks = self.transform_bank(x)
        if self.trans_type == 'forward':
            x_t = masks
        elif self.trans_type == 'mul':
            x_t = torch.sigmoid(masks) * x.unsqueeze(1)
        elif self.trans_type == 'residual':
            x_t = masks + x.unsqueeze(1)
        else:
            raise ValueError(f'Unknown trans_type: {self.trans_type}')

        x_cat = torch.cat([x.unsqueeze(1), x_t], 1)
        zs = self.enc[0](x_cat.reshape(-1, x.shape[1], x.shape[2]))
        if self.pooler is not None:
            zs = self.pooler(zs)
        zs = self.proj(zs)
        zs = zs.reshape(x.shape[0], self.num_trans + 1, self.z_dim)

        return zs


def _dcl_score(z: torch.Tensor, temperature: float, eval: bool) -> torch.Tensor:
    """Compute the DCL score for a batch of representations.

    Args:
        z: Latent tensor of shape (batch, num_transforms + 1, latent_dim).
        temperature: Temperature scaling factor.
        eval: Whether to return per-sample scores instead of a mean loss.

    Returns:
        Tensor containing either per-sample scores or the mean score.
    """
    z = F.normalize(z, p=2, dim=-1)
    z_ori = z[:, 0]
    z_trans = z[:, 1:]
    num_trans = z.shape[1]

    logits = torch.matmul(z, z.mT) / temperature
    diag_mask = torch.eye(num_trans, device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(diag_mask, float('-inf'))
    trans_logsumexp = torch.logsumexp(logits[:, 1:], dim=-1)

    pos_log = torch.sum(z_trans * z_ori.unsqueeze(1), -1) / temperature
    k_trans = num_trans - 1
    scale = 1 / abs(k_trans * math.log(1.0 / k_trans))

    loss_tensor = (trans_logsumexp - pos_log) * scale

    score = loss_tensor.sum(1)
    if eval:
        return score
    return score.mean()


def _eucdcl_score(z: torch.Tensor, temperature: float, eval: bool) -> torch.Tensor:
    """Compute the Euclidean DCL score for a batch of representations.

    Args:
        z: Latent tensor of shape (batch, num_transforms + 1, latent_dim).
        temperature: Temperature scaling factor.
        eval: Whether to return per-sample scores instead of a mean loss.

    Returns:
        Tensor containing either per-sample scores or the mean score.
    """
    num_trans = z.shape[1]
    logits = -torch.cdist(z, z) / temperature
    diag_mask = torch.eye(num_trans, device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(diag_mask, float('-inf'))
    trans_logsumexp = torch.logsumexp(logits[:, 1:], dim=-1)
    pos_log = logits[:, 1:, 0]

    k_trans = num_trans - 1
    scale = 1 / abs(k_trans * math.log(1.0 / k_trans))
    score = (-pos_log + trans_logsumexp) * scale
    score = score.sum(1)
    if eval:
        return score
    return score.mean()


class NeutralADLoss(Loss):
    def __init__(self, temperature: float = 0.1, use_euclidean: bool = False):
        """Initialize the NeutralAD loss.

        Args:
            temperature: Temperature scaling factor for contrastive scoring.
            use_euclidean: Whether to use Euclidean distance-based scoring.
        """
        super().__init__()
        self.temperature = temperature
        self.use_euclidean = use_euclidean

    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...] = None,
                eval: bool = False, *args, **kwargs) -> torch.Tensor:
        """Compute the loss or anomaly score.

        Args:
            predictions: Tuple containing latent representations.
            targets: Optional targets (unused).
            eval: Whether to return per-sample scores instead of a mean loss.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Loss value or per-sample scores.
        """
        z, = predictions
        if self.use_euclidean:
            return _eucdcl_score(z, self.temperature, eval=eval)
        return _dcl_score(z, self.temperature, eval=eval)


class NeutralADAnomalyDetector(AnomalyDetector):
    def __init__(self, model: NeutralAD, loss: NeutralADLoss):
        """Initialize the anomaly detector wrapper.

        Args:
            model: NeutralAD model instance.
            loss: NeutralAD loss instance.
        """
        super().__init__()
        self.model = model
        self.loss = loss

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute anomaly scores for online inference.

        Args:
            inputs: Tuple containing a single input tensor.

        Returns:
            Anomaly scores per sample.
        """
        with torch.inference_mode():
            z = pack_tuple(self.model(inputs))

        return self.loss(z, eval=True)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute anomaly scores for offline inference."""
        raise NotImplementedError

    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs) -> None:
        """Fit the anomaly detector (not implemented)."""
        pass

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Format online targets to match expected output.

        Args:
            targets: Tuple containing label tensors.

        Returns:
            Tensor containing the last label for each sample.
        """
        label, = targets
        return label[:, -1]
