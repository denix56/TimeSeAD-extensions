# Local Neural Transformations (LNT) - a self-supervised method for
# anomalous region detection in time series
# Copyright (c) 2021 Robert Bosch GmbH
# Adapted for TimeSeAD framework
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

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from timesead.models.common import AnomalyDetector
from timesead.models import BaseModel
from timesead.optim.loss import Loss
from timesead.utils.utils import pack_tuple


class SimpleEncoder(nn.Module):
    """Simple CNN encoder for time series."""

    def __init__(self, input_channels: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, seq_len)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.conv3(x)
        return x  # shape: (batch, latent_dim, seq_len)


class TransformationNetwork(nn.Module):
    """Learns a single transformation of the latent space."""

    def __init__(self, latent_dim: int, hidden_dim: int, transformation_type: str = 'residual'):
        super().__init__()
        self.transformation_type = transformation_type

        # Simple MLP for transformation
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Activation based on transformation type
        if transformation_type == 'residual':
            self.activation = nn.Tanh()
        elif transformation_type == 'multiplicative':
            self.activation = nn.Sigmoid()
        else:  # feedforward
            self.activation = nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z shape: (batch, latent_dim)
        out = self.net(z)
        out = self.activation(out)

        if self.transformation_type == 'residual':
            return z + out
        elif self.transformation_type == 'multiplicative':
            return z * out
        else:  # feedforward
            return out


class NeuralTransformationLearner(nn.Module):
    """Multiple learned transformations for contrastive learning."""

    def __init__(self, latent_dim: int, hidden_dim: int, n_transformations: int,
                 transformation_type: str = 'residual'):
        super().__init__()
        self.n_transformations = n_transformations
        self.transformations = nn.ModuleList([
            TransformationNetwork(latent_dim, hidden_dim, transformation_type)
            for _ in range(n_transformations)
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z shape: (batch * seq_len, latent_dim)
        # Returns: (batch * seq_len, n_transformations, latent_dim)
        transformed = torch.stack([trans(z) for trans in self.transformations], dim=1)
        return transformed


class LNT(BaseModel):
    """Local Neural Transformations model for TimeSeAD."""

    def __init__(
        self,
        ts_channels: int,
        seq_len: int,
        n_transformations: int = 10,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        transformation_type: str = 'residual',
        encoder_type: str = 'cnn'
    ):
        super().__init__()

        self.ts_channels = ts_channels
        self.seq_len = seq_len
        self.n_transformations = n_transformations
        self.latent_dim = latent_dim

        # Encoder: maps time series to latent space
        if encoder_type == 'cnn':
            self.encoder = SimpleEncoder(ts_channels, hidden_dim, latent_dim)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        # Transformation learner
        self.transformation_learner = NeuralTransformationLearner(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_transformations=n_transformations,
            transformation_type=transformation_type
        )

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, = inputs
        # x shape: (batch, seq_len, channels)

        batch_size = x.shape[0]

        # Permute to (batch, channels, seq_len) for Conv1d
        x = x.permute(0, 2, 1).float()

        # Encode to latent space
        z = self.encoder(x)  # (batch, latent_dim, seq_len)

        # Permute to (batch, seq_len, latent_dim)
        z = z.permute(0, 2, 1)

        # Flatten for transformation
        z_flat = z.reshape(-1, self.latent_dim)  # (batch * seq_len, latent_dim)

        # Apply transformations
        z_transformed = self.transformation_learner(z_flat)  # (batch * seq_len, n_trans, latent_dim)

        # Reshape back
        z_transformed = z_transformed.reshape(batch_size, self.seq_len, self.n_transformations, self.latent_dim)

        # z: (batch, seq_len, latent_dim)
        # z_transformed: (batch, seq_len, n_trans, latent_dim)
        return z, z_transformed


class LNTLoss(Loss):
    """Deterministic Contrastive Loss for LNT."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    @staticmethod
    def cosine_similarity(x: torch.Tensor, x_: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between tensors."""
        x = F.normalize(x, p=2, dim=-1)
        x_ = F.normalize(x_, p=2, dim=-1)
        return torch.matmul(x, x_.transpose(-2, -1))

    def deterministic_contrastive_loss(
        self,
        z: torch.Tensor,
        z_transformed: torch.Tensor,
        average: bool = True
    ) -> torch.Tensor:
        """
        Compute DCL loss per timestep.

        Args:
            z: Original embeddings (batch, seq_len, latent_dim)
            z_transformed: Transformed embeddings (batch, seq_len, n_trans, latent_dim)
            average: Whether to average over batch and sequence

        Returns:
            Loss tensor
        """
        batch_size, seq_len, n_trans, latent_dim = z_transformed.shape

        # Flatten batch and seq dimensions
        z_flat = z.reshape(-1, latent_dim)  # (batch*seq_len, latent_dim)
        z_trans_flat = z_transformed.reshape(-1, n_trans, latent_dim)  # (batch*seq_len, n_trans, latent_dim)

        # Concatenate original and transformed: (batch*seq_len, n_trans+1, latent_dim)
        z_all = torch.cat([z_flat.unsqueeze(1), z_trans_flat], dim=1)

        # Compute similarity matrix: (batch*seq_len, n_trans+1, n_trans+1)
        sim = self.cosine_similarity(z_all, z_all) / self.temperature

        # Positive pairs: original vs each transformation
        sim_pos = sim[:, 1:, 0]  # (batch*seq_len, n_trans)

        # Negative pairs: exclude self-similarity (diagonal)
        n_all = n_trans + 1
        mask = (torch.ones(n_all, n_all, device=z.device) - torch.eye(n_all, device=z.device)).bool()
        sim_all = torch.masked_select(sim, mask).view(-1, n_all, n_all - 1)[:, 1:, :]  # (batch*seq_len, n_trans, n_all-1)

        # Log-sum-exp for numerical stability
        normalization_const = torch.logsumexp(sim_all, dim=-1)  # (batch*seq_len, n_trans)

        # DCL loss
        loss = -torch.sum(sim_pos - normalization_const, dim=-1)  # (batch*seq_len,)

        # Reshape to (batch, seq_len)
        loss = loss.reshape(batch_size, seq_len)

        if average:
            return torch.mean(loss)
        else:
            # Return per-timestep loss (mean over sequence for each sample)
            return torch.mean(loss, dim=1)

    def forward(
        self,
        predictions: Tuple[torch.Tensor, ...],
        targets: Tuple[torch.Tensor, ...] = None,
        eval: bool = False,
        *args,
        **kwargs
    ) -> torch.Tensor:
        z, z_transformed = predictions

        if eval:
            # Return per-sample anomaly scores
            return self.deterministic_contrastive_loss(z, z_transformed, average=False)
        else:
            # Return scalar loss for training
            return self.deterministic_contrastive_loss(z, z_transformed, average=True)


class LNTAnomalyDetector(AnomalyDetector):
    """Anomaly detector wrapper for LNT."""

    def __init__(self, model: LNT, loss: LNTLoss):
        super().__init__()
        self.model = model
        self.loss = loss

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
