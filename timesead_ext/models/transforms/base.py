from typing import Iterable, List

import torch
import torch.nn as nn


class Transform(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class TransformBank(nn.Module):
    def __init__(self, transforms: Iterable[Transform]):
        super().__init__()
        self.transforms = nn.ModuleList(list(transforms))

    def __len__(self) -> int:
        return len(self.transforms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs: List[torch.Tensor] = [transform(x) for transform in self.transforms]
        return torch.stack(outputs, dim=1)
