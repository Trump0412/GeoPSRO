from __future__ import annotations

import torch
from torch import nn


class GeometryGate(nn.Module):
    def __init__(self, init_value: float = 0.2, vector_dim: int | None = None) -> None:
        super().__init__()
        if vector_dim is None:
            init = torch.tensor(float(init_value))
        else:
            init = torch.full((vector_dim,), float(init_value))
        self.logit = nn.Parameter(torch.logit(init.clamp(1e-4, 1 - 1e-4)))

    def value(self) -> torch.Tensor:
        return torch.sigmoid(self.logit)

    def forward(self, geo_tokens: torch.Tensor) -> torch.Tensor:
        return geo_tokens * self.value()
