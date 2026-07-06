from __future__ import annotations

import torch
from torch import nn


class GeoProjector(nn.Module):
    def __init__(self, model_dim: int = 256, llm_dim: int = 2560) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )

    def forward(self, geo_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(geo_tokens)
