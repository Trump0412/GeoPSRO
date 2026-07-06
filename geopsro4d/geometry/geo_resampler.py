from __future__ import annotations

import torch
from torch import nn


class GeoResampler(nn.Module):
    def __init__(self, input_dim: int = 32, model_dim: int = 256, num_geo_tokens: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.num_geo_tokens = num_geo_tokens
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.queries = nn.Parameter(torch.randn(num_geo_tokens, model_dim) * 0.02)
        self.attn = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, geo_features: torch.Tensor, geo_mask: torch.Tensor) -> torch.Tensor:
        if geo_features.ndim == 2:
            geo_features = geo_features.unsqueeze(0)
            geo_mask = geo_mask.unsqueeze(0)
        keys = self.input_proj(geo_features)
        queries = self.queries.unsqueeze(0).expand(keys.shape[0], -1, -1)
        key_padding_mask = ~geo_mask.bool()
        out, _ = self.attn(queries, keys, keys, key_padding_mask=key_padding_mask)
        return self.norm(out)
