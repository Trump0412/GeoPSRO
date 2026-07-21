from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

from geopsro4d.geometry.geo_projector import GeoProjector
from geopsro4d.geometry.geo_resampler import GeoResampler
from geopsro4d.geometry.geo_tokenizer import GeoTokenizer
from geopsro4d.geometry.geometry_gate import GeometryGate
from geopsro4d.geometry.vggt_cache import VGGTCache, zero_geometry


GEOMETRY_NORMAL = "normal"
GEOMETRY_ZERO = "zero"
GEOMETRY_DROP = "drop"
GEOMETRY_SHUFFLE = "shuffle"
GEOMETRY_DEPTH_ONLY = "depth_only"
GEOMETRY_DEPTH_CAMERA = "depth_camera"
GEOMETRY_FULL = "full"
GEOMETRY_MODES = {
    GEOMETRY_NORMAL,
    GEOMETRY_ZERO,
    GEOMETRY_DROP,
    GEOMETRY_SHUFFLE,
    GEOMETRY_DEPTH_ONLY,
    GEOMETRY_DEPTH_CAMERA,
    GEOMETRY_FULL,
}


class QwenVGGTWrapper(nn.Module):
    """Geometry-token wrapper around a Qwen-style causal LM.

    The wrapper exposes trainable GeoTokenizer/Resampler/Projector/Gate modules.
    For real Qwen training, call `geometry_inputs_embeds` and concatenate the
    returned embeddings into the model's `inputs_embeds` sequence around `<geo>`
    markers. The training entrypoints keep this wrapper narrow so it can be used
    by both local smoke models and full Qwen3-VL jobs.
    """

    def __init__(
        self,
        base_model: nn.Module | None,
        *,
        llm_dim: int = 2560,
        raw_geo_dim: int = 32,
        adapter_dim: int = 256,
        num_geo_tokens: int = 64,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.geo_tokenizer = GeoTokenizer(feature_dim=raw_geo_dim)
        self.geo_resampler = GeoResampler(input_dim=raw_geo_dim, model_dim=adapter_dim, num_geo_tokens=num_geo_tokens)
        self.geo_projector = GeoProjector(model_dim=adapter_dim, llm_dim=llm_dim)
        self.geometry_gate = GeometryGate(init_value=0.2)
        self.null_geo_embeds = nn.Parameter(torch.empty(1, num_geo_tokens, llm_dim))
        nn.init.normal_(self.null_geo_embeds, mean=0.0, std=1e-4)
        self.num_geo_tokens = num_geo_tokens
        self.llm_dim = llm_dim

    def geometry_inputs_embeds(self, cache_data: dict[str, Any], *, mode: str = GEOMETRY_NORMAL) -> torch.Tensor:
        if mode in {GEOMETRY_NORMAL, GEOMETRY_SHUFFLE}:
            mode = GEOMETRY_FULL
        if mode not in GEOMETRY_MODES:
            raise ValueError(f"Unknown geometry mode: {mode}")
        device = next(self.geo_projector.parameters()).device
        if mode == GEOMETRY_ZERO:
            return torch.zeros_like(self.null_geo_embeds.to(device))
        if mode == GEOMETRY_DROP:
            return self.geometry_gate(self.null_geo_embeds.to(device))
        features, mask = self.geo_tokenizer(cache_data, mode=mode)
        features = features.to(device)
        mask = mask.to(device)
        tokens = self.geo_resampler(features.unsqueeze(0), mask.unsqueeze(0))
        projected = self.geo_projector(tokens)
        return self.geometry_gate(projected)

    def forward(self, cache_data: dict[str, Any], mode: str = GEOMETRY_NORMAL) -> torch.Tensor:
        return self.geometry_inputs_embeds(cache_data, mode=mode)

    def load_geometry(
        self,
        sample_id: str,
        *,
        cache_root: str | Path | None,
        frame_indices: list[int] | None = None,
        mode: str = GEOMETRY_NORMAL,
        shuffle_pool: list[str] | None = None,
    ) -> dict[str, Any]:
        frame_indices = frame_indices or list(range(8))
        if mode in {GEOMETRY_ZERO, GEOMETRY_DROP} or cache_root is None:
            return zero_geometry(sample_id, frame_indices)
        cache = VGGTCache(cache_root)
        load_id = sample_id
        if mode == GEOMETRY_SHUFFLE and shuffle_pool:
            choices = [x for x in shuffle_pool if x != sample_id]
            if choices:
                load_id = random.choice(choices)
        if not cache.exists(load_id):
            return zero_geometry(sample_id, frame_indices)
        data = cache.load(load_id)
        data["geometry_source_sample_id"] = load_id
        return data

    def trainable_geo_parameters(self):
        modules = [self.geo_tokenizer, self.geo_resampler, self.geo_projector, self.geometry_gate]
        for module in modules:
            yield from module.parameters()
        yield self.null_geo_embeds


class TinyQAModel(nn.Module):
    """Small language-like model used only for smoke tests."""

    def __init__(self, hidden_size: int = 64, vocab_size: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, extra_embeds: torch.Tensor | None = None, labels: torch.Tensor | None = None):
        hidden = self.embedding(input_ids)
        if extra_embeds is not None:
            pooled = extra_embeds.mean(dim=1, keepdim=True).to(hidden.dtype)
            hidden = hidden + pooled
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return {"loss": loss, "logits": logits}
