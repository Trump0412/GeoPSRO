from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from geopsro4d.geometry.vggt_cache import zero_geometry
from geopsro4d.model.qwen_vggt_wrapper import QwenVGGTWrapper, TinyQAModel
from geopsro4d.utils.io import ensure_dir, write_json


def make_smoke_cache(sample_id: str = "smoke", num_frames: int = 4) -> dict[str, Any]:
    cache = zero_geometry(sample_id, list(range(num_frames)))
    cache["geometry_valid"] = True
    cache["depth"] = torch.rand(num_frames, 1, 16, 16)
    cache["point_map"] = torch.rand(num_frames, 3, 16, 16)
    cache["confidence"] = torch.rand(num_frames, 16, 16)
    return cache


def make_smoke_model(hidden_size: int = 64, num_geo_tokens: int = 8) -> tuple[TinyQAModel, QwenVGGTWrapper]:
    base = TinyQAModel(hidden_size=hidden_size, vocab_size=128)
    wrapper = QwenVGGTWrapper(base, llm_dim=hidden_size, adapter_dim=32, num_geo_tokens=num_geo_tokens)
    return base, wrapper


def save_adapter(output: str | Path, wrapper: QwenVGGTWrapper, metrics: dict[str, Any]) -> None:
    output = ensure_dir(output)
    torch.save(
        {
            "geo_tokenizer": wrapper.geo_tokenizer.state_dict(),
            "geo_resampler": wrapper.geo_resampler.state_dict(),
            "geo_projector": wrapper.geo_projector.state_dict(),
            "geometry_gate": wrapper.geometry_gate.state_dict(),
            "null_geo_embeds": wrapper.null_geo_embeds.detach().cpu(),
            "metrics": metrics,
        },
        output / "geo_adapter.pt",
    )
    write_json(output / "metrics.json", metrics)
