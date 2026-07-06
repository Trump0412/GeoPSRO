from __future__ import annotations

from geopsro4d.data.formatters import psro_prompt, sft_prompt
from geopsro4d.data.schema import GeoSample


GEO_START = "<geo>"
GEO_END = "</geo>"


def build_prompt(sample: GeoSample, *, stage: str) -> str:
    if stage == "rft":
        return psro_prompt(sample)
    return sft_prompt(sample)


def add_geo_special_tokens(tokenizer) -> int:
    added = tokenizer.add_special_tokens({"additional_special_tokens": [GEO_START, GEO_END]})
    return int(added)
