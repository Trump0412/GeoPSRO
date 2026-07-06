from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from geopsro4d.data.frame_sampler import sample_frame_selection
from geopsro4d.data.schema import GeoSample


@dataclass(frozen=True)
class CollatedSample:
    sample: GeoSample
    frame_paths: list[str]
    frame_indices: list[int]
    geometry_mode: str
    cache_root: str | None


class GeoCollator:
    def __init__(
        self,
        *,
        num_frames: int = 8,
        geometry_on_ratio: float = 1.0,
        cache_root: str | Path | None = None,
        frame_mode: str = "uniform",
    ) -> None:
        self.num_frames = num_frames
        self.geometry_on_ratio = geometry_on_ratio
        self.cache_root = str(cache_root) if cache_root is not None else None
        self.frame_mode = frame_mode

    def __call__(self, samples: Iterable[GeoSample]) -> list[CollatedSample]:
        batch = []
        for sample in samples:
            frames = sample_frame_selection(sample.media_paths, self.num_frames, mode=self.frame_mode)
            geometry_mode = "normal" if random.random() < self.geometry_on_ratio else "zero"
            batch.append(
                CollatedSample(
                    sample=sample,
                    frame_paths=frames.paths,
                    frame_indices=frames.indices,
                    geometry_mode=geometry_mode,
                    cache_root=self.cache_root,
                )
            )
        return batch
