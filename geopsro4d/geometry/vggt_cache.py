from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class VGGTCache:
    def __init__(self, cache_root: str | Path) -> None:
        self.cache_root = Path(cache_root)

    def path_for(self, sample_id: str) -> Path:
        safe = sample_id.replace("/", "__")
        return self.cache_root / f"{safe}.pt"

    def exists(self, sample_id: str) -> bool:
        return self.path_for(sample_id).exists()

    def load(self, sample_id: str) -> dict[str, Any]:
        return torch.load(self.path_for(sample_id), map_location="cpu")

    def save(self, sample_id: str, data: dict[str, Any]) -> None:
        path = self.path_for(sample_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)


def zero_geometry(sample_id: str, frame_indices: list[int], *, image_size: tuple[int, int] = (518, 518)) -> dict[str, Any]:
    frame_count = len(frame_indices)
    return {
        "sample_id": sample_id,
        "frame_indices": frame_indices,
        "image_size": image_size,
        "camera_intrinsics": torch.zeros(frame_count, 3, 3),
        "camera_extrinsics": torch.zeros(frame_count, 4, 4),
        "depth": torch.zeros(frame_count, 1, 16, 16),
        "point_map": torch.zeros(frame_count, 3, 16, 16),
        "tracks": torch.zeros(0, frame_count, 2),
        "visibility": torch.zeros(0, frame_count),
        "confidence": torch.zeros(0, frame_count),
        "features": None,
        "geometry_valid": False,
    }
