from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class GeoTokenizer(nn.Module):
    """Convert cached VGGT outputs into raw geometry feature tokens."""

    def __init__(self, feature_dim: int = 32, pool_size: int = 8) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.pool_size = pool_size

    def forward(self, vggt_cache: dict[str, Any], mode: str = "full") -> tuple[torch.Tensor, torch.Tensor]:
        tokens: list[torch.Tensor] = []
        if mode not in {"zero", "drop"}:
            if mode in {"full", "depth_camera", "normal"}:
                tokens.extend(self._camera_tokens(vggt_cache))
            if mode in {"full", "depth_camera", "depth_only", "normal"}:
                tokens.extend(self._frame_geometry_tokens(vggt_cache, include_points=mode not in {"depth_only"}))
            if mode in {"full", "normal"}:
                tokens.extend(self._track_tokens(vggt_cache))
        if not tokens:
            tokens = [torch.zeros(self.feature_dim)]
        out = torch.stack([_fit_dim(t.float(), self.feature_dim) for t in tokens], dim=0)
        mask = torch.ones(out.shape[0], dtype=torch.bool)
        if mode in {"zero", "drop"}:
            out.zero_()
            mask.zero_()
        return out, mask

    def _camera_tokens(self, cache: dict[str, Any]) -> list[torch.Tensor]:
        intr = cache.get("camera_intrinsics")
        extr = cache.get("camera_extrinsics")
        if intr is None and extr is None:
            return []
        frame_count = int(intr.shape[0] if intr is not None else extr.shape[0])
        tokens = []
        for idx in range(frame_count):
            parts = []
            if intr is not None:
                parts.append(intr[idx].reshape(-1))
            if extr is not None:
                parts.append(extr[idx].reshape(-1))
                if frame_count > 0:
                    parts.append((extr[idx] - extr[0]).reshape(-1))
            tokens.append(torch.cat(parts) if parts else torch.zeros(1))
        return tokens

    def _frame_geometry_tokens(self, cache: dict[str, Any], *, include_points: bool) -> list[torch.Tensor]:
        depth = cache.get("depth")
        point_map = cache.get("point_map") if include_points else None
        conf = cache.get("confidence")
        if depth is None and point_map is None:
            return []
        frame_count = int((depth if depth is not None else point_map).shape[0])
        tokens = []
        for frame in range(frame_count):
            parts = []
            if depth is not None:
                pooled = F.adaptive_avg_pool2d(depth[frame].float().unsqueeze(0), (self.pool_size, self.pool_size)).flatten()
                parts.extend([pooled.mean().view(1), pooled.std().view(1), pooled])
            if point_map is not None:
                pooled_xyz = F.adaptive_avg_pool2d(point_map[frame].float().unsqueeze(0), (self.pool_size, self.pool_size)).flatten()
                parts.extend([pooled_xyz.mean().view(1), pooled_xyz.std().view(1), pooled_xyz])
            if conf is not None and conf.ndim >= 3:
                parts.append(conf[frame].float().mean().view(1))
            tokens.append(torch.cat(parts) if parts else torch.zeros(1))
        return tokens

    def _track_tokens(self, cache: dict[str, Any]) -> list[torch.Tensor]:
        tracks = cache.get("tracks")
        visibility = cache.get("visibility")
        confidence = cache.get("confidence")
        if tracks is None or not torch.is_tensor(tracks) or tracks.numel() == 0:
            return []
        if tracks.ndim < 3:
            return []
        tokens = []
        for idx in range(tracks.shape[0]):
            xy = tracks[idx].float()
            start = xy[0].reshape(-1)
            end = xy[-1].reshape(-1)
            displacement = (xy[-1] - xy[0]).reshape(-1)
            parts = [start, end, xy.mean(dim=0).reshape(-1), displacement]
            if visibility is not None and visibility.ndim >= 2:
                parts.append(visibility[idx].float().mean().view(1))
            if confidence is not None and confidence.ndim >= 2:
                parts.append(confidence[idx].float().mean().view(1))
            tokens.append(torch.cat(parts))
        return tokens


def _fit_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.numel() >= dim:
        return x[:dim]
    return F.pad(x, (0, dim - x.numel()))
