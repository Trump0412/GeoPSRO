from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from geopsro4d.data.frame_sampler import VIDEO_EXTS, sample_frame_selection, sample_video_frames
from geopsro4d.data.schema import normalize_sample
from geopsro4d.geometry.vggt_cache import VGGTCache, zero_geometry
from geopsro4d.utils.io import iter_jsonl, write_jsonl
from geopsro4d.utils.logging import get_logger


LOGGER = get_logger(__name__)


class VGGTExtractor:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        source_path: str | Path | None = None,
        device: str = "cuda",
        cache_resolution: int = 16,
        cache_dtype: str = "float16",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.source_path = Path(source_path) if source_path else None
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.cache_resolution = int(cache_resolution)
        self.cache_dtype = cache_dtype
        self._model = None

    def load_model(self):
        if self._model is not None:
            return self._model
        if self.source_path:
            sys.path.insert(0, str(self.source_path))
        from vggt.models.vggt import VGGT

        path = Path(self.model_name_or_path)
        if path.exists():
            model = VGGT()
            weight = _resolve_vggt_weight(path)
            state = torch.load(weight, map_location="cpu") if weight.suffix != ".safetensors" else _load_safetensors(weight)
            model.load_state_dict(state.get("model", state) if isinstance(state, dict) else state, strict=False)
        else:
            model = VGGT.from_pretrained(self.model_name_or_path)
        self._model = model.eval().to(self.device)
        return self._model

    @torch.inference_mode()
    def extract(self, sample_id: str, frame_paths: list[str], frame_indices: list[int]) -> dict[str, Any]:
        model = self.load_model()
        if self.source_path:
            sys.path.insert(0, str(self.source_path))
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        images = load_and_preprocess_images(frame_paths).to(self.device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            pred = model(images)
        pose_enc = pred.get("pose_enc")
        extrinsic, intrinsic = (None, None)
        if pose_enc is not None:
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            extrinsic = extrinsic.squeeze(0).detach().cpu().float()
            intrinsic = intrinsic.squeeze(0).detach().cpu().float()
        depth = _compact_tensor(_chw(pred.get("depth")), self.cache_resolution, self.cache_dtype)
        point_map = _compact_tensor(_chw(pred.get("world_points")), self.cache_resolution, self.cache_dtype)
        confidence = _compact_tensor(_chw(pred.get("world_points_conf")), self.cache_resolution, self.cache_dtype)
        height, width = images.shape[-2:]
        return {
            "sample_id": sample_id,
            "frame_indices": frame_indices,
            "image_size": (int(width), int(height)),
            "camera_intrinsics": intrinsic,
            "camera_extrinsics": extrinsic,
            "depth": depth,
            "point_map": point_map,
            "tracks": None,
            "visibility": None,
            "confidence": confidence,
            "features": None,
            "geometry_valid": True,
            "nan_count": _nan_count([depth, point_map, confidence]),
            "cache_profile": f"compact_pool{self.cache_resolution}_{self.cache_dtype}",
        }


def extract_dataset(
    *,
    dataset_json: Path,
    cache_root: Path,
    num_frames: int,
    model_name_or_path: str,
    source_path: Path | None,
    overwrite: bool,
    device: str,
    max_samples: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
    log_every: int = 25,
    cache_resolution: int = 16,
    cache_dtype: str = "float16",
) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = VGGTCache(cache_root)
    extractor = VGGTExtractor(
        model_name_or_path,
        source_path=source_path,
        device=device,
        cache_resolution=cache_resolution,
        cache_dtype=cache_dtype,
    )
    logs = []
    processed = 0
    log_path = cache_root / (
        f"vggt_cache_log.shard{shard_index:03d}-of-{num_shards:03d}.jsonl"
        if num_shards > 1
        else "vggt_cache_log.jsonl"
    )
    for row_idx, row in enumerate(iter_jsonl(dataset_json)):
        if row_idx % num_shards != shard_index:
            continue
        if max_samples is not None and processed >= max_samples:
            break
        sample = normalize_sample(row, default_dataset=dataset_json.stem)
        if cache.exists(sample.sample_id) and not overwrite:
            continue
        try:
            media = list(sample.media_paths)
            if len(media) == 1 and Path(media[0]).suffix.lower() in VIDEO_EXTS:
                selection = sample_video_frames(media[0], cache_root / "_frames" / sample.sample_id, num_frames)
            else:
                selection = sample_frame_selection(media, num_frames)
            data = extractor.extract(sample.sample_id, selection.paths, selection.indices)
        except Exception as exc:  # noqa: BLE001 - cache extraction should mark failed samples
            LOGGER.exception("VGGT failed for %s", sample.sample_id)
            selection = sample_frame_selection(sample.media_paths, num_frames)
            data = zero_geometry(sample.sample_id, selection.indices)
            data["error"] = str(exc)
        cache.save(sample.sample_id, data)
        logs.append(
            {
                "sample_id": sample.sample_id,
                "frame_indices": data.get("frame_indices"),
                "geometry_valid": data.get("geometry_valid", False),
                "nan_count": data.get("nan_count", 0),
                "cache_profile": data.get("cache_profile", "unknown"),
                "cache_path": str(cache.path_for(sample.sample_id)),
            }
        )
        processed += 1
        if log_every > 0 and processed % log_every == 0:
            write_jsonl(log_path, logs)
    write_jsonl(log_path, logs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json", type=Path, required=True)
    parser.add_argument("--cache_root", type=Path, required=True)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--source_path", type=Path)
    parser.add_argument("--overwrite", default="false")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--cache_resolution", type=int, default=16)
    parser.add_argument("--cache_dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()
    extract_dataset(
        dataset_json=args.dataset_json,
        cache_root=args.cache_root,
        num_frames=args.num_frames,
        model_name_or_path=args.model_name_or_path,
        source_path=args.source_path,
        overwrite=args.overwrite.lower() in {"1", "true", "yes"},
        device=args.device,
        max_samples=args.max_samples,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        log_every=args.log_every,
        cache_resolution=args.cache_resolution,
        cache_dtype=args.cache_dtype,
    )


def _resolve_vggt_weight(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("model.safetensors", "model.pt", "pytorch_model.bin"):
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No VGGT weights found under {path}")


def _load_safetensors(path: Path):
    from safetensors.torch import load_file

    return load_file(str(path))


def _squeeze_batch(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.squeeze(0).detach().cpu().float()


def _chw(tensor: torch.Tensor | None) -> torch.Tensor | None:
    tensor = _squeeze_batch(tensor)
    if tensor is None:
        return None
    if tensor.ndim == 3:
        return tensor.unsqueeze(1).contiguous()
    if tensor.ndim == 4 and tensor.shape[-1] in {1, 2, 3, 4}:
        return tensor.permute(0, 3, 1, 2).contiguous()
    return tensor


def _compact_tensor(tensor: torch.Tensor | None, resolution: int, dtype: str) -> torch.Tensor | None:
    if tensor is None:
        return None
    tensor = tensor.detach().cpu().float()
    if resolution > 0 and tensor.ndim == 4:
        tensor = F.adaptive_avg_pool2d(tensor, (resolution, resolution))
    if dtype == "float16":
        return tensor.half()
    if dtype == "bfloat16":
        return tensor.bfloat16()
    return tensor


def _nan_count(tensors: list[torch.Tensor | None]) -> int:
    return int(sum(torch.isnan(t).sum().item() for t in tensors if t is not None and t.is_floating_point()))


if __name__ == "__main__":
    main()
