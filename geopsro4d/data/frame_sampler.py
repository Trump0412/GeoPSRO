from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class FrameSelection:
    paths: list[str]
    indices: list[int]


def sample_frames(media_paths: list[str] | tuple[str, ...], num_frames: int, mode: str = "uniform") -> list[str]:
    """Return exactly `num_frames` visual inputs whenever possible.

    For multi-image inputs, order is preserved. Videos should be extracted by
    `sample_video_frames` before calling this function.
    """

    return sample_frame_selection(media_paths, num_frames, mode=mode).paths


def sample_frame_selection(media_paths: list[str] | tuple[str, ...], num_frames: int, mode: str = "uniform") -> FrameSelection:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    paths = [str(p) for p in media_paths]
    if not paths:
        raise ValueError("media_paths is empty")
    if len(paths) == 1 and Path(paths[0]).suffix.lower() in VIDEO_EXTS:
        return FrameSelection(paths=paths * num_frames, indices=list(range(num_frames)))
    if len(paths) >= num_frames:
        if mode == "first":
            idx = list(range(num_frames))
        elif mode == "last":
            idx = list(range(len(paths) - num_frames, len(paths)))
        else:
            idx = _uniform_indices(len(paths), num_frames)
        return FrameSelection(paths=[paths[i] for i in idx], indices=idx)
    idx = list(range(len(paths)))
    while len(idx) < num_frames:
        idx.append(idx[-1])
    return FrameSelection(paths=[paths[i] for i in idx], indices=idx)


def sample_video_frames(video_path: str | Path, output_dir: str | Path, num_frames: int, mode: str = "uniform") -> FrameSelection:
    """Extract uniformly sampled video frames to `output_dir` using OpenCV if available."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for video frame extraction") from exc
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _uniform_indices(frame_count, num_frames) if mode == "uniform" else list(range(min(frame_count, num_frames)))
    paths: list[str] = []
    for out_idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        path = output / f"frame_{out_idx:04d}.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(str(path))
    cap.release()
    if not paths:
        raise RuntimeError(f"No frames extracted from {video_path}")
    while len(paths) < num_frames:
        paths.append(paths[-1])
        indices.append(indices[-1])
    return FrameSelection(paths=paths[:num_frames], indices=indices[:num_frames])


def _uniform_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        raise ValueError("length must be positive")
    if count == 1:
        return [0]
    return [round(i * (length - 1) / (count - 1)) for i in range(count)]
