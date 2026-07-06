from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GeoSample:
    sample_id: str
    dataset: str
    media_paths: tuple[str, ...]
    question: str
    answer: str
    choices: tuple[str, ...] | None = None
    task_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_sample(row: dict[str, Any], *, default_dataset: str = "unknown") -> GeoSample:
    sample_id = str(row.get("sample_id") or row.get("id") or row.get("clip_id") or row.get("qid") or "")
    if not sample_id:
        raise ValueError("sample requires sample_id/id/clip_id/qid")
    media_paths = row.get("media_paths") or row.get("frame_paths") or row.get("images") or row.get("video") or []
    if isinstance(media_paths, str):
        media_paths = [media_paths]
    if not media_paths:
        raise ValueError(f"sample {sample_id} has no media_paths")
    question = str(row.get("question") or row.get("prompt") or "")
    answer = str(row.get("answer") or row.get("target") or row.get("label") or "")
    if not question or not answer:
        raise ValueError(f"sample {sample_id} requires question and answer")
    choices = row.get("choices")
    if choices is not None:
        choices = tuple(str(x) for x in choices)
    metadata = dict(row.get("metadata") or {})
    for key in ("scene_id", "frame_indices", "timestamps_s", "split"):
        if key in row and key not in metadata:
            metadata[key] = row[key]
    return GeoSample(
        sample_id=sample_id,
        dataset=str(row.get("dataset") or row.get("source_dataset") or default_dataset),
        media_paths=tuple(str(x) for x in media_paths),
        question=question,
        answer=answer,
        choices=choices,
        task_type=row.get("task_type"),
        metadata=metadata,
    )


def sample_to_dict(sample: GeoSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "media_paths": list(sample.media_paths),
        "question": sample.question,
        "answer": sample.answer,
        "choices": list(sample.choices) if sample.choices is not None else None,
        "task_type": sample.task_type,
        "metadata": sample.metadata,
    }
