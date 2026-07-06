from __future__ import annotations

from collections import defaultdict
from typing import Any


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def summarize_binary(rows: list[dict[str, Any]], key: str) -> float:
    return mean([1.0 if row.get(key) else 0.0 for row in rows])


def accuracy_by_task(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("task_type") or "unknown")].append(1.0 if row.get("correct") else 0.0)
    return {task: mean(vals) for task, vals in grouped.items()}
