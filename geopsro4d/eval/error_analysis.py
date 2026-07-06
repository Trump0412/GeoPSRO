from __future__ import annotations

from collections import Counter
from pathlib import Path

from geopsro4d.utils.io import iter_jsonl, write_json


def summarize_errors(scored_jsonl: str | Path, output: str | Path) -> dict:
    counts = Counter()
    for row in iter_jsonl(scored_jsonl):
        if not row.get("correct"):
            counts[str(row.get("task_type") or "unknown")] += 1
    summary = dict(counts)
    write_json(output, summary)
    return summary
