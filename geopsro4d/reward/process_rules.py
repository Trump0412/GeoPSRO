from __future__ import annotations

from typing import Any


SPATIAL_TERMS = {
    "left", "right", "above", "below", "front", "behind", "near", "far", "closer", "farther",
    "between", "inside", "outside", "overlap", "occlude", "visible", "hidden", "distance",
    "direction", "orientation", "viewpoint", "camera", "object", "region",
}
TEMPORAL_TERMS = {
    "moves", "moved", "moving", "changes", "becomes", "remains", "approaches", "recedes",
    "appears", "disappears", "before", "after", "first", "then", "later", "earlier",
    "transition", "across frames", "from frame", "to frame",
}


def process_score(parsed: dict[str, Any], task_type: str | None = None) -> float:
    observation = parsed.get("observation", "").lower()
    transition = parsed.get("transition", "").lower()
    derivation = parsed.get("derivation", "").lower()
    answer = parsed.get("answer", "").lower()
    score = 0.0
    if _has_any(observation, SPATIAL_TERMS) and len(observation.split()) >= 3:
        score += 0.25
    needs_transition = _is_dynamic(task_type) or bool(transition)
    if not needs_transition or _has_any(transition, TEMPORAL_TERMS):
        score += 0.25
    if "observation" in derivation or "transition" in derivation or _has_any(derivation, SPATIAL_TERMS):
        score += 0.25
    if answer and answer not in observation and answer not in transition:
        score += 0.25
    return min(1.0, score)


def length_penalty(text: str, *, max_words: int = 180) -> float:
    words = text.split()
    if len(words) <= max_words:
        return 0.0
    repeated = max(0, len(words) - len(set(words)))
    return min(1.0, (len(words) - max_words) / max_words + repeated / max_words)


def _has_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _is_dynamic(task_type: str | None) -> bool:
    if not task_type:
        return False
    t = task_type.lower()
    return any(key in t for key in ("dynamic", "motion", "transition", "video", "temporal", "4d"))
