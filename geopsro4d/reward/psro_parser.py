from __future__ import annotations

import re
from typing import Any


SECTION_NAMES = ("observation", "transition", "derivation", "answer")
HEADER_RE = re.compile(r"(?im)^\s*(observation|transition|derivation|answer)\s*[:：]\s*")


def parse_psro(text: str) -> dict[str, Any]:
    matches = list(HEADER_RE.finditer(text))
    sections = {name: "" for name in SECTION_NAMES}
    present = {name: False for name in SECTION_NAMES}
    for idx, match in enumerate(matches):
        name = match.group(1).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()
        present[name] = True
    empty = {name: not bool(sections[name].strip()) for name in SECTION_NAMES}
    return {**sections, "present": present, "empty": empty}


def psro_format_reward(parsed: dict[str, Any]) -> float:
    score = 0.0
    for name in SECTION_NAMES:
        if parsed["present"].get(name) and not parsed["empty"].get(name):
            score += 0.25
    return min(1.0, score)
