from __future__ import annotations

import re
import unicodedata


OPTION_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?:[\).:：、\s]|$)")


def normalize_answer(text: str | None) -> str | None:
    if text is None:
        return None
    text = unicodedata.normalize("NFKC", str(text)).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .。,:：;；")
    return text.upper() if len(text) == 1 and text.isalpha() else text.lower()


def parse_answer(text: str, choices: list[str] | tuple[str, ...] | None = None) -> str | None:
    text = unicodedata.normalize("NFKC", text)
    answer_sections = re.findall(r"(?is)answer\s*[:：]\s*(.*)", text)
    target = answer_sections[-1].strip() if answer_sections else text.strip()
    if choices:
        matches = [m.group(1).upper() for m in OPTION_RE.finditer(target.upper())]
        labels = {chr(ord("A") + i) for i in range(len(choices))}
        matches = [m for m in matches if m in labels]
        if len(set(matches)) == 1:
            return matches[-1]
        if len(set(matches)) > 1:
            return None
        first = target.strip()[:1].upper()
        return first if first in labels else None
    target = re.split(r"[\n\r]", target)[0].strip()
    return normalize_answer(target) if target else None
