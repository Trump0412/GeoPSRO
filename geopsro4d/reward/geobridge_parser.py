"""Structured rationale parsers for gap-4D PSRO."""

from __future__ import annotations

import re
from typing import Dict

from geopsro4d.reward.psro_parser import parse_psro


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

SECTION_ALIASES = {
    "observation": {
        "spatialobservation",
        "observation",
    },
    "transition": {
        "spatialtransition",
        "transition",
    },
    "derivation": {
        "answerderivation",
        "derivation",
    },
    "reference_type": {
        "referencetype",
        "referenceframe",
        "reference",
    },
    "target_object": {
        "targetobject",
        "targetobjects",
        "object",
    },
    "explanation": {
        "explanation",
    },
}

LABEL_TO_KEY = {
    alias: key
    for key, aliases in SECTION_ALIASES.items()
    for alias in aliases
}

HEADER_RE = re.compile(
    r"^\s*(?:\[(?P<bracket>[^\]]+)\]|(?P<plain>[A-Za-z][A-Za-z _/-]*))\s*:\s*(?P<rest>.*)$"
)


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (label or "").lower())


def _parse_sections(body: str) -> Dict[str, str]:
    sections = {
        "observation": "",
        "transition": "",
        "derivation": "",
        "reference_type": "",
        "target_object": "",
        "explanation": "",
    }
    current_key = ""
    buffers = {key: [] for key in sections}
    for line in (body or "").splitlines():
        match = HEADER_RE.match(line)
        if match:
            label = match.group("bracket") or match.group("plain") or ""
            key = LABEL_TO_KEY.get(_normalize_label(label))
            if key:
                current_key = key
                rest = match.group("rest").strip()
                if rest:
                    buffers[current_key].append(rest)
                continue
        if current_key:
            buffers[current_key].append(line.strip())

    for key, values in buffers.items():
        sections[key] = "\n".join(value for value in values if value).strip()
    return sections


def _extract_section(body: str, name: str) -> str:
    pattern = re.compile(rf"{re.escape(name)}\s*:\s*(.*?)(?=\n[A-Z][A-Za-z ]+\s*:|\Z)", re.DOTALL)
    match = pattern.search(body)
    return match.group(1).strip() if match else ""


def parse_geobridge_response(text: str) -> Dict[str, str | bool]:
    text = text or ""
    think_matches = THINK_RE.findall(text)
    answer_matches = ANSWER_RE.findall(text)
    think_match = THINK_RE.search(text)
    answer_match = ANSWER_RE.search(text)
    think_body = think_match.group(1).strip() if think_match else ""
    answer_body = answer_match.group(1).strip() if answer_match else ""
    if not think_body and not answer_body:
        parsed = parse_psro(text)
        return {
            "has_think": False,
            "has_answer": bool(parsed.get("answer")),
            "has_single_think": False,
            "has_single_answer": True,
            "has_observation": bool(parsed.get("observation")),
            "has_transition": bool(parsed.get("transition")),
            "has_derivation": bool(parsed.get("derivation")),
            "observation": str(parsed.get("observation", "")),
            "transition": str(parsed.get("transition", "")),
            "derivation": str(parsed.get("derivation", "")),
            "reference_type": "",
            "target_object": "",
            "explanation": "",
            "answer": str(parsed.get("answer", "")),
            "raw_think": text,
            "raw_text": text,
        }

    sections = _parse_sections(think_body)
    observation = sections["observation"] or _extract_section(think_body, "Spatial Observation")
    transition = sections["transition"] or _extract_section(think_body, "Spatial Transition")
    derivation = sections["derivation"] or _extract_section(think_body, "Answer Derivation")
    return {
        "has_think": bool(think_match),
        "has_answer": bool(answer_match),
        "has_single_think": len(think_matches) == 1,
        "has_single_answer": len(answer_matches) == 1,
        "has_observation": bool(observation),
        "has_transition": bool(transition),
        "has_derivation": bool(derivation),
        "observation": observation,
        "transition": transition,
        "derivation": derivation,
        "reference_type": sections["reference_type"],
        "target_object": sections["target_object"],
        "explanation": sections["explanation"],
        "answer": answer_body,
        "raw_think": think_body,
        "raw_text": text,
    }
