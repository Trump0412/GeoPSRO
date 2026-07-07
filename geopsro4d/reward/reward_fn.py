"""GeoBridge PSRO custom reward function for verl.

Stage3 v1 uses a deliberately low-cost rule reward:

    R_total = R_answer + 0.5 * R_format + 0.05 * R_words

It relies only on QA supervision and the model output text. It does not use
intermediate visual labels, detection models, segmentation, 3D boxes, or online
MLLM judges.
"""

from __future__ import annotations

import math
import pathlib
import re
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

try:
    from geopsro4d.reward.geobridge_parser import parse_geobridge_response
except ImportError:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from geopsro4d.reward.geobridge_parser import parse_geobridge_response


MAX_THINK_CHARS = 1200
MAX_ANSWER_CHARS = 128
MAX_RESPONSE_CHARS = 4096

YES_ALIASES = {
    "yes",
    "true",
    "correct",
    "right",
    "affirmative",
    "indeed",
    "是",
    "对",
    "正确",
}

NO_ALIASES = {
    "no",
    "false",
    "incorrect",
    "wrong",
    "negative",
    "not",
    "否",
    "不",
    "错误",
}

DIRECTION_ALIASES = {
    "left": {"left", "to the left", "left side", "on the left", "leftward"},
    "right": {"right", "to the right", "right side", "on the right", "rightward"},
    "front": {"front", "in front", "in front of", "ahead", "forward"},
    "behind": {"behind", "back", "at the back", "in the back", "rear"},
    "near": {"near", "nearby", "close", "close to", "closer", "adjacent", "next to"},
    "far": {"far", "far away", "farther", "distant", "away from"},
    "above": {"above", "over", "on top", "higher", "upper"},
    "below": {"below", "under", "beneath", "lower", "down"},
    "inside": {"inside", "within", "in", "contained"},
    "outside": {"outside", "out", "external"},
}

NUMBER_WORDS = {
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
}

ORDINAL_TO_LETTER = {
    "first": "A",
    "1st": "A",
    "one": "A",
    "second": "B",
    "2nd": "B",
    "two": "B",
    "third": "C",
    "3rd": "C",
    "three": "C",
    "fourth": "D",
    "4th": "D",
    "four": "D",
}

OBSERVATION_WORDS = {
    "observe",
    "observed",
    "observation",
    "see",
    "seen",
    "visible",
    "visibly",
    "show",
    "shown",
    "shows",
    "display",
    "displayed",
    "depict",
    "depicted",
    "appear",
    "appears",
    "appearance",
    "present",
    "presence",
    "identify",
    "identified",
    "view",
    "viewed",
    "frame",
    "frames",
    "image",
    "images",
    "video",
    "clip",
    "sequence",
    "scene",
    "visual input",
    "visual evidence",
    "visual cue",
    "visual cues",
    "visual context",
    "visible evidence",
    "spatial evidence",
    "key evidence",
    "relevant evidence",
    "scene layout",
    "layout",
    "spatial layout",
    "arrangement",
    "configuration",
    "context",
    "surroundings",
    "environment",
    "room",
    "background",
    "foreground",
    "midground",
    "object",
    "objects",
    "item",
    "items",
    "entity",
    "entities",
    "target",
    "reference",
    "landmark",
    "anchor",
    "region",
    "area",
    "part",
    "surface",
    "plane",
    "floor",
    "ground",
    "wall",
    "ceiling",
    "corner",
    "edge",
    "boundary",
    "shape",
    "structure",
    "visible object",
    "target object",
    "reference object",
    "anchor object",
    "located",
    "location",
    "position",
    "positioned",
    "placement",
    "placed",
    "situated",
    "standing",
    "sitting",
    "lying",
    "resting",
    "facing",
    "oriented",
    "orientation",
    "aligned",
    "alignment",
    "centered",
    "central",
    "middle",
    "side",
    "left side",
    "right side",
    "front side",
    "back side",
    "upper side",
    "lower side",
    "nearby",
    "adjacent",
    "relative",
    "relative position",
    "relative relation",
    "relationship",
    "spatial relation",
    "spatial relationship",
    "relation",
    "object relation",
    "positional relation",
    "geometric relation",
    "layout relation",
    "visibility",
    "partially visible",
    "fully visible",
    "hidden",
    "occluded",
    "occlusion",
    "blocked",
    "covered",
    "overlapping",
    "clear",
    "line of sight",
    "depth",
    "depth cue",
    "depth cues",
    "distance",
    "relative distance",
    "scale",
    "size",
    "apparent size",
    "perspective",
    "viewpoint",
    "camera view",
    "angle",
    "geometry",
    "geometric",
    "spatial",
    "3d",
    "three-dimensional",
    "projection",
    "view geometry",
    "spatial cue",
    "clearly",
    "evidently",
    "visually",
    "based on the frame",
    "based on the image",
    "from the frame",
    "from the image",
    "in the frame",
    "in the image",
    "within the scene",
    "as shown",
    "as visible",
    "as observed",
}

SPATIAL_RELATION_WORDS = {
    "left",
    "right",
    "leftward",
    "rightward",
    "to the left",
    "to the right",
    "on the left",
    "on the right",
    "left side",
    "right side",
    "front",
    "behind",
    "back",
    "rear",
    "ahead",
    "forward",
    "backward",
    "in front",
    "in front of",
    "above",
    "below",
    "over",
    "under",
    "beneath",
    "underneath",
    "on top of",
    "higher",
    "lower",
    "upper",
    "upward",
    "downward",
    "vertical",
    "height",
    "near",
    "nearby",
    "nearer",
    "close",
    "close to",
    "closer",
    "far",
    "farther",
    "distant",
    "away",
    "away from",
    "distance",
    "relative distance",
    "separated",
    "separation",
    "gap",
    "proximity",
    "adjacent",
    "next to",
    "beside",
    "inside",
    "outside",
    "within",
    "contained",
    "surrounded",
    "around",
    "between",
    "among",
    "across",
    "along",
    "against",
    "touching",
    "contact",
    "overlapping",
    "connected",
    "facing",
    "facing toward",
    "facing away",
    "toward",
    "opposite",
    "same side",
    "aligned",
    "parallel",
    "perpendicular",
    "horizontal",
    "diagonal",
    "orientation",
    "direction",
    "pose",
    "center",
    "central",
    "middle",
    "edge",
    "corner",
    "side",
    "boundary",
    "relative position",
    "spatial relation",
    "spatial relationship",
    "positional relation",
    "geometric relation",
    "layout relation",
    "relation remains",
    "relation changes",
    "same relation",
    "different relation",
    "spatial configuration",
    "spatial arrangement",
}

TRANSITION_WORDS = {
    "before",
    "after",
    "then",
    "later",
    "earlier",
    "initially",
    "finally",
    "first",
    "second",
    "third",
    "next",
    "previous",
    "subsequent",
    "following",
    "during",
    "while",
    "over time",
    "temporal",
    "sequence",
    "sequential",
    "progression",
    "state evolution",
    "across frames",
    "between frames",
    "from frame",
    "to frame",
    "frame to frame",
    "in the earlier frame",
    "in the later frame",
    "in the first frame",
    "in the next frame",
    "over the frames",
    "through the frames",
    "visual sequence",
    "multi-frame",
    "cross-frame",
    "inter-frame",
    "view",
    "views",
    "viewpoint",
    "viewpoints",
    "camera view",
    "perspective",
    "view change",
    "viewpoint change",
    "perspective shift",
    "camera angle",
    "angle change",
    "camera",
    "camera moves",
    "camera moved",
    "camera motion",
    "ego-motion",
    "agent moves",
    "observer moves",
    "turn",
    "turns",
    "turned",
    "rotate",
    "rotates",
    "rotation",
    "pan",
    "tilt",
    "zoom",
    "approach",
    "approaches",
    "move closer",
    "moves closer",
    "move away",
    "shift",
    "shifts",
    "trajectory",
    "path",
    "motion",
    "movement",
    "move",
    "moves",
    "moved",
    "displacement",
    "change",
    "changes",
    "changed",
    "state change",
    "spatial change",
    "position changes",
    "relation changes",
    "layout changes",
    "becomes",
    "transition",
    "transitions",
    "moves left",
    "moves right",
    "moves forward",
    "moves backward",
    "enters",
    "leaves",
    "remain",
    "remains",
    "remained",
    "unchanged",
    "same",
    "stable",
    "consistent",
    "preserved",
    "maintained",
    "continues",
    "continuity",
    "spatial continuity",
    "state continuity",
    "relation remains",
    "position remains",
    "still",
    "still visible",
    "no change",
    "does not change",
    "same side",
    "same position",
    "same relation",
    "appear",
    "appears",
    "disappear",
    "disappears",
    "reappear",
    "becomes visible",
    "becomes hidden",
    "visible again",
    "occluded",
    "occlusion",
    "blocked",
    "covered",
    "revealed",
    "enters view",
    "leaves view",
    "field of view",
    "visibility changes",
    "gets closer",
    "becomes closer",
    "gets farther",
    "becomes farther",
    "moves to the left",
    "moves to the right",
    "shifts left",
    "shifts right",
    "moves in front",
    "moves behind",
    "changes side",
    "relative position changes",
    "relative position remains",
    "relation holds",
    "relation is preserved",
    "spatial state",
    "previous state",
    "current state",
    "later state",
}

DERIVATION_WORDS = {
    "therefore",
    "thus",
    "so",
    "hence",
    "because",
    "since",
    "given",
    "based on",
    "according to",
    "from this",
    "from these observations",
    "this means",
    "this indicates",
    "this shows",
    "this supports",
    "as a result",
    "for this reason",
    "due to",
    "infer",
    "inferred",
    "inference",
    "conclude",
    "conclusion",
    "deduce",
    "deduced",
    "derive",
    "derived",
    "derivation",
    "reason",
    "reasoning",
    "determine",
    "determined",
    "decide",
    "decision",
    "judge",
    "identify",
    "evidence",
    "visual evidence",
    "spatial evidence",
    "cue",
    "cues",
    "supports",
    "supported by",
    "matches",
    "consistent with",
    "aligned with",
    "confirms",
    "implies",
    "leads to",
    "points to",
    "follows from",
    "derived from",
    "based on the observation",
    "based on the transition",
    "answer",
    "the answer",
    "final answer",
    "correct answer",
    "option",
    "choice",
    "selected option",
    "correct choice",
    "choose",
    "chosen",
    "select",
    "selected",
    "letter",
    "yes",
    "no",
    "left",
    "right",
    "front",
    "behind",
    "near",
    "far",
    "above",
    "below",
    "inside",
    "outside",
    "same",
    "different",
    "best matches",
    "most consistent",
    "rather than",
    "instead of",
    "must be",
    "is correct",
    "is supported",
}

EVIDENCE_GEOMETRY_WORDS = {
    "evidence",
    "visual evidence",
    "spatial evidence",
    "geometric evidence",
    "image evidence",
    "frame evidence",
    "visible cue",
    "spatial cue",
    "geometric cue",
    "depth cue",
    "perspective cue",
    "occlusion cue",
    "layout cue",
    "view cue",
    "relation cue",
    "motion cue",
    "camera cue",
    "supporting evidence",
    "observed evidence",
    "grounded",
    "visually grounded",
    "spatially grounded",
    "geometry",
    "geometric",
    "3d",
    "three-dimensional",
    "depth",
    "surface",
    "plane",
    "projection",
    "perspective",
    "view geometry",
    "camera geometry",
    "pose",
    "camera pose",
    "viewpoint",
    "parallax",
    "occlusion",
    "visibility",
    "field of view",
    "line of sight",
    "spatial layout",
    "scene structure",
    "shape",
    "scale",
    "relative scale",
    "distance",
    "relative distance",
    "topology",
    "consistent",
    "consistency",
    "geometric consistency",
    "spatial consistency",
    "visual consistency",
    "cross-frame consistency",
    "multi-frame consistency",
    "temporal consistency",
    "state consistency",
    "relation consistency",
    "aligned",
    "alignment",
    "correspondence",
    "matching",
    "same object",
    "same relation",
    "same state",
    "stable relation",
    "stable position",
    "continuous",
    "continuity",
    "spatial continuity",
    "geometric continuity",
    "supported",
    "matches",
    "agrees",
    "confirms",
    "verifies",
    "compare",
    "compared",
    "comparison",
    "cross-check",
    "validate",
}

UNSUPPORTED_RISK_WORDS = {
    "maybe",
    "possibly",
    "probably",
    "seems",
    "appears to",
    "might",
    "could be",
    "uncertain",
    "unclear",
    "cannot determine",
    "not enough information",
    "i cannot see",
    "not visible",
    "unknown",
    "guess",
    "assume",
    "assumption",
    "unsupported",
    "not shown",
    "not provided",
    "not in the image",
}


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _flatten_strings(value: Any) -> List[str]:
    out: List[str] = []
    for item in _as_list(value):
        if isinstance(item, (list, tuple, set)):
            out.extend(_flatten_strings(item))
        elif item is not None:
            text = str(item).strip()
            if text:
                out.append(text)
    return out


def _normalize_text(text: Any) -> str:
    text = str(text or "").lower()
    text = text.replace("’", "'")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9.+\-\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _boundary_contains(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    if " " in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _unique_phrase_hits(text: Any, vocab: Iterable[str]) -> Set[str]:
    norm = _normalize_text(text)
    hits: Set[str] = set()
    for phrase in sorted(vocab, key=len, reverse=True):
        phrase_norm = _normalize_text(phrase)
        if _boundary_contains(norm, phrase_norm):
            hits.add(phrase)
    return hits


def _count_all_phrase_hits(text: Any, vocabs: Sequence[Iterable[str]]) -> int:
    norm = _normalize_text(text)
    count = 0
    for vocab in vocabs:
        for phrase in vocab:
            phrase_norm = _normalize_text(phrase)
            if not phrase_norm:
                continue
            if " " in phrase_norm:
                count += norm.count(phrase_norm)
            else:
                count += len(re.findall(rf"\b{re.escape(phrase_norm)}\b", norm))
    return count


def _looks_like_word_list(text: Any) -> bool:
    norm = _normalize_text(text)
    comma_parts = [part.strip() for part in str(text or "").lower().split(",") if part.strip()]
    if len(comma_parts) >= 12:
        avg_len = sum(len(part.split()) for part in comma_parts) / max(len(comma_parts), 1)
        if avg_len <= 3:
            return True
    sentence_marks = sum(norm.count(mark) for mark in (".", ";", " because ", " since ", " therefore ", " so "))
    word_count = len(norm.split())
    return word_count >= 30 and sentence_marks == 0 and len(comma_parts) >= 8


def _section_looks_like_vocab_list(text: Any, hits: Set[str]) -> bool:
    norm = _normalize_text(text)
    words = [word for word in norm.split() if word]
    if len(words) < 6 or len(hits) < 5:
        return False
    unique_words = set(words)
    hit_ratio = len({_normalize_text(hit) for hit in hits}) / max(1, len(unique_words))
    sentence_cues = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "in",
        "of",
        "to",
        "while",
        "since",
        "because",
        "therefore",
    }
    cue_count = sum(1 for cue in sentence_cues if cue in unique_words)
    return hit_ratio >= 0.65 and cue_count <= 2


def _choice_map(choice_set: Any) -> Dict[str, str]:
    choices: Dict[str, str] = {}
    raw = choice_set
    if isinstance(choice_set, Mapping):
        raw = [f"{key}. {value}" for key, value in choice_set.items()]
    for choice in _flatten_strings(raw):
        match = re.match(r"^\s*([A-Za-z])\s*[\.\):：-]\s*(.*?)\s*$", choice)
        if match:
            choices[match.group(1).upper()] = match.group(2).strip()
    return choices


def _extract_choice_letter(text: Any) -> str:
    raw = str(text or "").strip()
    match = re.match(r"^\s*(?:the\s+)?(?:option\s*)?([A-Da-d])(?:\s*[\.\):：-]|\s*$)", raw, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    norm = _normalize_text(raw)
    for token, letter in ORDINAL_TO_LETTER.items():
        if re.search(rf"\b(?:the\s+)?{re.escape(token)}(?:\s+option)?\b", norm):
            return letter
    return norm.upper() if len(norm) == 1 and norm.upper() in {"A", "B", "C", "D"} else ""


def _canonical_bool(text: Any) -> str:
    norm = _normalize_text(text)
    if norm in YES_ALIASES:
        return "yes"
    if norm in NO_ALIASES:
        return "no"
    return ""


def _canonical_direction(text: Any) -> str:
    norm = _normalize_text(text)
    for canon, aliases in DIRECTION_ALIASES.items():
        if any(_boundary_contains(norm, _normalize_text(alias)) for alias in aliases):
            return canon
    return ""


def _numbers(text: Any) -> List[float]:
    norm = _normalize_text(text)
    values = [float(match) for match in re.findall(r"[-+]?\d+(?:\.\d+)?", norm)]
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", norm):
            values.append(value)
    return values


def _numeric_match(pred: Any, gt: Any) -> bool:
    pred_nums = _numbers(pred)
    gt_nums = _numbers(gt)
    if not pred_nums or not gt_nums:
        return False
    gt_value = gt_nums[0]
    tol = max(0.05 * abs(gt_value), 0.5)
    return any(abs(pred_value - gt_value) <= tol for pred_value in pred_nums)


def _valid_answer_texts(ground_truth: Any, extra: Mapping[str, Any], choices: Mapping[str, str]) -> List[str]:
    valid = [str(ground_truth or "")]
    valid.extend(_flatten_strings(extra.get("valid_answers")))
    valid.extend(_flatten_strings(extra.get("answer_annotated")))
    gt_letter = _extract_choice_letter(ground_truth)
    if gt_letter and gt_letter in choices:
        valid.extend([gt_letter, choices[gt_letter], f"{gt_letter}. {choices[gt_letter]}"])
    return [item for item in dict.fromkeys(text.strip() for text in valid) if item]


def _fallback_answer_text(raw_text: Any) -> str:
    text = str(raw_text or "")
    match = re.search(r"(?:^|\b)(?:final\s+)?answer\s*:\s*(.+?)\s*$", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    answer = match.group(1).strip()
    answer = re.split(r"\n\s*\n|</think>|<think>", answer, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return answer[:MAX_ANSWER_CHARS]


def _answer_score(pred_answer: Any, ground_truth: Any, extra: Mapping[str, Any]) -> Tuple[float, str]:
    pred_answer = str(pred_answer or "").strip()
    choices = _choice_map(extra.get("choice_set") or extra.get("choices"))
    answer_type = _normalize_text(extra.get("answer_type", ""))
    pred_letter = _extract_choice_letter(pred_answer)
    gt_letter = _extract_choice_letter(ground_truth)

    if choices or "multiple" in answer_type or gt_letter:
        if pred_letter and gt_letter and pred_letter == gt_letter:
            return 1.0, gt_letter
        if pred_letter and pred_letter in choices:
            for valid in _valid_answer_texts(ground_truth, extra, choices):
                if _normalize_text(choices[pred_letter]) == _normalize_text(valid):
                    return 1.0, pred_letter
        pred_norm = _normalize_text(pred_answer)
        pred_dir = _canonical_direction(pred_answer)
        for valid in _valid_answer_texts(ground_truth, extra, choices):
            valid_letter = _extract_choice_letter(valid)
            if pred_letter and valid_letter and pred_letter == valid_letter:
                return 1.0, valid_letter
            if pred_norm == _normalize_text(valid):
                return 1.0, valid
            valid_dir = _canonical_direction(valid)
            if pred_dir and valid_dir and pred_dir == valid_dir:
                return 1.0, valid_dir
        return 0.0, ""

    if "numeric" in answer_type or _numbers(ground_truth):
        return (1.0, str(ground_truth)) if _numeric_match(pred_answer, ground_truth) else (0.0, "")

    pred_dir = _canonical_direction(pred_answer)
    gt_dir = _canonical_direction(ground_truth)
    if pred_dir or gt_dir:
        return (1.0, gt_dir) if pred_dir and pred_dir == gt_dir else (0.0, "")

    pred_bool = _canonical_bool(pred_answer)
    gt_bool = _canonical_bool(ground_truth)
    if pred_bool or gt_bool:
        return (1.0, gt_bool) if pred_bool and pred_bool == gt_bool else (0.0, "")

    pred_norm = _normalize_text(pred_answer)
    for valid in _valid_answer_texts(ground_truth, extra, choices):
        if pred_norm == _normalize_text(valid):
            return 1.0, valid
    return 0.0, ""


def _answer_contains_multiple_conflicting_options(answer_text: Any) -> bool:
    letters = set(re.findall(r"\b[A-D]\b", str(answer_text or "").upper()))
    return len(letters) > 1


def _format_score(parsed: Mapping[str, Any], *, max_think_chars: int, max_answer_chars: int, max_response_chars: int) -> float:
    answer_text = str(parsed.get("answer", "") or "").strip()
    if not answer_text:
        return 0.0

    score = 0.0
    score += 0.20 if parsed.get("has_think") and parsed.get("has_single_think") else 0.0
    score += 0.20 if parsed.get("has_answer") and parsed.get("has_single_answer") else 0.0
    score += 0.20 if parsed.get("has_observation") else 0.0
    score += 0.20 if parsed.get("has_transition") else 0.0
    score += 0.20 if parsed.get("has_derivation") else 0.0

    if _answer_contains_multiple_conflicting_options(answer_text):
        score = min(score, 0.5)
    if len(answer_text) > max_answer_chars:
        score = min(score, 0.5)
    if len(str(parsed.get("raw_think", "") or "")) > max_think_chars:
        score = min(score, 0.7)
    if len(str(parsed.get("raw_text", "") or "")) > max_response_chars:
        score = min(score, 0.7)
    return _clip(score)


def _words_score(parsed: Mapping[str, Any], answer_score: float) -> Tuple[float, Dict[str, float]]:
    if answer_score < 1.0:
        return 0.0, {
            "words_observation": 0.0,
            "words_transition": 0.0,
            "words_derivation": 0.0,
            "words_relation": 0.0,
            "words_evidence": 0.0,
            "words_unique_hits": 0.0,
            "words_total_hits": 0.0,
            "words_unsupported_hits": 0.0,
        }

    observation = str(parsed.get("observation", ""))
    transition = str(parsed.get("transition", ""))
    derivation = str(parsed.get("derivation", ""))
    think = str(parsed.get("raw_think", ""))

    obs_hits = _unique_phrase_hits(observation, OBSERVATION_WORDS)
    trans_hits = _unique_phrase_hits(transition, TRANSITION_WORDS)
    deriv_hits = _unique_phrase_hits(derivation, DERIVATION_WORDS)
    relation_hits = _unique_phrase_hits(think, SPATIAL_RELATION_WORDS)
    evidence_hits = _unique_phrase_hits(think, EVIDENCE_GEOMETRY_WORDS)
    unsupported_hits = _unique_phrase_hits(think, UNSUPPORTED_RISK_WORDS)

    obs_score = min(len(obs_hits) / 5.0, 1.0)
    trans_score = min(len(trans_hits) / 5.0, 1.0)
    deriv_score = min(len(deriv_hits) / 4.0, 1.0)
    relation_score = min(len(relation_hits) / 4.0, 1.0)
    evidence_score = min(len(evidence_hits) / 3.0, 1.0)

    words_score = (
        0.24 * obs_score
        + 0.28 * trans_score
        + 0.22 * deriv_score
        + 0.16 * relation_score
        + 0.10 * evidence_score
    )

    all_unique = obs_hits | trans_hits | deriv_hits | relation_hits | evidence_hits
    total_hits = _count_all_phrase_hits(
        think,
        [OBSERVATION_WORDS, TRANSITION_WORDS, DERIVATION_WORDS, SPATIAL_RELATION_WORDS, EVIDENCE_GEOMETRY_WORDS],
    )
    vocab_list_sections = sum(
        (
            _section_looks_like_vocab_list(observation, obs_hits),
            _section_looks_like_vocab_list(transition, trans_hits),
            _section_looks_like_vocab_list(derivation, deriv_hits),
        )
    )
    if total_hits >= 20 and len(all_unique) <= 6:
        words_score = 0.0
    if vocab_list_sections >= 2:
        words_score = 0.0
    if _looks_like_word_list(think):
        words_score = 0.0

    return _clip(words_score), {
        "words_observation": float(obs_score),
        "words_transition": float(trans_score),
        "words_derivation": float(deriv_score),
        "words_relation": float(relation_score),
        "words_evidence": float(evidence_score),
        "words_unique_hits": float(len(all_unique)),
        "words_total_hits": float(total_hits),
        "words_unsupported_hits": float(len(unsupported_hits)),
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] | None = None,
    *,
    format_weight: float = 0.5,
    words_weight: float = 0.05,
    max_think_chars: int = MAX_THINK_CHARS,
    max_answer_chars: int = MAX_ANSWER_CHARS,
    max_response_chars: int = MAX_RESPONSE_CHARS,
) -> Dict[str, float]:
    """Compute the Stage3 v1 PSRO reward.

    The `data_source` argument is accepted for verl compatibility. The first
    version intentionally keeps the reward source-agnostic.
    """

    del data_source
    extra = dict(extra_info or {})
    parsed = parse_geobridge_response(solution_str)

    answer_text = parsed.get("answer", "") or _fallback_answer_text(parsed.get("raw_text", ""))
    answer, matched_answer = _answer_score(answer_text, ground_truth, extra)
    fmt = _format_score(
        parsed,
        max_think_chars=int(max_think_chars),
        max_answer_chars=int(max_answer_chars),
        max_response_chars=int(max_response_chars),
    )
    words, word_breakdown = _words_score(parsed, answer)
    score = answer + float(format_weight) * fmt + float(words_weight) * words
    if math.isnan(score) or math.isinf(score):
        score = 0.0

    response_len = len(str(parsed.get("raw_text", "") or ""))
    think_len = len(str(parsed.get("raw_think", "") or ""))
    out = {
        "score": float(score),
        "reward": float(score),
        "reward_total": float(score),
        "R_answer": float(answer),
        "R_format": float(fmt),
        "R_words": float(words),
        "r_acc": float(answer),
        "r_psro_fmt": float(fmt),
        "r_out_fmt": float(fmt),
        "r_proc": float(words),
        "r_len": 0.0,
        "answer": float(answer),
        "format": float(fmt),
        "words": float(words),
        "pred_answer": str(answer_text).strip(),
        "matched_answer": str(matched_answer),
        "format_pass_rate": float(1.0 if fmt >= 0.99 else 0.0),
        "answer_acc_proxy": float(answer),
        "avg_response_len": float(response_len),
        "avg_think_len": float(think_len),
        "word_score_mean": float(words),
        "word_score_nonzero_rate": float(1.0 if words > 0.0 else 0.0),
    }
    out.update(word_breakdown)
    return out


def compute_reward(
    response: str,
    gold_answer: str,
    choices: Sequence[str] | Mapping[str, str] | None = None,
    task_type: str | None = None,
) -> Dict[str, float | str]:
    """GeoPSRO wrapper around the GeoBridge Stage3 reward.

    Keeps the lightweight local API used by smoke tests while the underlying
    reward matches the old GeoBridge formula:
    R_answer + 0.5 * R_format + 0.05 * R_words.
    """

    extra_info: Dict[str, Any] = {}
    if choices:
        if isinstance(choices, Mapping):
            extra_info["choices"] = dict(choices)
        else:
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            extra_info["choices"] = [f"{labels[idx]}. {choice}" for idx, choice in enumerate(choices)]
    if task_type:
        extra_info["task_type"] = task_type
        if any(key in task_type.lower() for key in ("choice", "multiple", "mcq")):
            extra_info["answer_type"] = "multiple_choice"
    return compute_score(
        data_source="geopsro4d",
        solution_str=response,
        ground_truth=gold_answer,
        extra_info=extra_info,
    )
