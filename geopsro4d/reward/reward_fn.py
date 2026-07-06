from __future__ import annotations

from geopsro4d.reward.answer_parser import normalize_answer, parse_answer
from geopsro4d.reward.process_rules import length_penalty, process_score
from geopsro4d.reward.psro_parser import parse_psro, psro_format_reward


def output_format_reward(pred_answer: str | None, response: str) -> float:
    if pred_answer is None:
        return 0.0
    return 1.0 if response.lower().count("answer") <= 2 else 0.5


def compute_reward(
    response: str,
    gold_answer: str,
    choices: list[str] | tuple[str, ...] | None = None,
    task_type: str | None = None,
) -> dict[str, float | str | None]:
    parsed = parse_psro(response)
    pred = parse_answer(response, choices)
    gold = normalize_answer(gold_answer)
    pred_norm = normalize_answer(pred)
    r_acc = 1.0 if pred_norm is not None and pred_norm == gold else 0.0
    r_out_fmt = output_format_reward(pred, response)
    r_psro_fmt = psro_format_reward(parsed)
    r_proc = process_score(parsed, task_type)
    r_len = length_penalty(response)
    total = r_acc + 0.20 * r_out_fmt + 0.15 * r_psro_fmt + 0.10 * r_acc * r_proc - 0.05 * r_len
    return {
        "reward": float(total),
        "r_acc": r_acc,
        "r_out_fmt": r_out_fmt,
        "r_psro_fmt": r_psro_fmt,
        "r_proc": r_proc,
        "r_len": r_len,
        "pred_answer": pred_norm,
    }
