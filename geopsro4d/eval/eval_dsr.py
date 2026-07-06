from __future__ import annotations

import argparse
from pathlib import Path

from geopsro4d.reward.answer_parser import normalize_answer, parse_answer
from geopsro4d.reward.psro_parser import parse_psro, psro_format_reward
from geopsro4d.utils.io import iter_jsonl, write_csv, write_json, write_jsonl
from geopsro4d.utils.metrics import accuracy_by_task, mean


def evaluate_predictions(predictions: Path, output: Path, *, prompt_mode: str, geometry_mode: str) -> dict:
    rows = []
    for row in iter_jsonl(predictions):
        response = str(row.get("raw_response") or row.get("response") or row.get("prediction") or "")
        choices = row.get("choices")
        gold = row.get("answer") or row.get("gold_answer")
        pred = parse_answer(response, choices)
        correct = normalize_answer(pred) == normalize_answer(gold)
        parsed = parse_psro(response)
        out = {
            "sample_id": row.get("sample_id") or row.get("id"),
            "question": row.get("question"),
            "choices": choices,
            "gold_answer": gold,
            "raw_response": response,
            "pred_answer": pred,
            "correct": bool(correct),
            "geometry_mode": geometry_mode,
            "frame_indices": row.get("frame_indices") or [],
            "task_type": row.get("task_type"),
            "psro_format": psro_format_reward(parsed) >= 1.0 if prompt_mode == "psro" else False,
            "length": len(response.split()),
        }
        rows.append(out)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "dsr_predictions_scored.jsonl", rows)
    summary = {
        "accuracy": mean([1.0 if r["correct"] else 0.0 for r in rows]),
        "answer_parse_rate": mean([1.0 if r["pred_answer"] is not None else 0.0 for r in rows]),
        "psro_format_rate": mean([1.0 if r["psro_format"] else 0.0 for r in rows]),
        "avg_output_length": mean([float(r["length"]) for r in rows]),
        "accuracy_by_task_type": accuracy_by_task(rows),
        "num_samples": len(rows),
    }
    write_json(output / "dsr_bench_summary.json", summary)
    write_csv(output / "dsr_bench_summary.csv", [{"metric": k, "value": v} for k, v in summary.items() if k != "accuracy_by_task_type"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/eval/dsr"))
    parser.add_argument("--prompt-mode", choices=["direct_answer", "psro"], default="psro")
    parser.add_argument("--geometry-mode", default="normal")
    args = parser.parse_args()
    print(evaluate_predictions(args.predictions, args.output, prompt_mode=args.prompt_mode, geometry_mode=args.geometry_mode))


if __name__ == "__main__":
    main()
