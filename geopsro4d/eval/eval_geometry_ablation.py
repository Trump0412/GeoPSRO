from __future__ import annotations

import argparse
from pathlib import Path

from geopsro4d.eval.eval_dsr import evaluate_predictions
from geopsro4d.utils.io import write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/ablations/geometry_ablation.csv"))
    parser.add_argument("--modes", nargs="+", default=["normal", "zero", "shuffle", "depth_only", "depth_camera", "full"])
    args = parser.parse_args()
    rows = []
    for mode in args.modes:
        summary = evaluate_predictions(args.predictions, args.output.parent / mode, prompt_mode="psro", geometry_mode=mode)
        rows.append(
            {
                "model": "geopsro4d",
                "checkpoint": "pending",
                "geometry_mode": mode,
                "dataset": "dsr",
                "accuracy": summary["accuracy"],
                "answer_parse_rate": summary["answer_parse_rate"],
                "psro_format_rate": summary["psro_format_rate"],
                "avg_length": summary["avg_output_length"],
            }
        )
    write_csv(args.output, rows)
    print(args.output)


if __name__ == "__main__":
    main()
