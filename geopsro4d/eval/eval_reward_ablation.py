from __future__ import annotations

import argparse
from pathlib import Path

from geopsro4d.utils.io import write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/ablations/reward_ablation.csv"))
    args = parser.parse_args()
    rows = []
    for reward in ["answer_only", "answer_plus_output_format", "answer_plus_psro_format", "answer_plus_process", "full_psro_rft"]:
        rows.append({"reward": reward, "checkpoint": "pending", "eval_summary": "pending", "rollout_samples": "pending"})
    write_csv(args.output, rows)
    print(args.output)


if __name__ == "__main__":
    main()
