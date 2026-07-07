from __future__ import annotations

import argparse
from pathlib import Path

from geopsro4d.data.dataset_rft import RFTDataset
from geopsro4d.data.formatters import psro_prompt
from geopsro4d.data.schema import sample_to_dict
from geopsro4d.reward.reward_fn import compute_reward
from geopsro4d.utils.io import ensure_dir, write_json, write_jsonl


def run_smoke(output: Path) -> dict[str, float | int]:
    output = ensure_dir(output)
    response = (
        "<think>\n"
        "Spatial Observation: the red cube is left of the blue cube in the first frame.\n"
        "Spatial Transition: across frames, the red cube moves to the right while the blue cube remains visible.\n"
        "Answer Derivation: the observation and transition show the red cube changes position relative to the blue cube.\n"
        "</think>\n"
        "<answer>A</answer>"
    )
    reward = compute_reward(response, "A", ["red moves right", "blue moves left"], "dynamic_transition")
    row = {"sample_id": "smoke", "response": response, **reward}
    write_jsonl(output / "rollout_samples.jsonl", [row])
    write_jsonl(output / "reward_logs.jsonl", [row])
    write_json(output / "metrics.json", row)
    return {"reward": float(reward["reward"]), "r_acc": int(reward["r_acc"])}


def prepare_verl_dataset(rft_jsonl: Path, output: Path) -> Path:
    dataset = RFTDataset.from_jsonl(rft_jsonl)
    rows = []
    for sample in dataset:
        row = sample_to_dict(sample)
        row["prompt"] = psro_prompt(sample)
        rows.append(row)
    out_path = ensure_dir(output) / "psro_rft_train.jsonl"
    write_jsonl(out_path, rows)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/stage3_psro_rft"))
    parser.add_argument("--rft-jsonl", type=Path)
    parser.add_argument("--stage2-checkpoint", type=Path)
    parser.add_argument("--backend", choices=["prepare", "verl"], default="prepare")
    args = parser.parse_args()
    if args.smoke:
        print(run_smoke(args.output))
        return
    if not args.rft_jsonl or not args.stage2_checkpoint:
        raise SystemExit("Stage 3 real run requires --rft-jsonl and --stage2-checkpoint")
    verl_data = prepare_verl_dataset(args.rft_jsonl, args.output)
    plan = {
        "backend": args.backend,
        "algorithm": "gspo",
        "group_size": 8,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 512,
        "kl_coef": 0.02,
        "train_file": str(verl_data),
        "init_model": str(args.stage2_checkpoint),
        "reward_fn": "geopsro4d.reward.reward_fn.compute_reward",
    }
    write_json(args.output / "verl_launch_plan.json", plan)
    print(plan)


if __name__ == "__main__":
    main()
