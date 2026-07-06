from __future__ import annotations

import argparse
from pathlib import Path

import torch

from geopsro4d.train.common import make_smoke_cache, make_smoke_model, save_adapter
from geopsro4d.utils.io import write_jsonl
from geopsro4d.utils.seed import seed_everything


def run_smoke(output: Path, steps: int) -> dict[str, float | int]:
    seed_everything(3407)
    base, wrapper = make_smoke_model()
    for param in base.parameters():
        param.requires_grad_(False)
    opt = torch.optim.AdamW(wrapper.trainable_geo_parameters(), lr=1e-3)
    rows = []
    cache = make_smoke_cache()
    for step in range(steps):
        input_ids = torch.randint(0, 127, (2, 16))
        labels = input_ids.roll(shifts=-1, dims=1)
        geo = wrapper.geometry_inputs_embeds(cache, mode="normal")
        loss = base(input_ids, extra_embeds=geo, labels=labels)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        rows.append({"step": step + 1, "loss": float(loss.detach()), "gate": float(wrapper.geometry_gate.value().mean())})
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "metrics.jsonl", rows)
    final = rows[-1]
    save_adapter(output, wrapper, final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/stage1_vggt_align"))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    if args.smoke:
        print(run_smoke(args.output, args.steps))
        return
    if not args.train_jsonl or not args.cache_root:
        raise SystemExit("Stage 1 real run requires --train-jsonl and --cache-root")
    raise SystemExit("Real Qwen Stage 1 launch is configured through scripts/run_stage1_align.sh; use --smoke for local validation.")


if __name__ == "__main__":
    main()
