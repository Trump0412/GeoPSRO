from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from geopsro4d.train.common import make_smoke_cache, make_smoke_model, save_adapter
from geopsro4d.utils.io import write_json, write_jsonl
from geopsro4d.utils.seed import seed_everything


def run_smoke(output: Path, steps: int, geometry_on_ratio: float) -> dict[str, float | int]:
    seed_everything(3408)
    base, wrapper = make_smoke_model()
    opt = torch.optim.AdamW(list(base.parameters()) + list(wrapper.trainable_geo_parameters()), lr=1e-3)
    cache = make_smoke_cache()
    rows = []
    geometry_on = 0
    for step in range(steps):
        mode = "normal" if random.random() < geometry_on_ratio else "zero"
        geometry_on += int(mode == "normal")
        input_ids = torch.randint(0, 127, (2, 16))
        labels = input_ids.roll(shifts=-1, dims=1)
        geo = wrapper.geometry_inputs_embeds(cache, mode=mode)
        loss = base(input_ids, extra_embeds=geo, labels=labels)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        rows.append({"step": step + 1, "loss": float(loss.detach()), "geometry_mode": mode, "gate": float(wrapper.geometry_gate.value().mean())})
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "metrics.jsonl", rows)
    final = {**rows[-1], "geometry_on_ratio_observed": geometry_on / max(1, steps)}
    save_adapter(output, wrapper, final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/stage2_vggt_sft"))
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--geometry-on-ratio", type=float, default=0.7)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--spar-jsonl", type=Path)
    parser.add_argument("--llava-jsonl", type=Path)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    if args.smoke:
        print(run_smoke(args.output, args.steps, args.geometry_on_ratio))
        return
    if not args.stage1_checkpoint or not args.spar_jsonl or not args.llava_jsonl or not args.cache_root:
        raise SystemExit("Stage 2 real run requires --stage1-checkpoint, --spar-jsonl, --llava-jsonl and --cache-root")
    write_json(args.output / "launch_plan.json", vars(args))
    raise SystemExit("Stage 2 launch plan written; run the server script with real Qwen/LoRA settings.")


if __name__ == "__main__":
    main()
