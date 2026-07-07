from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import torch

from geopsro4d.data.schema import normalize_sample
from geopsro4d.geometry.vggt_cache import VGGTCache
from geopsro4d.model.qwen_vggt_wrapper import QwenVGGTWrapper
from geopsro4d.train.common import make_smoke_cache, make_smoke_model, save_adapter
from geopsro4d.utils.io import ensure_dir, iter_jsonl, write_json, write_jsonl
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


def run_qwen_smoke(
    *,
    output: Path,
    model_path: Path,
    spar_jsonl: Path,
    llava_jsonl: Path,
    spar_cache_root: Path,
    llava_cache_root: Path,
    steps: int,
    geometry_on_ratio: float,
    num_geo_tokens: int,
    lr_lora: float,
    lr_geo: float,
    max_text_tokens: int,
) -> dict[str, float | int | str]:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    seed_everything(3418)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    ).to(device)
    model.train()

    hidden_size = int(model.config.text_config.hidden_size)
    wrapper = QwenVGGTWrapper(
        None,
        llm_dim=hidden_size,
        raw_geo_dim=32,
        adapter_dim=256,
        num_geo_tokens=num_geo_tokens,
    ).to(device)
    params = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": lr_lora},
        {"params": list(wrapper.trainable_geo_parameters()), "lr": lr_geo},
    ]
    opt = torch.optim.AdamW(params)

    usable = []
    usable.extend(_cached_samples(spar_jsonl, spar_cache_root, limit=max(steps * 4, 128)))
    usable.extend(_cached_samples(llava_jsonl, llava_cache_root, limit=max(steps * 4, 128)))
    if not usable:
        raise SystemExit("No cached samples found for Stage 2 Qwen smoke")
    random.shuffle(usable)

    rows = []
    geometry_on = 0
    for step, (sample, cache_root) in zip(range(steps), itertools.cycle(usable)):
        mode = "full" if random.random() < geometry_on_ratio else "zero"
        geometry_on += int(mode == "full")
        cache_data = VGGTCache(cache_root).load(sample.sample_id)
        prompt_ids, answer_ids = _encode_prompt_answer(tokenizer, sample.question, sample.answer, max_text_tokens)
        prompt_ids = prompt_ids.to(device)
        answer_ids = answer_ids.to(device)
        input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        text_embeds = model.get_input_embeddings()(input_ids)
        geo = wrapper.geometry_inputs_embeds(cache_data, mode=mode).to(device=device, dtype=text_embeds.dtype)
        inputs_embeds = torch.cat([geo, text_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        labels = torch.full(inputs_embeds.shape[:2], -100, dtype=torch.long, device=device)
        labels[:, geo.shape[1] + prompt_ids.shape[1] :] = answer_ids

        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
        rows.append(
            {
                "step": step + 1,
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "loss": float(out.loss.detach().float().cpu()),
                "geometry_mode": mode,
                "gate": float(wrapper.geometry_gate.value().detach().float().mean().cpu()),
                "seq_len": int(inputs_embeds.shape[1]),
            }
        )

    output = ensure_dir(output)
    write_jsonl(output / "metrics.jsonl", rows)
    final = {
        **rows[-1],
        "geometry_on_ratio_observed": geometry_on / max(1, steps),
        "usable_cached_samples": len(usable),
        "model_path": str(model_path),
    }
    save_adapter(output, wrapper, final)
    model.save_pretrained(output / "lora_adapter")
    return final


def _cached_samples(jsonl: Path, cache_root: Path, *, limit: int) -> list[tuple[object, Path]]:
    cache = VGGTCache(cache_root)
    out = []
    for row in iter_jsonl(jsonl):
        sample = normalize_sample(row, default_dataset=jsonl.stem)
        if cache.exists(sample.sample_id):
            out.append((sample, cache_root))
        if len(out) >= limit:
            break
    return out


def _encode_prompt_answer(tokenizer, question: str, answer: str, max_text_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = f"Question: {question}\nAnswer:"
    answer_text = f" {answer}"
    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False, return_tensors="pt").input_ids
    if prompt_ids.shape[1] + answer_ids.shape[1] <= max_text_tokens:
        return prompt_ids, answer_ids
    answer_budget = min(answer_ids.shape[1], max(max_text_tokens // 2, 1))
    prompt_budget = max(max_text_tokens - answer_budget, 1)
    return prompt_ids[:, -prompt_budget:], answer_ids[:, :answer_budget]


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
    parser.add_argument("--spar-cache-root", type=Path)
    parser.add_argument("--llava-cache-root", type=Path)
    parser.add_argument("--qwen-smoke", action="store_true")
    parser.add_argument("--model-path", type=Path, default=Path("/mnt/guojh/lq/new/models/Qwen/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--num-geo-tokens", type=int, default=64)
    parser.add_argument("--lr-lora", type=float, default=1e-5)
    parser.add_argument("--lr-geo", type=float, default=5e-5)
    parser.add_argument("--max-text-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.smoke:
        print(run_smoke(args.output, args.steps, args.geometry_on_ratio))
        return
    if args.qwen_smoke:
        if not args.spar_jsonl or not args.llava_jsonl:
            raise SystemExit("Stage 2 Qwen smoke requires --spar-jsonl and --llava-jsonl")
        spar_cache = args.spar_cache_root or (args.cache_root / "spar" if args.cache_root else None)
        llava_cache = args.llava_cache_root or (args.cache_root / "llava_hound" if args.cache_root else None)
        if spar_cache is None or llava_cache is None:
            raise SystemExit("Stage 2 Qwen smoke requires --spar-cache-root and --llava-cache-root")
        print(
            run_qwen_smoke(
                output=args.output,
                model_path=args.model_path,
                spar_jsonl=args.spar_jsonl,
                llava_jsonl=args.llava_jsonl,
                spar_cache_root=spar_cache,
                llava_cache_root=llava_cache,
                steps=args.steps,
                geometry_on_ratio=args.geometry_on_ratio,
                num_geo_tokens=args.num_geo_tokens,
                lr_lora=args.lr_lora,
                lr_geo=args.lr_geo,
                max_text_tokens=args.max_text_tokens,
            )
        )
        return
    if not args.stage1_checkpoint or not args.spar_jsonl or not args.llava_jsonl or not args.cache_root:
        raise SystemExit("Stage 2 real run requires --stage1-checkpoint, --spar-jsonl, --llava-jsonl and --cache-root")
    write_json(args.output / "launch_plan.json", vars(args))
    raise SystemExit("Stage 2 launch plan written; run the server script with real Qwen/LoRA settings.")


if __name__ == "__main__":
    main()
