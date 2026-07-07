from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch

from geopsro4d.data.schema import normalize_sample
from geopsro4d.geometry.vggt_cache import VGGTCache
from geopsro4d.model.qwen_vggt_wrapper import QwenVGGTWrapper
from geopsro4d.train.common import make_smoke_cache, make_smoke_model, save_adapter
from geopsro4d.utils.io import iter_jsonl
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


def run_qwen_smoke(
    *,
    output: Path,
    model_path: Path,
    train_jsonl: Path,
    cache_root: Path,
    steps: int,
    num_geo_tokens: int,
    lr: float,
    max_text_tokens: int,
) -> dict[str, float | int | str]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    seed_everything(3417)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    hidden_size = int(model.config.text_config.hidden_size)
    wrapper = QwenVGGTWrapper(
        None,
        llm_dim=hidden_size,
        raw_geo_dim=32,
        adapter_dim=256,
        num_geo_tokens=num_geo_tokens,
    ).to(device)
    opt = torch.optim.AdamW(wrapper.trainable_geo_parameters(), lr=lr)
    cache = VGGTCache(cache_root)
    samples = [normalize_sample(row, default_dataset=train_jsonl.stem) for row in iter_jsonl(train_jsonl)]
    usable = [sample for sample in samples if cache.exists(sample.sample_id)]
    if not usable:
        raise SystemExit(f"No cached samples found under {cache_root}")

    rows = []
    for step, sample in zip(range(steps), itertools.cycle(usable)):
        cache_data = cache.load(sample.sample_id)
        prompt_ids, answer_ids = _encode_prompt_answer(tokenizer, sample.question, sample.answer, max_text_tokens)
        prompt_ids = prompt_ids.to(device)
        answer_ids = answer_ids.to(device)
        input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        text_embeds = model.get_input_embeddings()(input_ids)
        geo = wrapper.geometry_inputs_embeds(cache_data, mode="full").to(device=device, dtype=text_embeds.dtype)
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
                "loss": float(out.loss.detach().float().cpu()),
                "gate": float(wrapper.geometry_gate.value().detach().float().mean().cpu()),
                "seq_len": int(inputs_embeds.shape[1]),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "metrics.jsonl", rows)
    final = {**rows[-1], "usable_cached_samples": len(usable), "model_path": str(model_path)}
    save_adapter(output, wrapper, final)
    return final


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
    parser.add_argument("--output", type=Path, default=Path("outputs/stage1_vggt_align"))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--qwen-smoke", action="store_true")
    parser.add_argument("--model-path", type=Path, default=Path("/mnt/guojh/lq/new/models/Qwen/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--num-geo-tokens", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-text-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.smoke:
        print(run_smoke(args.output, args.steps))
        return
    if args.qwen_smoke:
        if not args.train_jsonl or not args.cache_root:
            raise SystemExit("Qwen Stage 1 smoke requires --train-jsonl and --cache-root")
        print(
            run_qwen_smoke(
                output=args.output,
                model_path=args.model_path,
                train_jsonl=args.train_jsonl,
                cache_root=args.cache_root,
                steps=args.steps,
                num_geo_tokens=args.num_geo_tokens,
                lr=args.learning_rate,
                max_text_tokens=args.max_text_tokens,
            )
        )
        return
    if not args.train_jsonl or not args.cache_root:
        raise SystemExit("Stage 1 real run requires --train-jsonl and --cache-root")
    raise SystemExit("Real Qwen Stage 1 launch is configured through scripts/run_stage1_align.sh; use --smoke for local validation.")


if __name__ == "__main__":
    main()
