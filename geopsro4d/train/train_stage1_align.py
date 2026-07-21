from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

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


def _setup_distributed() -> dict[str, int | bool | torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return {"distributed": distributed, "world_size": world_size, "rank": rank, "local_rank": local_rank, "device": device}


def _unwrap_model(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def _barrier(dist_ctx: dict[str, int | bool | torch.device]) -> None:
    if dist_ctx["distributed"] and dist.is_initialized():
        dist.barrier()


def _cleanup_distributed(dist_ctx: dict[str, int | bool | torch.device]) -> None:
    if dist_ctx["distributed"] and dist.is_initialized():
        dist.destroy_process_group()


def _average_gradients(parameters, dist_ctx: dict[str, int | bool | torch.device]) -> None:
    if not dist_ctx["distributed"] or not dist.is_initialized():
        return
    world_size = int(dist_ctx["world_size"])
    for param in parameters:
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


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
    model_dtype: str,
    batch_size: int,
    max_cached_samples: int | None,
    log_every: int,
    save_every: int,
) -> dict[str, float | int | str]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dist_ctx = _setup_distributed()
    seed_everything(3417 + dist_ctx["rank"])
    device = dist_ctx["device"]
    is_main = dist_ctx["rank"] == 0
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer = processor.tokenizer
    dtype = torch.float32
    if model_dtype == "bfloat16" and device.type == "cuda":
        dtype = torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        dtype=dtype,
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
    usable = []
    for row in iter_jsonl(train_jsonl):
        sample = normalize_sample(row, default_dataset=train_jsonl.stem)
        if cache.exists(sample.sample_id):
            usable.append(sample)
        if max_cached_samples is not None and len(usable) >= max_cached_samples:
            break
    if not usable:
        raise SystemExit(f"No cached samples found under {cache_root}")
    if dist_ctx["distributed"]:
        usable = usable[dist_ctx["rank"] :: dist_ctx["world_size"]]
        if not usable:
            raise SystemExit(f"No cached samples assigned to rank {dist_ctx['rank']}")

    rows = []
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / ("metrics.jsonl" if is_main else f"metrics_rank{dist_ctx['rank']}.jsonl")
    sample_iter = itertools.cycle(usable)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(steps):
            batch = []
            for _ in range(batch_size):
                sample = next(sample_iter)
                cache_data = cache.load(sample.sample_id)
                batch.append(_make_training_example(model, tokenizer, wrapper, sample, cache_data, max_text_tokens, device))

            inputs_embeds, attention_mask, labels = _pad_training_examples(batch, device)
            out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(out.loss.detach()):
                row = _make_step_row(step + 1, batch, _unwrap_model(wrapper), out.loss, inputs_embeds)
                row.update({"loss": float("nan"), "skipped": 1, "skip_reason": "nonfinite_loss"})
                rows.append(row)
                _write_metric(metrics_file, row, step + 1, log_every if is_main else 0)
                continue
            out.loss.backward()
            _average_gradients(wrapper.trainable_geo_parameters(), dist_ctx)
            opt.step()
            row = _make_step_row(step + 1, batch, wrapper, out.loss, inputs_embeds)
            row.update({"loss": float(out.loss.detach().float().cpu()), "skipped": 0})
            rows.append(row)
            _write_metric(metrics_file, row, step + 1, log_every if is_main else 0)
            if save_every > 0 and (step + 1) % save_every == 0:
                if is_main:
                    save_adapter(output / f"checkpoint_step_{step + 1}", wrapper, row)
                _barrier(dist_ctx)

    final = {
        **rows[-1],
        "usable_cached_samples": len(usable),
        "model_path": str(model_path),
        "batch_size": batch_size,
        "world_size": dist_ctx["world_size"],
        "global_batch_size": batch_size * dist_ctx["world_size"],
        "rank": dist_ctx["rank"],
    }
    if is_main:
        save_adapter(output, wrapper, final)
    _barrier(dist_ctx)
    _cleanup_distributed(dist_ctx)
    return final


def _make_training_example(model, tokenizer, wrapper, sample, cache_data, max_text_tokens: int, device: torch.device) -> dict[str, object]:
    prompt_ids, answer_ids = _encode_prompt_answer(tokenizer, sample.question, sample.answer, max_text_tokens)
    prompt_ids = prompt_ids.to(device)
    answer_ids = answer_ids.to(device)
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    text_embeds = model.get_input_embeddings()(input_ids)
    if isinstance(wrapper, torch.nn.parallel.DistributedDataParallel):
        geo = wrapper(cache_data, mode="full")
    else:
        geo = wrapper.geometry_inputs_embeds(cache_data, mode="full")
    geo = geo.to(device=device, dtype=text_embeds.dtype)
    seq_embeds = torch.cat([geo, text_embeds], dim=1).squeeze(0)
    seq_labels = torch.full((seq_embeds.shape[0],), -100, dtype=torch.long, device=device)
    seq_labels[geo.shape[1] + prompt_ids.shape[1] :] = answer_ids.squeeze(0)
    return {
        "sample_id": sample.sample_id,
        "inputs_embeds": seq_embeds,
        "labels": seq_labels,
        "seq_len": int(seq_embeds.shape[0]),
    }


def _pad_training_examples(batch: list[dict[str, object]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(int(example["seq_len"]) for example in batch)
    hidden = int(batch[0]["inputs_embeds"].shape[-1])
    dtype = batch[0]["inputs_embeds"].dtype
    inputs_embeds = torch.zeros(len(batch), max_len, hidden, dtype=dtype, device=device)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long, device=device)
    for idx, example in enumerate(batch):
        seq_len = int(example["seq_len"])
        inputs_embeds[idx, :seq_len] = example["inputs_embeds"]
        attention_mask[idx, :seq_len] = 1
        labels[idx, :seq_len] = example["labels"]
    return inputs_embeds, attention_mask, labels


def _make_step_row(step: int, batch: list[dict[str, object]], wrapper: QwenVGGTWrapper, loss: torch.Tensor, inputs_embeds: torch.Tensor) -> dict[str, object]:
    row: dict[str, object] = {
        "step": step,
        "loss": float(loss.detach().float().cpu()) if torch.isfinite(loss.detach()) else float("nan"),
        "gate": float(wrapper.geometry_gate.value().detach().float().mean().cpu()),
        "batch_size": len(batch),
        "seq_len_max": int(inputs_embeds.shape[1]),
        "seq_len_min": min(int(example["seq_len"]) for example in batch),
        "sample_ids": [str(example["sample_id"]) for example in batch[:4]],
    }
    if len(batch) == 1:
        row.update({"sample_id": str(batch[0]["sample_id"]), "seq_len": int(batch[0]["seq_len"])})
    return row


def _write_metric(metrics_file, row: dict[str, object], step: int, log_every: int) -> None:
    metrics_file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    if log_every > 0 and step % log_every == 0:
        metrics_file.flush()
        print(row, flush=True)


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
    parser.add_argument("--model-path", type=Path, default=Path("models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--num-geo-tokens", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-text-tokens", type=int, default=1024)
    parser.add_argument("--model-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-cached-samples", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    args = parser.parse_args()
    if args.smoke:
        print(run_smoke(args.output, args.steps))
        return
    if args.qwen_smoke:
        if not args.train_jsonl or not args.cache_root:
            raise SystemExit("Qwen Stage 1 smoke requires --train-jsonl and --cache-root")
        result = run_qwen_smoke(
            output=args.output,
            model_path=args.model_path,
            train_jsonl=args.train_jsonl,
            cache_root=args.cache_root,
            steps=args.steps,
            num_geo_tokens=args.num_geo_tokens,
            lr=args.learning_rate,
            max_text_tokens=args.max_text_tokens,
            model_dtype=args.model_dtype,
            batch_size=args.batch_size,
            max_cached_samples=args.max_cached_samples,
            log_every=args.log_every,
            save_every=args.save_every,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(result)
        return
    if not args.train_jsonl or not args.cache_root:
        raise SystemExit("Stage 1 real run requires --train-jsonl and --cache-root")
    result = run_qwen_smoke(
        output=args.output,
        model_path=args.model_path,
        train_jsonl=args.train_jsonl,
        cache_root=args.cache_root,
        steps=args.steps,
        num_geo_tokens=args.num_geo_tokens,
        lr=args.learning_rate,
        max_text_tokens=args.max_text_tokens,
        model_dtype=args.model_dtype,
        batch_size=args.batch_size,
        max_cached_samples=args.max_cached_samples,
        log_every=args.log_every,
        save_every=args.save_every,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(result)


if __name__ == "__main__":
    main()
