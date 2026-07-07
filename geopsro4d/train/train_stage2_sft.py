from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist

from geopsro4d.data.schema import normalize_sample
from geopsro4d.geometry.vggt_cache import VGGTCache
from geopsro4d.model.qwen_vggt_wrapper import QwenVGGTWrapper
from geopsro4d.train.common import load_adapter, make_smoke_cache, make_smoke_model, save_adapter
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
    spar_jsonl: Path,
    llava_jsonl: Path,
    spar_cache_root: Path,
    llava_cache_root: Path,
    steps: int,
    geometry_on_ratio: float,
    num_geo_tokens: int,
    stage1_checkpoint: Path | None,
    lr_lora: float,
    lr_geo: float,
    max_text_tokens: int,
    max_grad_norm: float,
    model_dtype: str,
    batch_size: int,
    max_cached_samples_per_dataset: int | None,
    log_every: int,
    save_every: int,
) -> dict[str, float | int | str]:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dist_ctx = _setup_distributed()
    seed_everything(3418 + dist_ctx["rank"])
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
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.train()

    hidden_size = int(model.config.text_config.hidden_size)
    wrapper = QwenVGGTWrapper(
        None,
        llm_dim=hidden_size,
        raw_geo_dim=32,
        adapter_dim=256,
        num_geo_tokens=num_geo_tokens,
    ).to(device)
    stage1_metrics = {}
    if stage1_checkpoint:
        stage1_metrics = load_adapter(stage1_checkpoint, wrapper)
    if dist_ctx["distributed"]:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist_ctx["local_rank"]], find_unused_parameters=False)
    params = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": lr_lora},
        {"params": [p for p in wrapper.trainable_geo_parameters()], "lr": lr_geo},
    ]
    trainable_params = [p for group in params for p in group["params"]]
    opt = torch.optim.AdamW(params)

    sample_limit = max(steps * batch_size * 4, 128)
    if max_cached_samples_per_dataset is not None:
        sample_limit = max_cached_samples_per_dataset
    usable = []
    usable.extend(_cached_samples(spar_jsonl, spar_cache_root, limit=sample_limit))
    usable.extend(_cached_samples(llava_jsonl, llava_cache_root, limit=sample_limit))
    if not usable:
        raise SystemExit("No cached samples found for Stage 2 Qwen smoke")
    random.shuffle(usable)
    if dist_ctx["distributed"]:
        usable = usable[dist_ctx["rank"] :: dist_ctx["world_size"]]
        if not usable:
            raise SystemExit(f"No cached samples assigned to rank {dist_ctx['rank']}")
    usable_iter = itertools.cycle(usable)
    cache_handles: dict[Path, VGGTCache] = {}

    output = ensure_dir(output)
    metrics_path = output / ("metrics.jsonl" if is_main else f"metrics_rank{dist_ctx['rank']}.jsonl")
    rows = []
    geometry_on = 0
    total_examples = 0
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(steps):
            batch = []
            for _ in range(batch_size):
                sample, cache_root = next(usable_iter)
                mode = "full" if random.random() < geometry_on_ratio else "zero"
                geometry_on += int(mode == "full")
                total_examples += 1
                if mode == "full":
                    cache = cache_handles.setdefault(cache_root, VGGTCache(cache_root))
                    cache_data = cache.load(sample.sample_id)
                else:
                    cache_data = {}
                batch.append(_make_training_example(model, tokenizer, wrapper, sample, cache_data, mode, max_text_tokens, device))

            inputs_embeds, attention_mask, labels = _pad_training_examples(batch, device)
            out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            opt.zero_grad(set_to_none=True)
            row = _make_step_row(step + 1, batch, _unwrap_model(wrapper), out.loss, inputs_embeds)
            if not torch.isfinite(out.loss.detach()):
                row.update({"loss": float("nan"), "skipped": 1, "skip_reason": "nonfinite_loss"})
                rows.append(row)
                _write_metric(metrics_file, row, step + 1, log_every if is_main else 0)
                continue
            out.loss.backward()
            _average_gradients(wrapper.trainable_geo_parameters(), dist_ctx)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            if not torch.isfinite(grad_norm):
                opt.zero_grad(set_to_none=True)
                row.update(
                    {
                        "loss": float(out.loss.detach().float().cpu()),
                        "grad_norm": float("nan"),
                        "skipped": 1,
                        "skip_reason": "nonfinite_grad",
                    }
                )
                rows.append(row)
                _write_metric(metrics_file, row, step + 1, log_every if is_main else 0)
                continue
            opt.step()
            row.update({"loss": float(out.loss.detach().float().cpu()), "grad_norm": float(grad_norm.detach().float().cpu()), "skipped": 0})
            rows.append(row)
            _write_metric(metrics_file, row, step + 1, log_every if is_main else 0)
            if save_every > 0 and (step + 1) % save_every == 0:
                if is_main:
                    checkpoint_dir = output / f"checkpoint_step_{step + 1}"
                    save_adapter(checkpoint_dir, wrapper, row)
                    _unwrap_model(model).save_pretrained(checkpoint_dir / "lora_adapter")
                _barrier(dist_ctx)

    final = {
        **rows[-1],
        "geometry_on_ratio_observed": geometry_on / max(1, total_examples),
        "usable_cached_samples": len(usable),
        "world_size": dist_ctx["world_size"],
        "global_batch_size": batch_size * dist_ctx["world_size"],
        "rank": dist_ctx["rank"],
        "model_path": str(model_path),
        "stage1_checkpoint": str(stage1_checkpoint) if stage1_checkpoint else "",
        "stage1_loss": stage1_metrics.get("loss", ""),
        "batch_size": batch_size,
    }
    if is_main:
        save_adapter(output, wrapper, final)
        _unwrap_model(model).save_pretrained(output / "lora_adapter")
    _barrier(dist_ctx)
    _cleanup_distributed(dist_ctx)
    return final


def _make_training_example(model, tokenizer, wrapper, sample, cache_data, mode: str, max_text_tokens: int, device: torch.device) -> dict[str, object]:
    prompt_ids, answer_ids = _encode_prompt_answer(tokenizer, sample.question, sample.answer, max_text_tokens)
    prompt_ids = prompt_ids.to(device)
    answer_ids = answer_ids.to(device)
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    text_embeds = _unwrap_model(model).get_input_embeddings()(input_ids)
    if isinstance(wrapper, torch.nn.parallel.DistributedDataParallel):
        geo = wrapper(cache_data, mode=mode)
    else:
        geo = wrapper.geometry_inputs_embeds(cache_data, mode=mode)
    geo = geo.to(device=device, dtype=text_embeds.dtype)
    seq_embeds = torch.cat([geo, text_embeds], dim=1).squeeze(0)
    seq_labels = torch.full((seq_embeds.shape[0],), -100, dtype=torch.long, device=device)
    seq_labels[geo.shape[1] + prompt_ids.shape[1] :] = answer_ids.squeeze(0)
    return {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "mode": mode,
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
    modes = Counter(str(example["mode"]) for example in batch)
    datasets = Counter(str(example["dataset"]) for example in batch)
    row: dict[str, object] = {
        "step": step,
        "loss": float(loss.detach().float().cpu()) if torch.isfinite(loss.detach()) else float("nan"),
        "geometry_mode": next(iter(modes)) if len(modes) == 1 else "mixed",
        "geometry_full_count": modes.get("full", 0),
        "geometry_zero_count": modes.get("zero", 0),
        "datasets": dict(datasets),
        "gate": float(wrapper.geometry_gate.value().detach().float().mean().cpu()),
        "batch_size": len(batch),
        "seq_len_max": int(inputs_embeds.shape[1]),
        "seq_len_min": min(int(example["seq_len"]) for example in batch),
        "sample_ids": [str(example["sample_id"]) for example in batch[:4]],
    }
    if len(batch) == 1:
        row.update(
            {
                "sample_id": str(batch[0]["sample_id"]),
                "dataset": str(batch[0]["dataset"]),
                "seq_len": int(batch[0]["seq_len"]),
            }
        )
    return row


def _write_metric(metrics_file, row: dict[str, object], step: int, log_every: int) -> None:
    metrics_file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    if log_every > 0 and step % log_every == 0:
        metrics_file.flush()
        print(row, flush=True)


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
    parser.add_argument("--lr-lora", type=float, default=1e-6)
    parser.add_argument("--lr-geo", type=float, default=1e-5)
    parser.add_argument("--max-text-tokens", type=int, default=1024)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--model-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-cached-samples-per-dataset", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
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
        result = run_qwen_smoke(
            output=args.output,
            model_path=args.model_path,
            spar_jsonl=args.spar_jsonl,
            llava_jsonl=args.llava_jsonl,
            spar_cache_root=spar_cache,
            llava_cache_root=llava_cache,
            steps=args.steps,
            geometry_on_ratio=args.geometry_on_ratio,
            num_geo_tokens=args.num_geo_tokens,
            stage1_checkpoint=args.stage1_checkpoint,
            lr_lora=args.lr_lora,
            lr_geo=args.lr_geo,
            max_text_tokens=args.max_text_tokens,
            max_grad_norm=args.max_grad_norm,
            model_dtype=args.model_dtype,
            batch_size=args.batch_size,
            max_cached_samples_per_dataset=args.max_cached_samples_per_dataset,
            log_every=args.log_every,
            save_every=args.save_every,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(result)
        return
    if not args.stage1_checkpoint or not args.spar_jsonl or not args.llava_jsonl:
        raise SystemExit("Stage 2 real run requires --stage1-checkpoint, --spar-jsonl and --llava-jsonl")
    spar_cache = args.spar_cache_root or (args.cache_root / "spar" if args.cache_root else None)
    llava_cache = args.llava_cache_root or (args.cache_root / "llava_hound" if args.cache_root else None)
    if spar_cache is None or llava_cache is None:
        raise SystemExit("Stage 2 real run requires --cache-root or both --spar-cache-root and --llava-cache-root")
    result = run_qwen_smoke(
        output=args.output,
        model_path=args.model_path,
        spar_jsonl=args.spar_jsonl,
        llava_jsonl=args.llava_jsonl,
        spar_cache_root=spar_cache,
        llava_cache_root=llava_cache,
        steps=args.steps,
        geometry_on_ratio=args.geometry_on_ratio,
        num_geo_tokens=args.num_geo_tokens,
        stage1_checkpoint=args.stage1_checkpoint,
        lr_lora=args.lr_lora,
        lr_geo=args.lr_geo,
        max_text_tokens=args.max_text_tokens,
        max_grad_norm=args.max_grad_norm,
        model_dtype=args.model_dtype,
        batch_size=args.batch_size,
        max_cached_samples_per_dataset=args.max_cached_samples_per_dataset,
        log_every=args.log_every,
        save_every=args.save_every,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(result)


if __name__ == "__main__":
    main()
