# GeoPSRO-4D

Geometry-augmented post-training pipeline for 4D spatial reasoning.

This repository implements the design in:

- `GeoPSRO4D_Core_Idea.md`
- `GeoPSRO4D_Codex_Code_Implementation_Instructions.md`

The method is intentionally **not** a trace-verifier system. It follows:

```text
Stage 1: VGGT Geometry Alignment
Stage 2: Geometry-Augmented Spatial SFT
Stage 3: PSRO-RFT
```

The PSRO format is introduced only in RFT:

```text
Observation:
Transition:
Derivation:
Answer:
```

## Quick Smoke

```bash
python -m pytest -q
python -m geopsro4d.train.train_stage1_align --smoke --output outputs/smoke_stage1
python -m geopsro4d.train.train_stage2_sft --smoke --output outputs/smoke_stage2
python -m geopsro4d.train.train_stage3_rft --smoke --output outputs/smoke_stage3
```

## Server Defaults

The scripts default to the shared server layout used by previous GeoThinker/GeoBridge work:

```text
/mnt/guojh/lq/new/conda/envs/geothinker
/mnt/guojh/lq/new/weights/base_models/Qwen3-VL-4B-Instruct
/mnt/guojh/lq/new/weights/base_models/VGGT-1B
/mnt/guojh/lq/new/verl
```

Use `HF_ENDPOINT=https://hf-mirror.com` for Hugging Face downloads.

## Main Commands

```bash
bash scripts/run_vggt_cache.sh
bash scripts/run_stage1_align.sh
bash scripts/run_stage2_sft.sh
bash scripts/run_stage3_rft.sh
bash scripts/run_eval_dsr.sh outputs/stage3_psro_rft normal
```

Generated data, cache, checkpoints, and result artifacts are intentionally ignored by Git.
