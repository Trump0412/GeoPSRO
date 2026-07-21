# gap-4D

gap-4D is a geometry-augmented post-training pipeline for dynamic 4D spatial
reasoning. It uses frozen geometry features during alignment and supervised
fine-tuning, then applies process-oriented reinforcement fine-tuning.

```text
Stage 1: VGGT geometry alignment
Stage 2: geometry-augmented spatial SFT
Stage 3: GSPO/PSRO-style reinforcement fine-tuning
```

Generated datasets, checkpoints, caches, logs, and evaluation artifacts are not
stored in this repository.

## Installation

```bash
python -m pip install -e .
python -m pytest -q
```

## Quick Smoke

```bash
bash scripts/smoke_stage1.sh
bash scripts/smoke_stage2.sh
bash scripts/smoke_stage3.sh
```

## Configure Paths

Set local paths before real training:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export PYTHON_BIN=python
export MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct
export VGGT_MODEL=/path/to/VGGT-1B
export VGGT_SOURCE=/path/to/vggt/source
export VERL_ROOT=/path/to/verl
```

## Main Commands

```bash
bash scripts/run_vggt_cache.sh
bash scripts/run_stage1_align.sh
bash scripts/run_stage2_sft.sh
bash scripts/run_stage3_rft.sh
bash scripts/run_eval_dsr.sh outputs/stage3_psro_rft normal
```

The Stage 3 answer format is:

```text
<think>
Spatial Observation:
Spatial Transition:
Answer Derivation:
</think>
<answer>
final answer only
</answer>
```

## Notes

- Keep benchmark annotations and media outside Git.
- Pass machine-specific paths through environment variables or config files.
