# GeoPSRO-4D Progress

Last updated: 2026-07-07

## Implemented

- Repository structure requested by the implementation instructions.
- Shared SFT/RFT/eval sample schema normalization.
- Frame sampling for multi-image inputs, plus optional video extraction with OpenCV.
- VGGT cache interface and extraction entrypoint.
- Geometry token construction from camera, depth, point map, confidence, and tracks.
- Geo Resampler, Geo Projector, and Geometry Gate.
- Qwen-VGGT wrapper support for geometry modes:
  `normal`, `zero`, `drop`, `shuffle`, `depth_only`, `depth_camera`, `full`.
- Stage 1 alignment smoke training.
- Stage 2 geometry-augmented SFT smoke training with geometry drop logging.
- Stage 3 PSRO-RFT data preparation and reward-log smoke.
- PSRO answer parser, section parser, weak process rules, and reward function.
- DSR-style scorer plus geometry/reward ablation output scaffolds.
- Restored GeoBridge-compatible Stage 3 reward as the main reward:
  `R_answer + 0.5 * R_format + 0.05 * R_words`, with the original spatial vocabulary scoring and answer-gated word reward.
- Added `GeoPSRO_METHOD_TRAINING_DETAILS.md` as the canonical method/training
  record for prompt format, losses, data mixtures, batch plans, and go/no-go checks.
- Run scripts and YAML configs for VGGT cache, Stage 1, Stage 2, Stage 3, DSR eval, and ablations.

## Validation

```text
python -m pytest -q
8 passed
```

Smoke commands passed:

```bash
python -m geopsro4d.train.train_stage1_align --smoke --output outputs/smoke_stage1
python -m geopsro4d.train.train_stage2_sft --smoke --output outputs/smoke_stage2
python -m geopsro4d.train.train_stage3_rft --smoke --output outputs/smoke_stage3
python -m geopsro4d.eval.eval_dsr --predictions outputs/eval_smoke_predictions.jsonl --output outputs/eval_smoke --prompt-mode psro --geometry-mode normal
python -m geopsro4d.eval.eval_geometry_ablation --predictions outputs/eval_smoke_predictions.jsonl --output outputs/geometry_ablation_smoke.csv
```

## 2026-07-07 Stage 2 Formal Queue

- Stage 1 status: only `outputs/stage1_qwen2b_smoke_50/geo_adapter.pt` has
  completed for Qwen3-VL-2B. A long formal Stage 1 checkpoint was not found.
- Stage 2 datasets selected for the formal run:
  SPAR-234K (`234056` samples) plus LLaVA-Hound (`63750` samples).
- Full compact VGGT cache is required before the formal SFT run. The previous
  pilot cache only covered `1024 + 1024` samples.
- Node `10.99.8.18` / `node218` has 8 free A100 80GB GPUs and can access the
  shared `/mnt/guojh/lq/new` mount.
- Full Stage 2 cache queue is running on node218:
  6 shards for SPAR on GPUs 0-5 and 2 shards for LLaVA-Hound on GPUs 6-7.
- Stage 2 SFT now supports `torchrun` DDP. A 2-GPU DDP smoke passed with
  `world_size=2` and `skipped=0`.
- A watcher tmux session `geopsro_stage2_wait_and_train` is queued on node218.
  It waits for the full cache to finish, checks cache count thresholds, and then
  starts Stage 2 SFT with:

```text
NPROC_PER_NODE=8
BATCH_SIZE=16
global_batch_size=128
MODEL_DTYPE=bfloat16
SPAR_CACHE_ROOT=cache/vggt/spar
LLAVA_CACHE_ROOT=cache/vggt/llava_hound
MAX_CACHED_SAMPLES_PER_DATASET=300000
```

The watcher logs to `logs/stage2_wait_and_train.log`. Formal Stage 2 output will
use the pattern `outputs/stage2_sft_spar234k_llavahound64k_b16x8_*`.

## Server Defaults

The scripts default to:

```text
Qwen3-VL-2B: /mnt/guojh/lq/new/models/Qwen/Qwen3-VL-2B-Instruct
VGGT-1B: /mnt/guojh/lq/new/weights/base_models/VGGT-1B
VGGT source: /mnt/guojh/lq/new/GeoWire/third_party/vggt
Qwen/Stage 2 env: /mnt/guojh/lq/new/conda/envs/geobridge-verl
VGGT cache env: /mnt/guojh/lq/new/conda/envs/geothinker
VERL root: /mnt/guojh/lq/new/verl
```

## Next Real-Run Inputs

- `data/spar/train.jsonl` and `data/spar/val.jsonl`
- `data/llava_hound/train.jsonl`
- `data/spatialladder/train.jsonl`
- `data/4dthinker/train.jsonl`
- `data/dsr/test.jsonl`
- VGGT caches under `cache/vggt/...`
- Stage 1 and Stage 2 real checkpoints before full Stage 3 VERL launch

## Boundaries

- No trace verifier, evidence graph, or trace supervision modules are implemented.
- PSRO formatting is only introduced in Stage 3 RFT and PSRO evaluation prompts.
- Stage 1/2 keep original answer style.
