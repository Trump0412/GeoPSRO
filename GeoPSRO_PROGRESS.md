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

## Server Defaults

The scripts default to:

```text
Qwen3-VL-4B: /mnt/guojh/lq/new/weights/base_models/Qwen3-VL-4B-Instruct
VGGT-1B: /mnt/guojh/lq/new/weights/base_models/VGGT-1B
VGGT source: /mnt/guojh/lq/new/GeoWire/third_party/vggt
Conda env: /mnt/guojh/lq/new/conda/envs/geothinker
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
