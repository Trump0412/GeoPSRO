#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/guojh/lq/new/conda/envs/geobridge-verl/bin/python}"
OUTPUT="${OUTPUT:-outputs/stage2_vggt_sft}"

export PYTHONPATH="${PYTHONPATH:-$PWD}"

args=(
  --stage1-checkpoint "${STAGE1_CHECKPOINT:-outputs/stage1_qwen2b_smoke_50/geo_adapter.pt}"
  --model-path "${MODEL_PATH:-/mnt/guojh/lq/new/models/Qwen/Qwen3-VL-2B-Instruct}"
  --spar-jsonl "${SPAR_JSONL:-/mnt/guojh/lq/new/datasets/manifests/geowire_formal_f8/spar.jsonl}"
  --llava-jsonl "${LLAVA_JSONL:-/mnt/guojh/lq/new/datasets/manifests/geowire_formal_f8/llava_hound.jsonl}"
  --spar-cache-root "${SPAR_CACHE_ROOT:-cache/vggt_pilot/spar_1k}"
  --llava-cache-root "${LLAVA_CACHE_ROOT:-cache/vggt_pilot/llava_hound_1k}"
  --output "${OUTPUT}"
  --steps "${STEPS:-1000}"
  --batch-size "${BATCH_SIZE:-1}"
  --geometry-on-ratio "${GEOMETRY_ON_RATIO:-0.7}"
  --model-dtype "${MODEL_DTYPE:-bfloat16}"
  --lr-lora "${LR_LORA:-1e-6}"
  --lr-geo "${LR_GEO:-1e-5}"
  --max-grad-norm "${MAX_GRAD_NORM:-1.0}"
  --log-every "${LOG_EVERY:-10}"
  --save-every "${SAVE_EVERY:-0}"
)

if [[ -n "${MAX_CACHED_SAMPLES_PER_DATASET:-}" ]]; then
  args+=(--max-cached-samples-per-dataset "${MAX_CACHED_SAMPLES_PER_DATASET}")
fi
if [[ -n "${MAX_TEXT_TOKENS:-}" ]]; then
  args+=(--max-text-tokens "${MAX_TEXT_TOKENS}")
fi

"${PYTHON_BIN}" -m geopsro4d.train.train_stage2_sft "${args[@]}"
