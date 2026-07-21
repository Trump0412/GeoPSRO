#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
VGGT_MODEL="${VGGT_MODEL:-models/VGGT-1B}"
VGGT_SOURCE="${VGGT_SOURCE:-external/vggt}"
DATASET_JSON="${DATASET_JSON:-data/spar/train.jsonl}"
CACHE_ROOT="${CACHE_ROOT:-cache/vggt/spar}"
NUM_FRAMES="${NUM_FRAMES:-8}"
DEVICE="${DEVICE:-cuda}"
OVERWRITE="${OVERWRITE:-false}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
LOG_EVERY="${LOG_EVERY:-25}"
CACHE_RESOLUTION="${CACHE_RESOLUTION:-16}"
CACHE_DTYPE="${CACHE_DTYPE:-float16}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="${VGGT_SOURCE}:$PWD:${PYTHONPATH:-}"

args=(
  --dataset_json "${DATASET_JSON}"
  --cache_root "${CACHE_ROOT}"
  --num_frames "${NUM_FRAMES}"
  --model_name_or_path "${VGGT_MODEL}"
  --source_path "${VGGT_SOURCE}"
  --device "${DEVICE}"
  --overwrite "${OVERWRITE}"
  --num_shards "${NUM_SHARDS}"
  --shard_index "${SHARD_INDEX}"
  --log_every "${LOG_EVERY}"
  --cache_resolution "${CACHE_RESOLUTION}"
  --cache_dtype "${CACHE_DTYPE}"
)
if [[ -n "${MAX_SAMPLES}" ]]; then
  args+=(--max_samples "${MAX_SAMPLES}")
fi

"${PYTHON_BIN}" -m geopsro4d.geometry.vggt_extractor "${args[@]}"
