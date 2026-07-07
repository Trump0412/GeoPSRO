#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/guojh/lq/new/conda/envs/geobridge-verl/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
OUTPUT="${OUTPUT:-outputs/stage1_vggt_align}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

export PYTHONPATH="${PYTHONPATH:-$PWD}"

args=(
  --model-path "${MODEL_PATH:-/mnt/guojh/lq/new/models/Qwen/Qwen3-VL-2B-Instruct}"
  --train-jsonl "${TRAIN_JSONL:-/mnt/guojh/lq/new/datasets/manifests/geowire_formal_f8/spar.jsonl}"
  --cache-root "${CACHE_ROOT:-cache/vggt/spar}"
  --output "${OUTPUT}"
  --steps "${STEPS:-1000}"
  --batch-size "${BATCH_SIZE:-16}"
  --model-dtype "${MODEL_DTYPE:-bfloat16}"
  --learning-rate "${LEARNING_RATE:-1e-4}"
  --max-text-tokens "${MAX_TEXT_TOKENS:-1024}"
  --log-every "${LOG_EVERY:-10}"
  --save-every "${SAVE_EVERY:-0}"
)

if [[ -n "${MAX_CACHED_SAMPLES:-}" ]]; then
  args+=(--max-cached-samples "${MAX_CACHED_SAMPLES}")
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  "${TORCHRUN_BIN}" --standalone --nproc_per_node "${NPROC_PER_NODE}" -m geopsro4d.train.train_stage1_align "${args[@]}"
else
  "${PYTHON_BIN}" -m geopsro4d.train.train_stage1_align "${args[@]}"
fi
