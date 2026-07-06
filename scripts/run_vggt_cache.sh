#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/guojh/lq/new/conda/envs/geothinker/bin/python}"
VGGT_MODEL="${VGGT_MODEL:-/mnt/guojh/lq/new/weights/base_models/VGGT-1B}"
VGGT_SOURCE="${VGGT_SOURCE:-/mnt/guojh/lq/new/GeoWire/third_party/vggt}"
DATASET_JSON="${DATASET_JSON:-data/spar/train.jsonl}"
CACHE_ROOT="${CACHE_ROOT:-cache/vggt/spar}"
NUM_FRAMES="${NUM_FRAMES:-8}"
DEVICE="${DEVICE:-cuda}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="${VGGT_SOURCE}:$PWD:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m geopsro4d.geometry.vggt_extractor \
  --dataset_json "${DATASET_JSON}" \
  --cache_root "${CACHE_ROOT}" \
  --num_frames "${NUM_FRAMES}" \
  --model_name_or_path "${VGGT_MODEL}" \
  --source_path "${VGGT_SOURCE}" \
  --device "${DEVICE}"
