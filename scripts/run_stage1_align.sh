#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/guojh/lq/new/conda/envs/geothinker/bin/python}"
OUTPUT="${OUTPUT:-outputs/stage1_vggt_align}"

"${PYTHON_BIN}" -m geopsro4d.train.train_stage1_align \
  --train-jsonl "${TRAIN_JSONL:-data/spar/train.jsonl}" \
  --cache-root "${CACHE_ROOT:-cache/vggt/spar}" \
  --output "${OUTPUT}"
