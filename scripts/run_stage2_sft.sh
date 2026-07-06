#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/guojh/lq/new/conda/envs/geothinker/bin/python}"
OUTPUT="${OUTPUT:-outputs/stage2_vggt_sft}"

"${PYTHON_BIN}" -m geopsro4d.train.train_stage2_sft \
  --stage1-checkpoint "${STAGE1_CHECKPOINT:-outputs/stage1_vggt_align/geo_adapter.pt}" \
  --spar-jsonl "${SPAR_JSONL:-data/spar/train.jsonl}" \
  --llava-jsonl "${LLAVA_JSONL:-data/llava_hound/train.jsonl}" \
  --cache-root "${CACHE_ROOT:-cache/vggt}" \
  --output "${OUTPUT}"
