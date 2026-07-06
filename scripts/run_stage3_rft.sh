#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/guojh/lq/new/conda/envs/geothinker/bin/python}"
OUTPUT="${OUTPUT:-outputs/stage3_psro_rft}"

"${PYTHON_BIN}" -m geopsro4d.train.train_stage3_rft \
  --rft-jsonl "${RFT_JSONL:-data/4dthinker/train.jsonl}" \
  --stage2-checkpoint "${STAGE2_CHECKPOINT:-outputs/stage2_vggt_sft}" \
  --output "${OUTPUT}" \
  --backend "${BACKEND:-prepare}"
