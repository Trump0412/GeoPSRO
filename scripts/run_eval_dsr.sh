#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-outputs/stage3_psro_rft}"
GEOMETRY_MODE="${2:-normal}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PREDICTIONS="${PREDICTIONS:-results/eval/dsr_predictions.jsonl}"
OUTPUT="${OUTPUT:-results/eval/dsr_${GEOMETRY_MODE}}"

echo "scoring checkpoint=${CHECKPOINT} geometry_mode=${GEOMETRY_MODE}"
"${PYTHON_BIN}" -m geopsro4d.eval.eval_dsr \
  --predictions "${PREDICTIONS}" \
  --output "${OUTPUT}" \
  --prompt-mode psro \
  --geometry-mode "${GEOMETRY_MODE}"
