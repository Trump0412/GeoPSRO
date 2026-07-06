#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PREDICTIONS="${PREDICTIONS:-results/eval/dsr_predictions.jsonl}"

"${PYTHON_BIN}" -m geopsro4d.eval.eval_geometry_ablation \
  --predictions "${PREDICTIONS}" \
  --output "${OUTPUT:-results/ablations/geometry_ablation.csv}"
