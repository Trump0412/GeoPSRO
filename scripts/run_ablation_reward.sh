#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" -m geopsro4d.eval.eval_reward_ablation \
  --output "${OUTPUT:-results/ablations/reward_ablation.csv}"
