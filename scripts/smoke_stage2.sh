#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT="${OUTPUT:-outputs/smoke_stage2}"

"${PYTHON_BIN}" -m geopsro4d.train.train_stage2_sft --smoke --output "${OUTPUT}"
