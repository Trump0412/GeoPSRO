#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT="${OUTPUT:-outputs/smoke_stage3}"

"${PYTHON_BIN}" -m geopsro4d.train.train_stage3_rft --smoke --output "${OUTPUT}"
