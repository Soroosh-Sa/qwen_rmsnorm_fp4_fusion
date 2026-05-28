#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs metrics outputs
python src/fold_checkpoint.py --config configs/qwen_small.yaml 2>&1 | tee logs/02_fold_checkpoint.log
