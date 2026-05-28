#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs metrics
python src/verify_folding.py --config configs/qwen_small.yaml 2>&1 | tee logs/01_verify_folding.log
