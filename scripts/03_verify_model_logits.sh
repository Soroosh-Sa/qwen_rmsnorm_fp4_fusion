#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src:${PYTHONPATH:-}
mkdir -p logs metrics
python src/verify_model_logits.py --config configs/qwen_small.yaml 2>&1 | tee logs/03_verify_model_logits.log
