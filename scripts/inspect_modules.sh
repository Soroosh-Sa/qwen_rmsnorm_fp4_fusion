#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src:${PYTHONPATH:-}
python src/inspect_qwen_modules.py --config configs/qwen_small.yaml 2>&1 | tee logs/inspect_modules.log
