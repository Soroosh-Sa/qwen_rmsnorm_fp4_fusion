#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src:${PYTHONPATH:-}
python src/summarize_metrics.py
