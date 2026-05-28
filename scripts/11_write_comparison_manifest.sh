#!/usr/bin/env bash
set -euo pipefail
python src/write_comparison_manifest.py \
  --out metrics/comparison_manifest.json \
  --csv metrics/comparison_targets.csv
