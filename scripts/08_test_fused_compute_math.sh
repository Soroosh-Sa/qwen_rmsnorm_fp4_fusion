#!/usr/bin/env bash
set -euo pipefail
python src/fused_runtime_notes.py | tee logs/fused_compute_math.log
