#!/bin/bash
set -euo pipefail
# Safe wrapper for the current Stage-6 folded-NVFP4 base-vs-plugin benchmark.
# Defaults to AUTO_ADJUST_TP_SIZE=1 for old workflows; the current engine path
# also reads TP_SIZE from checkpoint config when available.
export AUTO_ADJUST_TP_SIZE="${AUTO_ADJUST_TP_SIZE:-1}"
exec bash "$(dirname "$0")/run_selected_nvfp4_pair_benchmark.sh"
