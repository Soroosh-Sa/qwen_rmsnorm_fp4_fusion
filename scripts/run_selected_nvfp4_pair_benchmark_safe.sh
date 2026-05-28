#!/bin/bash
set -euo pipefail
# Same as run_selected_nvfp4_pair_benchmark.sh, but defaults to AUTO_ADJUST_TP_SIZE=1.
# For Qwen2.5-0.5B, TP_SIZE=4 is invalid; the script will auto-downgrade to 2.
export AUTO_ADJUST_TP_SIZE="${AUTO_ADJUST_TP_SIZE:-1}"
bash "$(dirname "$0")/run_selected_nvfp4_pair_benchmark.sh"
