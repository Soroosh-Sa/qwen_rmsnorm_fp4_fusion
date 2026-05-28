#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection
MODEL_TO_CHECK="${MODEL_TO_CHECK:-${MODEL_PATH:-$ORIGINAL_MODEL_PATH}}"
MAX_GPUS_FOR_TP="${MAX_GPUS_FOR_TP:-${GPU_COUNT:-8}}"
python src/check_tp_compatibility.py \
  --model "$MODEL_TO_CHECK" \
  --tp-size "${TP_SIZE:-1}" \
  --max-gpus "$MAX_GPUS_FOR_TP"
