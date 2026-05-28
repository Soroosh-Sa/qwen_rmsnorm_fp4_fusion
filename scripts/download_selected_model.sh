#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

mkdir -p "$MODEL_ROOT" logs
print_model_selection

python src/download_hf_checkpoint.py \
  --model "$QWEN_MODEL_ID" \
  --output-dir "$ORIGINAL_MODEL_PATH" | tee "logs/download_${MODEL_TAG}.log"

echo "Downloaded selected model to: $ORIGINAL_MODEL_PATH"
