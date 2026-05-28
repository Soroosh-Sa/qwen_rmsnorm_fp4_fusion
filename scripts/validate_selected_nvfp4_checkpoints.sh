#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/model_profiles.sh"

mkdir -p "metrics/${MODEL_TAG}"
TP_ARGS=()
if [[ -n "${TP_SIZE:-}" ]]; then
  TP_ARGS=(--expected-tp-size "$TP_SIZE")
fi

echo "Validating original NVFP4 checkpoint: $ORIGINAL_NVFP4_PATH"
python src/validate_trtllm_quantized_checkpoint.py \
  --model "$ORIGINAL_NVFP4_PATH" \
  "${TP_ARGS[@]}" \
  --report "metrics/${MODEL_TAG}/original_nvfp4_checkpoint_validation.json" || true

echo

echo "Validating folded NVFP4 checkpoint: $FOLDED_NVFP4_PATH"
python src/validate_trtllm_quantized_checkpoint.py \
  --model "$FOLDED_NVFP4_PATH" \
  "${TP_ARGS[@]}" \
  --report "metrics/${MODEL_TAG}/folded_nvfp4_checkpoint_validation.json" || true
