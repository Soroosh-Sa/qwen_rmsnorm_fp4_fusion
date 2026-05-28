#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection

# Quantize both versions of the same selected small model:
#   original BF16 -> NVFP4
#   folded BF16   -> NVFP4
# This is the controlled small-model experiment before 480B.

if [[ ! -d "$ORIGINAL_MODEL_PATH" ]]; then
  echo "Original model not found: $ORIGINAL_MODEL_PATH" >&2
  echo "Run: MODEL_PROFILE=$MODEL_PROFILE bash scripts/download_selected_model.sh" >&2
  exit 2
fi
if [[ ! -d "$FOLDED_BF16_PATH" ]]; then
  echo "Folded model not found: $FOLDED_BF16_PATH" >&2
  echo "Run: MODEL_PROFILE=$MODEL_PROFILE bash scripts/fold_selected_model_sharded.sh" >&2
  exit 2
fi

bash "$(dirname "$0")/quantize_selected_original_nvfp4.sh"
bash "$(dirname "$0")/quantize_selected_folded_nvfp4.sh"

echo "Done. Outputs:"
echo "  Original NVFP4: ${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-NVFP4"
echo "  Folded NVFP4:   $FOLDED_NVFP4_PATH"
