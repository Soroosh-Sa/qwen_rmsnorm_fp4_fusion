#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection

# Fast path for the real workflow:
#   folded BF16 weights -> folded NVFP4 TRT-LLM checkpoint -> plugin engine.
# It intentionally does not quantize the original model because the plugin only
# makes sense for the folded-weight target.

if [[ ! -d "$FOLDED_BF16_PATH" ]]; then
  if [[ "${RUN_FOLD_IF_MISSING:-0}" == "1" ]]; then
    echo "Folded BF16 checkpoint missing; folding first: $FOLDED_BF16_PATH"
    bash scripts/fold_selected_model_sharded.sh
  else
    echo "ERROR: Folded BF16 checkpoint missing: $FOLDED_BF16_PATH" >&2
    echo "Run folding first, or set RUN_FOLD_IF_MISSING=1." >&2
    exit 2
  fi
fi

if [[ ! -d "$FOLDED_NVFP4_PATH" || "${FORCE_QUANTIZE:-0}" == "1" ]]; then
  echo "Creating folded NVFP4 TensorRT-LLM checkpoint from folded BF16 weights..."
  export INPUT_MODEL="$FOLDED_BF16_PATH"
  export OUTPUT_DIR="$FOLDED_NVFP4_PATH"
  bash scripts/quantize_selected_folded_nvfp4.sh
else
  echo "Using existing folded NVFP4 checkpoint: $FOLDED_NVFP4_PATH"
fi

bash scripts/build_qwen_rms_scale_swiglu_plugin.sh
bash scripts/build_selected_folded_nvfp4_plugin_engine.sh

echo "DONE: folded NVFP4 plugin pipeline"
echo "  FOLDED_BF16_PATH=$FOLDED_BF16_PATH"
echo "  FOLDED_NVFP4_PATH=$FOLDED_NVFP4_PATH"
echo "  FOLDED_NVFP4_PLUGIN_ENGINE=${FOLDED_NVFP4_PLUGIN_ENGINE:-${ENGINE_ROOT}/${MODEL_TAG}-FOLDED-NVFP4-rms-scale-swiglu-plugin-engine-tp${TP_SIZE}}"
