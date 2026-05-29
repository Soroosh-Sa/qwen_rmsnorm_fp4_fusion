#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection

# Fast, explicit command for the current safe NVFP4 path:
# folded BF16 -> folded NVFP4 -> dual RMS-scale-SwiGLU plugin -> engine.
# The plugin returns BF16/FP16 intermediate. The existing NVFP4 proj linear
# quantizes internally through TensorRT-LLM's NVFP4 GEMM path.

export TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION=1
export TRTLLM_QWEN_RMS_SCALE_SWIGLU_PLUGIN=1
export TRTLLM_QWEN_RMS_SCALE_SWIGLU_PLUGIN_MODE="${TRTLLM_QWEN_RMS_SCALE_SWIGLU_PLUGIN_MODE:-bf16_intermediate}"
export TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION_ALLOW_QUANTIZED=1

if [[ ! -d "$FOLDED_NVFP4_PATH" || "${FORCE_QUANTIZE:-0}" == "1" ]]; then
  echo "Creating folded NVFP4 checkpoint from folded BF16 weights..."
  export INPUT_MODEL="$FOLDED_BF16_PATH"
  export OUTPUT_DIR="$FOLDED_NVFP4_PATH"
  bash scripts/quantize_selected_folded_nvfp4.sh
fi

bash scripts/run_folded_nvfp4_contract_check.sh
bash scripts/build_qwen_rms_scale_swiglu_plugin.sh
bash scripts/build_selected_folded_nvfp4_plugin_engine.sh

echo "DONE"
echo "Folded NVFP4 checkpoint: $FOLDED_NVFP4_PATH"
echo "Folded NVFP4 plugin engine: ${FOLDED_NVFP4_PLUGIN_ENGINE:-${ENGINE_ROOT}/${MODEL_TAG}-FOLDED-NVFP4-rms-scale-swiglu-plugin-engine-tp${TP_SIZE}}"
