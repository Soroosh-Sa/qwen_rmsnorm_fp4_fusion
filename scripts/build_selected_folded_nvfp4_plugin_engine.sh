#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

if [[ ! -d "$FOLDED_NVFP4_PATH" ]]; then
  echo "ERROR: Folded NVFP4 TensorRT-LLM checkpoint not found: $FOLDED_NVFP4_PATH" >&2
  echo "Create it first with:" >&2
  echo "  MODEL_PROFILE=$MODEL_PROFILE TP_SIZE=$TP_SIZE bash scripts/quantize_selected_folded_nvfp4.sh" >&2
  exit 2
fi

PLUGIN_SO_DEFAULT="$REPO_ROOT/build/qwen_rms_scale_swiglu_plugin/libqwen_rms_scale_swiglu_plugin.so"
export QWEN_RMS_SCALE_SWIGLU_PLUGIN_SO="${QWEN_RMS_SCALE_SWIGLU_PLUGIN_SO:-$PLUGIN_SO_DEFAULT}"

if [[ ! -f "$QWEN_RMS_SCALE_SWIGLU_PLUGIN_SO" ]]; then
  echo "Plugin .so not found, building it first..."
  bash scripts/build_qwen_rms_scale_swiglu_plugin.sh
fi

if [[ ! -f "$QWEN_RMS_SCALE_SWIGLU_PLUGIN_SO" ]]; then
  echo "ERROR: plugin .so not found after build: $QWEN_RMS_SCALE_SWIGLU_PLUGIN_SO" >&2
  exit 3
fi

# Apply the model.py TensorRT graph patch by default, because the engine build
# must insert the plugin layer into the TensorRT network.
if [[ "${APPLY_TRTLLM_PATCHES:-1}" == "1" ]]; then
  bash patches/apply_trtllm_patches.sh
fi

# Prefer TP size stored in the folded NVFP4 checkpoint config. It is the source
# of truth for rank*.safetensors.
CKPT_TP="$($PYTHON_BIN - <<PY
import json, os
p=os.path.join("$FOLDED_NVFP4_PATH", "config.json")
try:
    c=json.load(open(p)); m=c.get("mapping",{}) or {}; print(m.get("tp_size") or m.get("attn_tp_size") or os.environ.get("TP_SIZE", "1"))
except Exception:
    print(os.environ.get("TP_SIZE", "1"))
PY
)"
if [[ -n "$CKPT_TP" && "$CKPT_TP" != "$TP_SIZE" ]]; then
  echo "WARNING: requested TP_SIZE=$TP_SIZE but folded NVFP4 checkpoint config uses TP_SIZE=$CKPT_TP. Using checkpoint TP_SIZE."
  export TP_SIZE="$CKPT_TP"
fi

export FOLDED_NVFP4_PLUGIN_ENGINE="${FOLDED_NVFP4_PLUGIN_ENGINE:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-rms-scale-swiglu-plugin-engine-tp${TP_SIZE}}"
export LD_PRELOAD="$QWEN_RMS_SCALE_SWIGLU_PLUGIN_SO:${LD_PRELOAD:-}"
export TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION=1
export TRTLLM_QWEN_RMS_SCALE_SWIGLU_PLUGIN=1
export TRTLLM_QWEN_RMS_SCALE_SWIGLU_PLUGIN_MODE="${TRTLLM_QWEN_RMS_SCALE_SWIGLU_PLUGIN_MODE:-bf16_intermediate}"
export TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION_ALLOW_QUANTIZED=1
export CLEAN_ENGINE_DIR="${CLEAN_ENGINE_DIR:-1}"

cat <<EOM
Folded NVFP4 plugin engine build mode:
  TRTLLM_QWEN_RMS_SCALE_SWIGLU_PLUGIN_MODE=$TRTLLM_QWEN_RMS_SCALE_SWIGLU_PLUGIN_MODE
  Note: bf16_intermediate returns BF16/FP16 from the SwiGLU plugin; the existing NVFP4 proj linear quantizes internally.
EOM

CHECKPOINT_DIR="$FOLDED_NVFP4_PATH" \
ENGINE_DIR="$FOLDED_NVFP4_PLUGIN_ENGINE" \
BUILD_LOG="${BUILD_LOG:-runtime_logs/build_${MODEL_TAG}_folded_nvfp4_rms_scale_swiglu_plugin.log}" \
bash scripts/build_trtllm_engine_from_checkpoint.sh

echo "Folded NVFP4 plugin engine: $FOLDED_NVFP4_PLUGIN_ENGINE"
if [[ -f "$FOLDED_NVFP4_PLUGIN_ENGINE/rank0.engine" ]]; then
  echo "Checking serialized plugin marker in rank0.engine..."
  timeout "${STRINGS_TIMEOUT_S:-30s}" strings "$FOLDED_NVFP4_PLUGIN_ENGINE/rank0.engine" \
    | grep -i "QwenRmsScaleSwiglu" \
    | head || true
fi
