#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

# Builds the three engines needed for the final Stage-6 comparison:
#   A) normal/original NVFP4 TensorRT-LLM engine
#   B) folded-weight NVFP4 TensorRT-LLM base engine, no custom plugin
#   C) folded-weight NVFP4 TensorRT-LLM engine with QwenRmsScaleSwiglu plugin
#
# The final paper/result comparison is C vs A.  B is kept as an ablation:
#   C vs B isolates the plugin/graph change after folding.
#   B vs A isolates the effect of folding + requantization without the plugin.

export ORIG_NVFP4_PATH="${ORIG_NVFP4_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-NVFP4}"
export ORIG_ENGINE_DIR="${ORIG_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-NVFP4-engine-tp${TP_SIZE}}"
export FOLDED_ENGINE_DIR="${FOLDED_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-engine-tp${TP_SIZE}}"
export FOLDED_NVFP4_PLUGIN_ENGINE="${FOLDED_NVFP4_PLUGIN_ENGINE:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-rms-scale-swiglu-plugin-engine-tp${TP_SIZE}}"

BUILD_MISSING_CHECKPOINTS="${BUILD_MISSING_CHECKPOINTS:-1}"
BUILD_ORIGINAL_ENGINE="${BUILD_ORIGINAL_ENGINE:-1}"
BUILD_FOLDED_BASE_ENGINE="${BUILD_FOLDED_BASE_ENGINE:-1}"
BUILD_FOLDED_PLUGIN_ENGINE="${BUILD_FOLDED_PLUGIN_ENGINE:-1}"
export CLEAN_ENGINE_DIR="${CLEAN_ENGINE_DIR:-1}"

if [[ ! -d "$ORIG_NVFP4_PATH" ]]; then
  if [[ "$BUILD_MISSING_CHECKPOINTS" == "1" ]]; then
    echo "Original NVFP4 checkpoint missing; creating it from the original model:"
    echo "  ORIG_NVFP4_PATH=$ORIG_NVFP4_PATH"
    OUTPUT_DIR="$ORIG_NVFP4_PATH" bash scripts/quantize_selected_original_nvfp4.sh
  else
    echo "ERROR: Original NVFP4 checkpoint not found: $ORIG_NVFP4_PATH" >&2
    echo "Create it with: MODEL_PROFILE=$MODEL_PROFILE TP_SIZE=$TP_SIZE bash scripts/quantize_selected_original_nvfp4.sh" >&2
    exit 2
  fi
fi

if [[ ! -d "$FOLDED_NVFP4_PATH" ]]; then
  if [[ "$BUILD_MISSING_CHECKPOINTS" == "1" ]]; then
    echo "Folded NVFP4 checkpoint missing; creating it from folded BF16 weights:"
    echo "  FOLDED_NVFP4_PATH=$FOLDED_NVFP4_PATH"
    bash scripts/quantize_selected_folded_nvfp4.sh
  else
    echo "ERROR: Folded NVFP4 checkpoint not found: $FOLDED_NVFP4_PATH" >&2
    echo "Create it with: MODEL_PROFILE=$MODEL_PROFILE TP_SIZE=$TP_SIZE bash scripts/quantize_selected_folded_nvfp4.sh" >&2
    exit 2
  fi
fi

# Prefer TP size stored in the folded checkpoint. Both original and folded
# checkpoints should normally use the same TP after selected-model quantization.
CKPT_TP="$($PYTHON_BIN - <<PY
import json, os
for p in [os.path.join("$FOLDED_NVFP4_PATH", "config.json"), os.path.join("$ORIG_NVFP4_PATH", "config.json")]:
    try:
        c=json.load(open(p)); m=c.get("mapping",{}) or {}
        v=m.get("tp_size") or m.get("attn_tp_size")
        if v:
            print(v); raise SystemExit
    except SystemExit:
        raise
    except Exception:
        pass
print(os.environ.get("TP_SIZE", "1"))
PY
)"
if [[ -n "$CKPT_TP" && "$CKPT_TP" != "$TP_SIZE" ]]; then
  echo "WARNING: requested TP_SIZE=$TP_SIZE but checkpoint config uses TP_SIZE=$CKPT_TP. Using checkpoint TP_SIZE."
  export TP_SIZE="$CKPT_TP"
  export ORIG_ENGINE_DIR="${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-NVFP4-engine-tp${TP_SIZE}"
  export FOLDED_ENGINE_DIR="${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-engine-tp${TP_SIZE}"
  export FOLDED_NVFP4_PLUGIN_ENGINE="${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-rms-scale-swiglu-plugin-engine-tp${TP_SIZE}"
fi

cat <<EOM
Building Stage-6 three-way NVFP4 engines
  A normal NVFP4 checkpoint:       $ORIG_NVFP4_PATH
  A normal NVFP4 engine:           $ORIG_ENGINE_DIR
  B folded NVFP4 checkpoint:       $FOLDED_NVFP4_PATH
  B folded NVFP4 base engine:      $FOLDED_ENGINE_DIR
  C folded NVFP4 plugin engine:    $FOLDED_NVFP4_PLUGIN_ENGINE
  TP_SIZE:                         $TP_SIZE
EOM

if [[ "$BUILD_ORIGINAL_ENGINE" == "1" ]]; then
  (
    unset TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION
    unset TRTLLM_QWEN_RMS_SCALE_SWIGLU_PLUGIN
    unset TRTLLM_QWEN_RMS_SCALE_SWIGLU_PLUGIN_MODE
    unset TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION_ALLOW_QUANTIZED
    CHECKPOINT_DIR="$ORIG_NVFP4_PATH" \
    ENGINE_DIR="$ORIG_ENGINE_DIR" \
    BUILD_LOG="runtime_logs/build_${MODEL_TAG}_original_nvfp4_engine.log" \
    bash scripts/build_trtllm_engine_from_checkpoint.sh
  )
fi

if [[ "$BUILD_FOLDED_BASE_ENGINE" == "1" ]]; then
  (
    unset TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION
    unset TRTLLM_QWEN_RMS_SCALE_SWIGLU_PLUGIN
    unset TRTLLM_QWEN_RMS_SCALE_SWIGLU_PLUGIN_MODE
    unset TRTLLM_QWEN_FOLDED_RMSNORM_MLP_FUSION_ALLOW_QUANTIZED
    CHECKPOINT_DIR="$FOLDED_NVFP4_PATH" \
    ENGINE_DIR="$FOLDED_ENGINE_DIR" \
    BUILD_LOG="runtime_logs/build_${MODEL_TAG}_folded_nvfp4_base_engine.log" \
    bash scripts/build_trtllm_engine_from_checkpoint.sh
  )
fi

if [[ "$BUILD_FOLDED_PLUGIN_ENGINE" == "1" ]]; then
  bash scripts/build_selected_folded_nvfp4_plugin_engine.sh
fi

cat <<EOM
Built/validated Stage-6 engines:
  A NORMAL_ENGINE_DIR=$ORIG_ENGINE_DIR
  B FOLDED_BASE_ENGINE_DIR=$FOLDED_ENGINE_DIR
  C FOLDED_PLUGIN_ENGINE_DIR=$FOLDED_NVFP4_PLUGIN_ENGINE
EOM
