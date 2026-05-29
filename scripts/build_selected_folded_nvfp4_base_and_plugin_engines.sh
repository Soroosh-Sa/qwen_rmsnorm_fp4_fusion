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

export FOLDED_ENGINE_DIR="${FOLDED_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-engine-tp${TP_SIZE}}"
export FOLDED_NVFP4_PLUGIN_ENGINE="${FOLDED_NVFP4_PLUGIN_ENGINE:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-rms-scale-swiglu-plugin-engine-tp${TP_SIZE}}"
export CLEAN_ENGINE_DIR="${CLEAN_ENGINE_DIR:-1}"

cat <<EOM
Building folded-NVFP4 base and plugin engines
  FOLDED_NVFP4_PATH=$FOLDED_NVFP4_PATH
  FOLDED_ENGINE_DIR=$FOLDED_ENGINE_DIR
  FOLDED_NVFP4_PLUGIN_ENGINE=$FOLDED_NVFP4_PLUGIN_ENGINE
  TP_SIZE=$TP_SIZE
EOM

# 1) Baseline folded NVFP4 engine: no custom fusion/plugin env vars.
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

# 2) Plugin engine: build plugin .so, apply TensorRT-LLM patch, then build with plugin env vars.
bash scripts/build_selected_folded_nvfp4_plugin_engine.sh

cat <<EOM
Built folded-NVFP4 benchmark engines:
  BASE_ENGINE=$FOLDED_ENGINE_DIR
  PLUGIN_ENGINE=$FOLDED_NVFP4_PLUGIN_ENGINE
EOM
