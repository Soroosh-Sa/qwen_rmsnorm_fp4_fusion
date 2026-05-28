#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection

# Resolve TP_SIZE from quantized checkpoint config when available. This is safer
# than relying on a stale requested TP_SIZE.
ORIG_NVFP4_PATH="${ORIG_NVFP4_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-NVFP4}"
if [[ -f "$ORIG_NVFP4_PATH/config.json" ]]; then
  CKPT_TP="$($PYTHON_BIN - <<PY
import json, os
p=os.path.join("$ORIG_NVFP4_PATH", "config.json")
try:
    c=json.load(open(p)); m=c.get("mapping",{}) or {}; print(m.get("tp_size") or m.get("attn_tp_size") or os.environ.get("TP_SIZE", "1"))
except Exception:
    print(os.environ.get("TP_SIZE", "1"))
PY
)"
  if [[ -n "$CKPT_TP" && "$CKPT_TP" != "$TP_SIZE" ]]; then
    echo "WARNING: requested TP_SIZE=$TP_SIZE but NVFP4 checkpoint uses TP_SIZE=$CKPT_TP. Using checkpoint TP_SIZE."
    export TP_SIZE="$CKPT_TP"
  fi
else
  eval "$(MODEL_TO_CHECK="$ORIGINAL_MODEL_PATH" bash scripts/resolve_selected_tp_size.sh | tail -n 3)"
fi

export ORIG_ENGINE_DIR="${ORIG_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-NVFP4-engine-tp${TP_SIZE}}"
export FOLDED_ENGINE_DIR="${FOLDED_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-engine-tp${TP_SIZE}}"

if [[ "${BUILD_ENGINES:-1}" == "1" ]]; then
  bash scripts/build_selected_nvfp4_pair_engines.sh
fi

PORT="$PORT" bash scripts/cleanup_trtllm_runtime.sh || true
trap 'PORT="$PORT" bash scripts/cleanup_trtllm_runtime.sh || true' EXIT INT TERM

export TARGET=selected_original_nvfp4_engine
export ENGINE_DIR="$ORIG_ENGINE_DIR"
bash scripts/run_one_engine_target_server_and_benchmark.sh

export TARGET=selected_folded_nvfp4_engine
export ENGINE_DIR="$FOLDED_ENGINE_DIR"
bash scripts/run_one_engine_target_server_and_benchmark.sh
