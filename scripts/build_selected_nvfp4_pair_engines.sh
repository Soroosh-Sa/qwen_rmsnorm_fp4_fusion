#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection

ORIG_NVFP4_PATH="${ORIG_NVFP4_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-NVFP4}"
ORIG_ENGINE_DIR="${ORIG_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-NVFP4-engine-tp${TP_SIZE}}"
FOLDED_ENGINE_DIR="${FOLDED_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-engine-tp${TP_SIZE}}"

# Prefer the TP embedded in the quantized checkpoint. It is the source of truth
# for rank*.safetensors and avoids naming/build mismatches.
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
  echo "WARNING: requested TP_SIZE=$TP_SIZE but quantized checkpoint config uses TP_SIZE=$CKPT_TP. Using checkpoint TP_SIZE."
  export TP_SIZE="$CKPT_TP"
fi

# Recompute engine directory names after possible TP adjustment unless user explicitly supplied them.
ORIG_ENGINE_DIR="${ORIG_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-NVFP4-engine-tp${TP_SIZE}}"
FOLDED_ENGINE_DIR="${FOLDED_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-engine-tp${TP_SIZE}}"

if [[ ! -d "$ORIG_NVFP4_PATH" ]]; then
  echo "ERROR: Original NVFP4 checkpoint not found: $ORIG_NVFP4_PATH" >&2
  exit 2
fi
if [[ ! -d "$FOLDED_NVFP4_PATH" ]]; then
  echo "ERROR: Folded NVFP4 checkpoint not found: $FOLDED_NVFP4_PATH" >&2
  exit 2
fi

export CLEAN_ENGINE_DIR="${CLEAN_ENGINE_DIR:-1}"

CHECKPOINT_DIR="$ORIG_NVFP4_PATH" ENGINE_DIR="$ORIG_ENGINE_DIR" bash scripts/build_trtllm_engine_from_checkpoint.sh
CHECKPOINT_DIR="$FOLDED_NVFP4_PATH" ENGINE_DIR="$FOLDED_ENGINE_DIR" bash scripts/build_trtllm_engine_from_checkpoint.sh

echo "Built pair engines:"
echo "  ORIGINAL_ENGINE_DIR=$ORIG_ENGINE_DIR"
echo "  FOLDED_ENGINE_DIR=$FOLDED_ENGINE_DIR"
