#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection

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
  export TP_SIZE="$CKPT_TP"
fi

ORIG_ENGINE_DIR="${ORIG_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-NVFP4-engine-tp${TP_SIZE}}"
FOLDED_ENGINE_DIR="${FOLDED_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-engine-tp${TP_SIZE}}"

python3 src/validate_trtllm_engine_dir.py --engine-dir "$ORIG_ENGINE_DIR" --expected-tp "$TP_SIZE"
python3 src/validate_trtllm_engine_dir.py --engine-dir "$FOLDED_ENGINE_DIR" --expected-tp "$TP_SIZE"
