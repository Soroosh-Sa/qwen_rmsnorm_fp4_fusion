#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection

ORIG_ENGINE_DIR="${ORIG_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-NVFP4-engine-tp${TP_SIZE}}"
FOLDED_ENGINE_DIR="${FOLDED_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-engine-tp${TP_SIZE}}"

python3 src/validate_trtllm_engine_dir.py --engine-dir "$ORIG_ENGINE_DIR" --expected-tp "$TP_SIZE"
python3 src/validate_trtllm_engine_dir.py --engine-dir "$FOLDED_ENGINE_DIR" --expected-tp "$TP_SIZE"
