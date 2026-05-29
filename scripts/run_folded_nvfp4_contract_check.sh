#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$FOLDED_NVFP4_PATH}"
mkdir -p runtime_logs

python src/check_folded_nvfp4_contract.py \
  --checkpoint "$CHECKPOINT_DIR" \
  --rank "${RANK_FILE:-rank0.safetensors}" \
  --layer "${LAYER_ID:-0}" \
  --check-post-ln \
  | tee "${OUT:-runtime_logs/folded_nvfp4_contract_check.txt}"
