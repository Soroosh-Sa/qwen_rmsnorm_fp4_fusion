#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

# Resolve a globally consistent TP_SIZE for the selected model.
# This avoids the common bug where quantization auto-adjusts TP_SIZE inside one
# process, but later engine build/benchmark scripts still use the old requested value.
MODEL_TO_CHECK="${MODEL_TO_CHECK:-${ORIGINAL_MODEL_PATH}}"
AUTO_ADJUST_TP_SIZE="${AUTO_ADJUST_TP_SIZE:-1}"
MAX_GPUS_FOR_TP="${MAX_GPUS_FOR_TP:-${GPU_COUNT:-16}}"

TP_CHECK_OUT="$(python src/check_tp_compatibility.py \
  --model "$MODEL_TO_CHECK" \
  --tp-size "${TP_SIZE:-1}" \
  --max-gpus "$MAX_GPUS_FOR_TP" \
  --print-shell 2>/dev/null || true)"
eval "$TP_CHECK_OUT"

if [[ "${TP_COMPATIBLE:-1}" != "1" ]]; then
  echo "WARNING: Requested TP_SIZE=${TP_SIZE:-unset} is not compatible with MODEL_TO_CHECK=$MODEL_TO_CHECK" >&2
  echo "  num_attention_heads=${MODEL_NUM_ATTENTION_HEADS:-unknown}" >&2
  echo "  num_key_value_heads=${MODEL_NUM_KEY_VALUE_HEADS:-unknown}" >&2
  echo "  valid TP sizes: ${VALID_TP_SIZES:-unknown}" >&2
  if [[ "$AUTO_ADJUST_TP_SIZE" == "1" && -n "${SUGGESTED_TP_SIZE:-}" ]]; then
    echo "Auto-adjusting global TP_SIZE: ${TP_SIZE:-unset} -> $SUGGESTED_TP_SIZE" >&2
    export TP_SIZE="$SUGGESTED_TP_SIZE"
  else
    echo "ERROR: Set TP_SIZE to one of: ${VALID_TP_SIZES:-unknown}, or set AUTO_ADJUST_TP_SIZE=1." >&2
    exit 3
  fi
fi

echo "export TP_SIZE='$TP_SIZE'"
echo "export VALID_TP_SIZES='${VALID_TP_SIZES:-}'"
echo "export SUGGESTED_TP_SIZE='${SUGGESTED_TP_SIZE:-}'"
