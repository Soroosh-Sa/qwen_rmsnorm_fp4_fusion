#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export MODEL_PROFILE="${MODEL_PROFILE:-qwen25_05b}"
export TP_SIZE="${TP_SIZE:-2}"
export MODEL_ROOT="${MODEL_ROOT:-/workspace/models}"

# Avoid sourcing common_env if the caller has explicit paths. Source only if it works quickly.
if timeout 15s bash -lc 'source scripts/common_env.sh >/dev/null 2>&1'; then
  # shellcheck source=/dev/null
  source scripts/common_env.sh
else
  echo "WARNING: scripts/common_env.sh did not finish quickly; using explicit fallback names."
  MODEL_TAG="${MODEL_TAG:-Qwen-Qwen2.5-0.5B-Instruct}"
  FOLDED_NVFP4_PATH="${FOLDED_NVFP4_PATH:-$MODEL_ROOT/${MODEL_TAG}-FOLDED-NVFP4}"
fi

bash scripts/collect_nvfp4_references.sh

if [[ -d "${FOLDED_NVFP4_PATH:-}" ]]; then
  python src/inspect_nvfp4_checkpoint.py \
    --checkpoint "$FOLDED_NVFP4_PATH" \
    --max-keys "${MAX_KEYS:-400}" \
    --output runtime_logs/folded_nvfp4_checkpoint_inspection.txt
else
  echo "Folded NVFP4 checkpoint not found yet: ${FOLDED_NVFP4_PATH:-unset}"
  echo "After quantization, rerun this script to inspect checkpoint tensors."
fi
