#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection

# Benchmarks original-NVFP4 and folded-NVFP4 of the same selected model.
# Assumes both outputs already exist.

ORIG_NVFP4_PATH="${ORIG_NVFP4_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-NVFP4}"
if [[ ! -d "$ORIG_NVFP4_PATH" ]]; then
  echo "Original NVFP4 path not found: $ORIG_NVFP4_PATH" >&2
  exit 2
fi
if [[ ! -d "$FOLDED_NVFP4_PATH" ]]; then
  echo "Folded NVFP4 path not found: $FOLDED_NVFP4_PATH" >&2
  exit 2
fi

# Always start from a clean runtime and clean again at the end.
PORT="$PORT" bash scripts/cleanup_trtllm_runtime.sh || true
trap 'PORT="$PORT" bash scripts/cleanup_trtllm_runtime.sh || true' EXIT INT TERM

export MODEL_PATH="$ORIG_NVFP4_PATH"
export TARGET=selected_original_nvfp4
bash "$(dirname "$0")/run_one_target_server_and_benchmark.sh"

export MODEL_PATH="$FOLDED_NVFP4_PATH"
export TARGET=selected_folded_nvfp4
bash "$(dirname "$0")/run_one_target_server_and_benchmark.sh"
