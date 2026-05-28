#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection

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
