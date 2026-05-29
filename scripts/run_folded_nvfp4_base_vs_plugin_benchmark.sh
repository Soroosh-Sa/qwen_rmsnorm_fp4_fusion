#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

# Stage-6 default: compare folded-NVFP4 base engine vs folded-NVFP4 plugin engine.
# This is the clean comparison for the current project milestone.
export RUN_FOLDED_BASE_ENGINE="${RUN_FOLDED_BASE_ENGINE:-1}"
export RUN_FOLDED_PLUGIN_ENGINE="${RUN_FOLDED_PLUGIN_ENGINE:-1}"
export RUN_ORIGINAL_ENGINE="${RUN_ORIGINAL_ENGINE:-0}"
export BUILD_ENGINES="${BUILD_ENGINES:-1}"

# Safer defaults for quick validation; override for full benchmark.
export CONTEXTS="${CONTEXTS:-1024 2048}"
export CONCURRENCIES="${CONCURRENCIES:-1 2}"
export NUM_REQUESTS="${NUM_REQUESTS:-20}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
export OPENAI_API_MODE="${OPENAI_API_MODE:-completion}"
export COMPLETION_STREAM="${COMPLETION_STREAM:-0}"
export STOP_ON_CASE_FAILURE="${STOP_ON_CASE_FAILURE:-0}"

# Prefer TP size stored in folded checkpoint config if present.
if [[ -f "$FOLDED_NVFP4_PATH/config.json" ]]; then
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
    echo "WARNING: requested TP_SIZE=$TP_SIZE but folded NVFP4 checkpoint uses TP_SIZE=$CKPT_TP. Using checkpoint TP_SIZE."
    export TP_SIZE="$CKPT_TP"
  fi
fi

export FOLDED_ENGINE_DIR="${FOLDED_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-engine-tp${TP_SIZE}}"
export FOLDED_NVFP4_PLUGIN_ENGINE="${FOLDED_NVFP4_PLUGIN_ENGINE:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-rms-scale-swiglu-plugin-engine-tp${TP_SIZE}}"
export BASE_OUT="${BASE_OUT:-results/${MODEL_TAG}_folded_nvfp4_base_engine_benchmark.csv}"
export PLUGIN_OUT="${PLUGIN_OUT:-results/${MODEL_TAG}_folded_nvfp4_plugin_engine_benchmark.csv}"
export SUMMARY_OUT="${SUMMARY_OUT:-results/${MODEL_TAG}_folded_nvfp4_base_vs_plugin_summary.csv}"

if [[ "$BUILD_ENGINES" == "1" ]]; then
  bash scripts/build_selected_folded_nvfp4_base_and_plugin_engines.sh
fi

PORT="$PORT" bash scripts/cleanup_trtllm_runtime.sh || true
trap 'PORT="$PORT" bash scripts/cleanup_trtllm_runtime.sh || true' EXIT INT TERM

if [[ "$RUN_FOLDED_BASE_ENGINE" == "1" ]]; then
  export TARGET=selected_folded_nvfp4_engine
  export ENGINE_DIR="$FOLDED_ENGINE_DIR"
  export OUT="$BASE_OUT"
  bash scripts/run_one_engine_target_server_and_benchmark.sh
fi

if [[ "$RUN_FOLDED_PLUGIN_ENGINE" == "1" ]]; then
  export TARGET=selected_folded_nvfp4_plugin_engine
  export ENGINE_DIR="$FOLDED_NVFP4_PLUGIN_ENGINE"
  export OUT="$PLUGIN_OUT"
  bash scripts/run_one_engine_target_server_and_benchmark.sh
fi

if [[ "$RUN_FOLDED_BASE_ENGINE" == "1" && "$RUN_FOLDED_PLUGIN_ENGINE" == "1" ]]; then
  python3 scripts/summarize_folded_nvfp4_base_vs_plugin.py \
    --base "$BASE_OUT" \
    --plugin "$PLUGIN_OUT" \
    --output "$SUMMARY_OUT"
  echo "Summary written to: $SUMMARY_OUT"
  cat "$SUMMARY_OUT"
fi
