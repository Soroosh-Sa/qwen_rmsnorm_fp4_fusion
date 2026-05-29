#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

# Final Stage-6 benchmark:
#   A) normal/original NVFP4 TensorRT-LLM engine
#   B) folded-weight NVFP4 base TensorRT-LLM engine, no plugin
#   C) folded-weight NVFP4 plugin TensorRT-LLM engine
#
# Summary reports:
#   C vs A: final real comparison against standard NVFP4 TensorRT-LLM
#   C vs B: plugin effect after folding
#   B vs A: folding + requantization effect without plugin

export ORIG_NVFP4_PATH="${ORIG_NVFP4_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-NVFP4}"
export ORIG_ENGINE_DIR="${ORIG_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-NVFP4-engine-tp${TP_SIZE}}"
export FOLDED_ENGINE_DIR="${FOLDED_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-engine-tp${TP_SIZE}}"
export FOLDED_NVFP4_PLUGIN_ENGINE="${FOLDED_NVFP4_PLUGIN_ENGINE:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-rms-scale-swiglu-plugin-engine-tp${TP_SIZE}}"

export BUILD_ENGINES="${BUILD_ENGINES:-1}"
export BUILD_MISSING_CHECKPOINTS="${BUILD_MISSING_CHECKPOINTS:-1}"
export RUN_NORMAL_ENGINE="${RUN_NORMAL_ENGINE:-1}"
export RUN_FOLDED_BASE_ENGINE="${RUN_FOLDED_BASE_ENGINE:-1}"
export RUN_FOLDED_PLUGIN_ENGINE="${RUN_FOLDED_PLUGIN_ENGINE:-1}"

# Safer defaults for quick validation. Override these for full benchmarking.
export CONTEXTS="${CONTEXTS:-1024 2048}"
export CONCURRENCIES="${CONCURRENCIES:-1 2}"
export NUM_REQUESTS="${NUM_REQUESTS:-20}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
export OPENAI_API_MODE="${OPENAI_API_MODE:-completion}"
export COMPLETION_STREAM="${COMPLETION_STREAM:-0}"
export STOP_ON_CASE_FAILURE="${STOP_ON_CASE_FAILURE:-0}"

# If checkpoint config has a different TP, update paths before building/running.
RESOLVE_FROM="$FOLDED_NVFP4_PATH"
if [[ ! -f "$RESOLVE_FROM/config.json" ]]; then
  RESOLVE_FROM="$ORIG_NVFP4_PATH"
fi
if [[ -f "$RESOLVE_FROM/config.json" ]]; then
  CKPT_TP="$($PYTHON_BIN - <<PY
import json, os
p=os.path.join("$RESOLVE_FROM", "config.json")
try:
    c=json.load(open(p)); m=c.get("mapping",{}) or {}; print(m.get("tp_size") or m.get("attn_tp_size") or os.environ.get("TP_SIZE", "1"))
except Exception:
    print(os.environ.get("TP_SIZE", "1"))
PY
)"
  if [[ -n "$CKPT_TP" && "$CKPT_TP" != "$TP_SIZE" ]]; then
    echo "WARNING: requested TP_SIZE=$TP_SIZE but checkpoint uses TP_SIZE=$CKPT_TP. Using checkpoint TP_SIZE."
    export TP_SIZE="$CKPT_TP"
    export ORIG_ENGINE_DIR="${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-NVFP4-engine-tp${TP_SIZE}"
    export FOLDED_ENGINE_DIR="${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-engine-tp${TP_SIZE}"
    export FOLDED_NVFP4_PLUGIN_ENGINE="${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-rms-scale-swiglu-plugin-engine-tp${TP_SIZE}"
  fi
fi

export NORMAL_OUT="${NORMAL_OUT:-results/${MODEL_TAG}_normal_nvfp4_engine_benchmark.csv}"
export FOLDED_BASE_OUT="${FOLDED_BASE_OUT:-results/${MODEL_TAG}_folded_nvfp4_base_engine_benchmark.csv}"
export PLUGIN_OUT="${PLUGIN_OUT:-results/${MODEL_TAG}_folded_nvfp4_plugin_engine_benchmark.csv}"
export THREE_WAY_SUMMARY_OUT="${THREE_WAY_SUMMARY_OUT:-results/${MODEL_TAG}_nvfp4_normal_vs_folded_vs_plugin_summary.csv}"

cat <<EOM
Stage-6 final NVFP4 benchmark
  A normal NVFP4 engine:        $ORIG_ENGINE_DIR
  B folded NVFP4 base engine:   $FOLDED_ENGINE_DIR
  C folded NVFP4 plugin engine: $FOLDED_NVFP4_PLUGIN_ENGINE

Benchmark grid
  CONTEXTS=$CONTEXTS
  CONCURRENCIES=$CONCURRENCIES
  NUM_REQUESTS=$NUM_REQUESTS
  MAX_NEW_TOKENS=$MAX_NEW_TOKENS

Outputs
  A CSV: $NORMAL_OUT
  B CSV: $FOLDED_BASE_OUT
  C CSV: $PLUGIN_OUT
  Summary: $THREE_WAY_SUMMARY_OUT
EOM

if [[ "$BUILD_ENGINES" == "1" ]]; then
  BUILD_ORIGINAL_ENGINE="$RUN_NORMAL_ENGINE" \
  BUILD_FOLDED_BASE_ENGINE="$RUN_FOLDED_BASE_ENGINE" \
  BUILD_FOLDED_PLUGIN_ENGINE="$RUN_FOLDED_PLUGIN_ENGINE" \
  bash scripts/build_selected_nvfp4_three_engines.sh
fi

PORT="$PORT" bash scripts/cleanup_trtllm_runtime.sh || true
trap 'PORT="$PORT" bash scripts/cleanup_trtllm_runtime.sh || true' EXIT INT TERM

if [[ "$RUN_NORMAL_ENGINE" == "1" ]]; then
  export TARGET=selected_original_nvfp4_engine
  export ENGINE_DIR="$ORIG_ENGINE_DIR"
  export OUT="$NORMAL_OUT"
  bash scripts/run_one_engine_target_server_and_benchmark.sh
fi

if [[ "$RUN_FOLDED_BASE_ENGINE" == "1" ]]; then
  export TARGET=selected_folded_nvfp4_engine
  export ENGINE_DIR="$FOLDED_ENGINE_DIR"
  export OUT="$FOLDED_BASE_OUT"
  bash scripts/run_one_engine_target_server_and_benchmark.sh
fi

if [[ "$RUN_FOLDED_PLUGIN_ENGINE" == "1" ]]; then
  export TARGET=selected_folded_nvfp4_plugin_engine
  export ENGINE_DIR="$FOLDED_NVFP4_PLUGIN_ENGINE"
  export OUT="$PLUGIN_OUT"
  bash scripts/run_one_engine_target_server_and_benchmark.sh
fi

if [[ "$RUN_NORMAL_ENGINE" == "1" && "$RUN_FOLDED_BASE_ENGINE" == "1" && "$RUN_FOLDED_PLUGIN_ENGINE" == "1" ]]; then
  python3 scripts/summarize_nvfp4_three_way.py \
    --normal "$NORMAL_OUT" \
    --folded-base "$FOLDED_BASE_OUT" \
    --plugin "$PLUGIN_OUT" \
    --output "$THREE_WAY_SUMMARY_OUT"
  echo "Three-way summary written to: $THREE_WAY_SUMMARY_OUT"
  cat "$THREE_WAY_SUMMARY_OUT"
else
  echo "Skipping three-way summary because not all three targets were run."
fi
