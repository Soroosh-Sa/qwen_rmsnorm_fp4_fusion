#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

export ORIG_NVFP4_PATH="${ORIG_NVFP4_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-NVFP4}"
export ORIG_ENGINE_DIR="${ORIG_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-NVFP4-engine-tp${TP_SIZE}}"
export FOLDED_ENGINE_DIR="${FOLDED_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-engine-tp${TP_SIZE}}"
export FOLDED_NVFP4_PLUGIN_ENGINE="${FOLDED_NVFP4_PLUGIN_ENGINE:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-rms-scale-swiglu-plugin-engine-tp${TP_SIZE}}"

export QUALITY_PROMPTS="${QUALITY_PROMPTS:-data/quality_prompts.jsonl}"
export QUALITY_MAX_NEW_TOKENS="${QUALITY_MAX_NEW_TOKENS:-128}"
export QUALITY_TIMEOUT_S="${QUALITY_TIMEOUT_S:-300}"
export OPENAI_API_MODE="${OPENAI_API_MODE:-completion}"
export COMPLETION_USE_CHAT_TEMPLATE="${COMPLETION_USE_CHAT_TEMPLATE:-1}"

export NORMAL_QUALITY_OUT="${NORMAL_QUALITY_OUT:-results/${MODEL_TAG}_normal_nvfp4_quality_outputs.jsonl}"
export FOLDED_BASE_QUALITY_OUT="${FOLDED_BASE_QUALITY_OUT:-results/${MODEL_TAG}_folded_nvfp4_base_quality_outputs.jsonl}"
export PLUGIN_QUALITY_OUT="${PLUGIN_QUALITY_OUT:-results/${MODEL_TAG}_folded_nvfp4_plugin_quality_outputs.jsonl}"
export QUALITY_SUMMARY_CSV="${QUALITY_SUMMARY_CSV:-results/${MODEL_TAG}_nvfp4_three_way_quality_summary.csv}"
export QUALITY_REPORT_MD="${QUALITY_REPORT_MD:-results/${MODEL_TAG}_nvfp4_three_way_quality_report.md}"

cat <<EOM
Stage-7 NVFP4 output quality/equivalence check
  A normal NVFP4 engine:        $ORIG_ENGINE_DIR
  B folded NVFP4 base engine:   $FOLDED_ENGINE_DIR
  C folded NVFP4 plugin engine: $FOLDED_NVFP4_PLUGIN_ENGINE

Prompts: $QUALITY_PROMPTS
Max new tokens: $QUALITY_MAX_NEW_TOKENS
API mode: $OPENAI_API_MODE

Outputs:
  A: $NORMAL_QUALITY_OUT
  B: $FOLDED_BASE_QUALITY_OUT
  C: $PLUGIN_QUALITY_OUT
  Summary CSV: $QUALITY_SUMMARY_CSV
  Report MD:   $QUALITY_REPORT_MD
EOM

for d in "$ORIG_ENGINE_DIR" "$FOLDED_ENGINE_DIR" "$FOLDED_NVFP4_PLUGIN_ENGINE"; do
  if [[ ! -d "$d" ]]; then
    echo "ERROR: required engine directory not found: $d" >&2
    echo "Build engines first with: bash scripts/run_nvfp4_normal_vs_folded_vs_plugin_benchmark.sh" >&2
    exit 3
  fi
done

PORT="$PORT" bash scripts/cleanup_trtllm_runtime.sh || true
trap 'PORT="$PORT" bash scripts/cleanup_trtllm_runtime.sh || true' EXIT INT TERM

TARGET=selected_original_nvfp4_engine \
ENGINE_DIR="$ORIG_ENGINE_DIR" \
QUALITY_LABEL=normal \
QUALITY_OUT="$NORMAL_QUALITY_OUT" \
SERVER_LOG="runtime_logs/normal_nvfp4_quality_server.log" \
bash scripts/run_one_engine_target_collect_outputs.sh

TARGET=selected_folded_nvfp4_engine \
ENGINE_DIR="$FOLDED_ENGINE_DIR" \
QUALITY_LABEL=folded_base \
QUALITY_OUT="$FOLDED_BASE_QUALITY_OUT" \
SERVER_LOG="runtime_logs/folded_nvfp4_base_quality_server.log" \
bash scripts/run_one_engine_target_collect_outputs.sh

TARGET=selected_folded_nvfp4_plugin_engine \
ENGINE_DIR="$FOLDED_NVFP4_PLUGIN_ENGINE" \
QUALITY_LABEL=plugin \
QUALITY_OUT="$PLUGIN_QUALITY_OUT" \
SERVER_LOG="runtime_logs/folded_nvfp4_plugin_quality_server.log" \
bash scripts/run_one_engine_target_collect_outputs.sh

python3 scripts/summarize_nvfp4_output_quality.py \
  --normal "$NORMAL_QUALITY_OUT" \
  --folded-base "$FOLDED_BASE_QUALITY_OUT" \
  --plugin "$PLUGIN_QUALITY_OUT" \
  --output-csv "$QUALITY_SUMMARY_CSV" \
  --output-md "$QUALITY_REPORT_MD"

echo ""
echo "Quality report:"
echo "$QUALITY_REPORT_MD"
echo ""
cat "$QUALITY_REPORT_MD"
