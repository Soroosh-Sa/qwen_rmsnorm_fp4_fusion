#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

TARGET="${TARGET:?TARGET must be set: selected_original_nvfp4_engine, selected_folded_nvfp4_engine, selected_folded_nvfp4_plugin_engine}"
QUALITY_LABEL="${QUALITY_LABEL:?QUALITY_LABEL must be set: normal, folded_base, plugin}"
QUALITY_OUT="${QUALITY_OUT:?QUALITY_OUT must be set}"
QUALITY_PROMPTS="${QUALITY_PROMPTS:-data/quality_prompts.jsonl}"
QUALITY_MAX_NEW_TOKENS="${QUALITY_MAX_NEW_TOKENS:-128}"
QUALITY_TIMEOUT_S="${QUALITY_TIMEOUT_S:-300}"
OPENAI_API_MODE="${OPENAI_API_MODE:-completion}"
SERVER_LOG="${SERVER_LOG:-runtime_logs/${TARGET}_quality_server.log}"
PID_FILE="${PID_FILE:-runtime_logs/${TARGET}_quality_server.pid}"
mkdir -p runtime_logs results

cleanup() {
  echo "Running TensorRT-LLM cleanup for TARGET=$TARGET ..."
  PORT="$PORT" bash scripts/cleanup_trtllm_runtime.sh || true
}
trap cleanup EXIT INT TERM

cleanup
TARGET="$TARGET" SERVER_LOG="$SERVER_LOG" bash scripts/serve_engine_target.sh > "$SERVER_LOG" 2>&1 &
echo $! > "$PID_FILE"
echo "Started TARGET=$TARGET quality server PID=$(cat "$PID_FILE") log=$SERVER_LOG"

SERVER_LOG="$SERVER_LOG" PID_FILE="$PID_FILE" bash scripts/wait_for_server.sh

MODEL_ID=$(curl -s "http://localhost:${PORT}/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')

case "$TARGET" in
  selected_original_nvfp4_engine)
    TOKENIZER_PATH="${TOKENIZER_PATH:-${ORIG_NVFP4_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-NVFP4}}"
    ;;
  selected_folded_nvfp4_engine|selected_folded_nvfp4_plugin_engine)
    TOKENIZER_PATH="${TOKENIZER_PATH:-${FOLDED_NVFP4_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-FOLDED-NVFP4}}"
    ;;
  *)
    TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}}"
    ;;
esac

python3 benchmark/collect_openai_outputs.py \
  --host localhost \
  --port "$PORT" \
  --model "$MODEL_ID" \
  --target "$QUALITY_LABEL" \
  --prompts "$QUALITY_PROMPTS" \
  --output "$QUALITY_OUT" \
  --api-mode "$OPENAI_API_MODE" \
  --max-tokens "$QUALITY_MAX_NEW_TOKENS" \
  --timeout-s "$QUALITY_TIMEOUT_S" \
  --tokenizer-path "$TOKENIZER_PATH"

echo "Quality outputs written to: $QUALITY_OUT"
