#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

TARGET="${TARGET:-selected_original_nvfp4_engine}"
SERVER_LOG="${SERVER_LOG:-runtime_logs/${TARGET}_server.log}"
PID_FILE="${PID_FILE:-runtime_logs/${TARGET}_server.pid}"
mkdir -p runtime_logs results

cleanup() {
  echo "Running TensorRT-LLM cleanup for TARGET=$TARGET ..."
  PORT="$PORT" bash scripts/cleanup_trtllm_runtime.sh || true
}
trap cleanup EXIT INT TERM

cleanup
TARGET="$TARGET" SERVER_LOG="$SERVER_LOG" bash scripts/serve_engine_target.sh > "$SERVER_LOG" 2>&1 &
echo $! > "$PID_FILE"
echo "Started TARGET=$TARGET server PID=$(cat "$PID_FILE") log=$SERVER_LOG"

SERVER_LOG="$SERVER_LOG" PID_FILE="$PID_FILE" bash scripts/wait_for_server.sh

# Reuse the same OpenAI-compatible benchmark client, but label the output by engine target.
TARGET="$TARGET" SERVER_LOG="$SERVER_LOG" bash scripts/run_quantized_target_benchmark.sh
