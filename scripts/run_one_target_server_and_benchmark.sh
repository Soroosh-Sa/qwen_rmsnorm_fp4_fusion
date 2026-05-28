#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

TARGET="${TARGET:-prequant_nvfp4}"
SERVER_LOG="${SERVER_LOG:-runtime_logs/${TARGET}_server.log}"
PID_FILE="${PID_FILE:-runtime_logs/${TARGET}_server.pid}"
mkdir -p runtime_logs results

# Clean previous server on same port.
bash scripts/stop_trtllm_server.sh || true

TARGET="$TARGET" SERVER_LOG="$SERVER_LOG" bash scripts/serve_quantized_target.sh > "$SERVER_LOG" 2>&1 &
echo $! > "$PID_FILE"
echo "Started TARGET=$TARGET server PID=$(cat "$PID_FILE") log=$SERVER_LOG"

SERVER_LOG="$SERVER_LOG" PID_FILE="$PID_FILE" bash scripts/wait_for_server.sh

TARGET="$TARGET" SERVER_LOG="$SERVER_LOG" bash scripts/run_quantized_target_benchmark.sh

bash scripts/stop_trtllm_server.sh || true
