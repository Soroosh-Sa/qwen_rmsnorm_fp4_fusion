#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

PORT="${PORT:-8000}"
SERVER_LOG="${SERVER_LOG:-${1:-}}"
TAIL_LINES="${TAIL_LINES:-120}"
OUT_DIR="${OUT_DIR:-results/diagnostics}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_FILE="${OUT_DIR}/diagnose_${STAMP}.txt"

mkdir -p "$OUT_DIR"

{
  echo "============================================================"
  echo "TensorRT-LLM diagnostic report"
  echo "Timestamp: $(date)"
  echo "Host: $(hostname || true)"
  echo "Port: ${PORT}"
  echo "Server log: ${SERVER_LOG:-<not provided>}"
  echo "============================================================"
  echo

  echo "--- nvidia-smi ---"
  nvidia-smi || true
  echo

  echo "--- Relevant processes ---"
  ps aux | grep -E "trtllm|uvicorn|python" | grep -v grep || true
  echo

  echo "--- Port bind check ---"
  python3 - <<PY || true
import socket
port = int("${PORT}")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("0.0.0.0", port))
    print(f"Port {port} is free / bindable")
except OSError as e:
    print(f"Port {port} is in use / not bindable: {e}")
finally:
    s.close()
PY
  echo

  echo "--- Health check ---"
  curl --max-time 5 --connect-timeout 2 -i "http://localhost:${PORT}/health" || true
  echo

  echo "--- Models endpoint ---"
  curl --max-time 5 --connect-timeout 2 -sS "http://localhost:${PORT}/v1/models" || true
  echo

  echo "--- Metrics endpoint, first 4000 chars ---"
  curl --max-time 10 --connect-timeout 2 -sS "http://localhost:${PORT}/metrics" | head -c 4000 || true
  echo
  echo

  if [[ -n "${SERVER_LOG:-}" && -f "$SERVER_LOG" ]]; then
    echo "--- Last ${TAIL_LINES} lines of server log: ${SERVER_LOG} ---"
    tail -n "$TAIL_LINES" "$SERVER_LOG" || true
  else
    echo "--- Server log not available ---"
  fi

  echo
  echo "============================================================"
  echo "End diagnostic report"
  echo "============================================================"
} | tee "$OUT_FILE"

echo "Diagnostic saved to: $OUT_FILE"
