#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/common_env.sh"

TIMEOUT_S="${TIMEOUT_S:-1800}"
SLEEP_S="${SLEEP_S:-10}"
# Detect a server-start hang where the process is alive but the log stops changing
# and /health never becomes ready. This happens on some long-context TensorRT-LLM
# configurations during executor/KV-cache estimation. Set to 0 to disable.
STARTUP_NO_PROGRESS_TIMEOUT_S="${STARTUP_NO_PROGRESS_TIMEOUT_S:-600}"
STARTUP_NO_PROGRESS_MIN_ELAPSED_S="${STARTUP_NO_PROGRESS_MIN_ELAPSED_S:-180}"
SERVER_LOG="${SERVER_LOG:-}"
PID_FILE="${PID_FILE:-server.pid}"
START_TIME=$(date +%s)

HEALTH_URL="http://localhost:${PORT}/health"
MODELS_URL="http://localhost:${PORT}/v1/models"

echo "Waiting for TensorRT-LLM server at ${HEALTH_URL}"
echo "Timeout: ${TIMEOUT_S}s"
if [[ -n "$SERVER_LOG" ]]; then
  echo "Server log: $SERVER_LOG"
fi
echo "No-progress startup timeout: ${STARTUP_NO_PROGRESS_TIMEOUT_S}s after min elapsed ${STARTUP_NO_PROGRESS_MIN_ELAPSED_S}s"

LAST_LOG_SIZE=0
LAST_LOG_CHANGE_TIME="$START_TIME"
if [[ -n "$SERVER_LOG" && -f "$SERVER_LOG" ]]; then
  LAST_LOG_SIZE=$(wc -c < "$SERVER_LOG" 2>/dev/null || echo 0)
fi

while true; do
  # If a PID file exists and the process already exited, fail early with diagnostics.
  if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE" || true)"
    if [[ -n "${PID:-}" ]] && ! kill -0 "$PID" 2>/dev/null; then
      echo "ERROR: Server process PID ${PID} exited before health check became ready."
      SERVER_LOG="$SERVER_LOG" bash scripts/diagnose_server.sh || true
      echo "Running full TensorRT-LLM cleanup after early server exit..."
      bash scripts/stop_trtllm_server.sh || true
      exit 1
    fi
  fi

  HTTP_CODE=$(curl --max-time 5 --connect-timeout 2 -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" || true)
  if [[ "$HTTP_CODE" == "200" ]]; then
    echo "Server health check passed."
    echo "Models:"
    curl --max-time 10 --connect-timeout 2 -sS "$MODELS_URL" || true
    echo
    exit 0
  fi

  NOW=$(date +%s)
  ELAPSED=$((NOW - START_TIME))

  # Update log-progress tracker. If the server process is alive but the log has
  # not changed for a long time and health still returns 000/non-200, classify
  # as a startup hang instead of waiting until the full stage timeout.
  if [[ -n "$SERVER_LOG" && -f "$SERVER_LOG" ]]; then
    CURRENT_LOG_SIZE=$(wc -c < "$SERVER_LOG" 2>/dev/null || echo 0)
    if [[ "$CURRENT_LOG_SIZE" != "$LAST_LOG_SIZE" ]]; then
      LAST_LOG_SIZE="$CURRENT_LOG_SIZE"
      LAST_LOG_CHANGE_TIME="$NOW"
    fi
    NO_PROGRESS_FOR=$((NOW - LAST_LOG_CHANGE_TIME))
    if (( STARTUP_NO_PROGRESS_TIMEOUT_S > 0 && ELAPSED >= STARTUP_NO_PROGRESS_MIN_ELAPSED_S && NO_PROGRESS_FOR >= STARTUP_NO_PROGRESS_TIMEOUT_S )); then
      echo "ERROR: Server startup appears hung: no new log output for ${NO_PROGRESS_FOR}s, health code ${HTTP_CODE}."
      echo "SERVER_START_HANG_NO_LOG_PROGRESS no_progress_for=${NO_PROGRESS_FOR}s elapsed=${ELAPSED}s" >> "$SERVER_LOG" || true
      if [[ -f "$PID_FILE" ]]; then
        PID="$(cat "$PID_FILE" || true)"
        if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
          echo "Killing hung server PID ${PID}."
          kill "$PID" 2>/dev/null || true
          sleep 5
          kill -9 "$PID" 2>/dev/null || true
        fi
      fi
      SERVER_LOG="$SERVER_LOG" bash scripts/diagnose_server.sh || true
      echo "Running full TensorRT-LLM cleanup after startup hang..."
      bash scripts/stop_trtllm_server.sh || true
      exit 124
    fi
  fi

  if (( ELAPSED >= TIMEOUT_S )); then
    echo "ERROR: Server did not become healthy within ${TIMEOUT_S}s. Last HTTP code: ${HTTP_CODE}"
    SERVER_LOG="$SERVER_LOG" bash scripts/diagnose_server.sh || true
    exit 1
  fi

  echo "Server not ready yet after ${ELAPSED}s. Last HTTP code: ${HTTP_CODE}. Sleeping ${SLEEP_S}s..."
  if [[ -n "$SERVER_LOG" && -f "$SERVER_LOG" ]]; then
    echo "Last 8 log lines:"
    tail -n 8 "$SERVER_LOG" || true
  fi
  sleep "$SLEEP_S"
done
