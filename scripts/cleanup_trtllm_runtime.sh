#!/bin/bash
set -euo pipefail
# Strong cleanup helper for repeated TensorRT-LLM experiments on Vast AI.
# It keeps Jupyter alive but kills trtllm-serve, TRT-LLM MPI workers, uvicorn
# children bound to PORT, and stale pid/log markers if requested.

PORT="${PORT:-8000}"
CLEAN_RUNTIME_LOGS="${CLEAN_RUNTIME_LOGS:-0}"
CLEAN_RESULTS_TMP="${CLEAN_RESULTS_TMP:-0}"

bash "$(dirname "$0")/stop_trtllm_server.sh" || true

# Extra patterns seen in TensorRT-LLM PyTorch backend worker failures.
for pattern in \
  "tensorrt_llm.executor.worker" \
  "tensorrt_llm.commands.serve" \
  "mpi4py.futures.server" \
  "trtllm-serve"; do
  PIDS="$(pgrep -f "$pattern" || true)"
  if [[ -n "$PIDS" ]]; then
    echo "Killing leftover process pattern: $pattern"
    echo "$PIDS"
    pkill -TERM -f "$pattern" || true
    sleep 3
    LEFT="$(pgrep -f "$pattern" || true)"
    if [[ -n "$LEFT" ]]; then
      pkill -9 -f "$pattern" || true
    fi
  fi
done

rm -f runtime_logs/*.pid server.pid 2>/dev/null || true

if [[ "$CLEAN_RUNTIME_LOGS" == "1" ]]; then
  rm -f runtime_logs/*.log 2>/dev/null || true
fi
if [[ "$CLEAN_RESULTS_TMP" == "1" ]]; then
  rm -rf results/tmp 2>/dev/null || true
fi

echo "Cleanup complete. Current GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
