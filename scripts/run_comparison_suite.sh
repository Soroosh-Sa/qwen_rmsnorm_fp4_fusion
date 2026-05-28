#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

TARGETS="${TARGETS:-prequant_nvfp4 folded_nvfp4 folded_gptq_int4}"
mkdir -p results runtime_logs

for T in $TARGETS; do
  echo "============================================================"
  echo "Running comparison target: $T"
  echo "============================================================"
  TARGET="$T" \
  OUT="results/${T}_benchmark.csv" \
  PLAN_OUT="results/${T}_safe_plan.json" \
  SERVER_LOG="runtime_logs/${T}_server.log" \
  PID_FILE="runtime_logs/${T}_server.pid" \
    bash scripts/run_one_target_server_and_benchmark.sh || {
      echo "WARNING: target $T failed; continuing to next target."
    }
done

python3 src/summarize_quantized_comparison.py \
  --inputs results/prequant_nvfp4_benchmark.csv results/folded_nvfp4_benchmark.csv results/folded_gptq_int4_benchmark.csv \
  --output results/quantized_comparison_summary.csv \
  --pivot-output results/quantized_comparison_pivot.csv || true
