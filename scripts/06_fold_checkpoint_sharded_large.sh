#!/usr/bin/env bash
set -euo pipefail

# Large-model offline folding path for Qwen 460B/480B-style checkpoints.
# This does NOT load the full model and does NOT require GPUs.
# It processes safetensors shards one by one on CPU.
#
# Required:
#   export INPUT_DIR=/path/to/original/qwen-460b-or-480b-hf-checkpoint
#   export OUTPUT_DIR=/path/to/folded/qwen-460b-or-480b-hf-checkpoint
# Optional:
#   export METRICS_DIR=metrics/qwen_460b_fold
#   export MAX_SHARDS=2      # debug only; omit for full run

if [[ -z "${INPUT_DIR:-}" ]]; then
  echo "ERROR: Set INPUT_DIR to the original sharded HF checkpoint directory." >&2
  exit 1
fi
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "ERROR: Set OUTPUT_DIR to the folded checkpoint output directory." >&2
  exit 1
fi

METRICS_DIR="${METRICS_DIR:-metrics/large_sharded_fold}"
MAX_SHARDS_ARG=""
if [[ -n "${MAX_SHARDS:-}" ]]; then
  MAX_SHARDS_ARG="--max-shards ${MAX_SHARDS}"
fi

mkdir -p "$METRICS_DIR" logs

python src/fold_safetensors_sharded.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --metrics-dir "$METRICS_DIR" \
  --overwrite \
  $MAX_SHARDS_ARG | tee logs/fold_checkpoint_sharded_large.log

echo "Large sharded folding complete. Output: $OUTPUT_DIR"
echo "Metrics: $METRICS_DIR"
