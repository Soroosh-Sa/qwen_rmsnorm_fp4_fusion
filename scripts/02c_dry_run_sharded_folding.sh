#!/usr/bin/env bash
set -euo pipefail

# Dry run: scans a local HF safetensors checkpoint and reports exactly which tensors would be changed.
# No output checkpoint shards are written.

INPUT_DIR="${INPUT_DIR:-checkpoints/qwen_small_original}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dry_run_unused}"
METRICS_DIR="${METRICS_DIR:-metrics}"
MAX_SHARDS_ARG=""
if [[ -n "${MAX_SHARDS:-}" ]]; then
  MAX_SHARDS_ARG="--max-shards ${MAX_SHARDS}"
fi

mkdir -p "$METRICS_DIR" logs

python src/fold_safetensors_sharded.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --metrics-dir "$METRICS_DIR" \
  --allow-single-file \
  --dry-run \
  $MAX_SHARDS_ARG | tee logs/dry_run_sharded_folding.log
