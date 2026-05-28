#!/usr/bin/env bash
set -euo pipefail

# Small-model shard-by-shard folding test.
# Use this after downloading/saving a small Qwen HF checkpoint locally.
# Example:
#   export INPUT_DIR="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/<snapshot>"
#   export OUTPUT_DIR="outputs/qwen_small_folded_sharded"
#   bash scripts/02b_fold_checkpoint_sharded_small.sh

INPUT_DIR="${INPUT_DIR:-checkpoints/qwen_small_original}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen_small_folded_sharded}"
METRICS_DIR="${METRICS_DIR:-metrics}"

mkdir -p "$METRICS_DIR" logs outputs checkpoints

python src/fold_safetensors_sharded.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --metrics-dir "$METRICS_DIR" \
  --allow-single-file \
  --overwrite

python src/validate_sharded_folding.py \
  --original-dir "$INPUT_DIR" \
  --folded-dir "$OUTPUT_DIR" \
  --max-tensors 20 \
  --check-norm-ones | tee logs/validate_sharded_small.log
