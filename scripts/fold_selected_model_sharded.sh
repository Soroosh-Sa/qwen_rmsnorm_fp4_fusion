#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

INPUT_DIR="${INPUT_DIR:-$ORIGINAL_MODEL_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:-$FOLDED_BF16_PATH}"
METRICS_DIR="${METRICS_DIR:-metrics/$MODEL_TAG}"

mkdir -p "$METRICS_DIR" logs outputs checkpoints
print_model_selection

echo "Folding selected model shard-by-shard"
echo "INPUT_DIR=$INPUT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "METRICS_DIR=$METRICS_DIR"

python src/fold_safetensors_sharded.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --metrics-dir "$METRICS_DIR" \
  --allow-single-file \
  --overwrite | tee "logs/fold_${MODEL_TAG}.log"

python src/validate_sharded_folding.py \
  --original-dir "$INPUT_DIR" \
  --folded-dir "$OUTPUT_DIR" \
  --max-tensors "${MAX_TENSORS:-20}" \
  --check-norm-ones | tee "logs/validate_${MODEL_TAG}.log"
