#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/qwen_small_original}"

mkdir -p checkpoints logs
python src/download_hf_checkpoint.py \
  --model "$MODEL_NAME" \
  --output-dir "$OUTPUT_DIR" | tee logs/download_small_checkpoint.log

echo "Downloaded checkpoint to: $OUTPUT_DIR"
