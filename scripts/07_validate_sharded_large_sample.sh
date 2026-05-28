#!/usr/bin/env bash
set -euo pipefail

# Sample validation for a large folded checkpoint.
# It checks only the first N target tensors, so it stays lightweight.

if [[ -z "${ORIGINAL_DIR:-}" ]]; then
  echo "ERROR: Set ORIGINAL_DIR to the original checkpoint directory." >&2
  exit 1
fi
if [[ -z "${FOLDED_DIR:-}" ]]; then
  echo "ERROR: Set FOLDED_DIR to the folded checkpoint directory." >&2
  exit 1
fi

MAX_TENSORS="${MAX_TENSORS:-32}"
mkdir -p logs

python src/validate_sharded_folding.py \
  --original-dir "$ORIGINAL_DIR" \
  --folded-dir "$FOLDED_DIR" \
  --max-tensors "$MAX_TENSORS" \
  --check-norm-ones | tee logs/validate_sharded_large_sample.log
