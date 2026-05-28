#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
mkdir -p "$MODEL_ROOT"
echo "Downloading baseline: $PREQUANT_NVFP4_MODEL_ID"
huggingface-cli download "$PREQUANT_NVFP4_MODEL_ID" \
  --local-dir "$PREQUANT_NVFP4_MODEL_PATH" \
  --local-dir-use-symlinks False
