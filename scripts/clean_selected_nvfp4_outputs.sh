#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/model_profiles.sh"

echo "This will remove generated NVFP4 outputs for MODEL_PROFILE=$MODEL_PROFILE"
echo "  $ORIGINAL_NVFP4_PATH"
echo "  $FOLDED_NVFP4_PATH"

if [[ "${CONFIRM_CLEAN:-0}" != "1" ]]; then
  echo "Set CONFIRM_CLEAN=1 to actually remove these directories."
  exit 0
fi

rm -rf "$ORIGINAL_NVFP4_PATH" "$FOLDED_NVFP4_PATH"
echo "Removed selected NVFP4 output directories."
