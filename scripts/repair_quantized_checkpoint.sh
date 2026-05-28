#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

SOURCE_MODEL="${SOURCE_MODEL:-${INPUT_MODEL:-}}"
OUTPUT_MODEL="${OUTPUT_MODEL:-${OUTPUT_DIR:-}}"
REPORT_PATH="${REPORT_PATH:-}"

if [[ -z "$SOURCE_MODEL" || -z "$OUTPUT_MODEL" ]]; then
  cat >&2 <<EOM
ERROR: SOURCE_MODEL and OUTPUT_MODEL are required.
Example:
  SOURCE_MODEL=/workspace/models/Qwen-Qwen2.5-0.5B-Instruct \\
  OUTPUT_MODEL=/workspace/models/Qwen-Qwen2.5-0.5B-Instruct-NVFP4 \\
  bash scripts/repair_quantized_checkpoint.sh
EOM
  exit 2
fi

if [[ -z "$REPORT_PATH" ]]; then
  REPORT_PATH="quantization_reports/$(basename "$OUTPUT_MODEL")/repair_report.json"
fi

python src/repair_trtllm_quantized_checkpoint.py \
  --source "$SOURCE_MODEL" \
  --output "$OUTPUT_MODEL" \
  --check-autoconfig \
  --report "$REPORT_PATH"
