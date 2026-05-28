#!/usr/bin/env bash
set -euo pipefail

# GPTQ usually produces INT4 weight-only quantization, not FP4.
# Use this as the hook for your chosen GPTQ tool: GPTQModel, AutoGPTQ,
# llm-compressor, or a company-provided quantizer.

: "${FOLDED_BF16_DIR:=outputs/qwen_small_folded_sharded}"
: "${FOLDED_GPTQ_INT4_DIR:=outputs/qwen_small_folded_gptq_int4}"
: "${METRICS_DIR:=metrics/gptq_int4}"
mkdir -p "$FOLDED_GPTQ_INT4_DIR" "$METRICS_DIR" logs

cat > "$METRICS_DIR/README.txt" <<EOF2
GPTQ INT4 quantization placeholder.
Input folded BF16 checkpoint: $FOLDED_BF16_DIR
Output GPTQ INT4 checkpoint: $FOLDED_GPTQ_INT4_DIR

Add the exact GPTQ command for the chosen tool here.
Remember: GPTQ INT4 is not FP4. It is a 4-bit integer baseline.
EOF2

cat <<EOF2 | tee logs/gptq_int4_placeholder.log
[placeholder] Quantize folded BF16 checkpoint to GPTQ INT4.
Input:  $FOLDED_BF16_DIR
Output: $FOLDED_GPTQ_INT4_DIR

Suggested next action:
  Replace this placeholder with your chosen GPTQ command.
EOF2
