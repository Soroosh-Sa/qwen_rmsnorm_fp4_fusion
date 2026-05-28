#!/usr/bin/env bash
set -euo pipefail

# NVFP4 is the true FP4 path for Blackwell + TensorRT-LLM.
# Use this as the hook for NVIDIA ModelOpt/TensorRT-LLM quantization/export.

: "${FOLDED_BF16_DIR:=outputs/qwen_small_folded_sharded}"
: "${FOLDED_TRTLLM_NVFP4_DIR:=outputs/qwen_small_folded_trtllm_nvfp4}"
: "${METRICS_DIR:=metrics/trtllm_nvfp4}"
mkdir -p "$FOLDED_TRTLLM_NVFP4_DIR" "$METRICS_DIR" logs

cat > "$METRICS_DIR/README.txt" <<EOF2
TensorRT-LLM / ModelOpt NVFP4 quantization placeholder.
Input folded BF16 checkpoint: $FOLDED_BF16_DIR
Output NVFP4 checkpoint/export: $FOLDED_TRTLLM_NVFP4_DIR

Add the exact ModelOpt/TensorRT-LLM command for your installed versions.
Use low-memory/distributed mode for 460B/480B.
EOF2

cat <<EOF2 | tee logs/trtllm_nvfp4_placeholder.log
[placeholder] Quantize folded BF16 checkpoint to NVFP4 for TensorRT-LLM.
Input:  $FOLDED_BF16_DIR
Output: $FOLDED_TRTLLM_NVFP4_DIR
EOF2
