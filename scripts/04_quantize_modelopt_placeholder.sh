#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs metrics outputs

cat <<'MSG'
This is a placeholder for ModelOpt/TensorRT-LLM FP4/NVFP4 quantization.

Suggested flow:
  input checkpoint:  outputs/qwen_folded_rmsnorm
  output checkpoint: outputs/qwen_folded_nvfp4

Add the exact command required by the company environment here after confirming:
  - TensorRT-LLM version
  - ModelOpt version
  - target FP4 format: nvfp4 / fp4 / mxfp4
  - calibration dataset and sequence length
  - expected export format

Important:
  Quantize the folded checkpoint, not the original checkpoint.
MSG
