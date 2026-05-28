#!/usr/bin/env bash
set -euo pipefail

# This script documents the benchmark comparison requested:
#   1) pre-quantized NVFP4 baseline
#   2) folded GPTQ INT4
#   3) folded TensorRT-LLM NVFP4
# It avoids original BF16 runtime comparison by design.

bash scripts/11_write_comparison_manifest.sh

cat <<'EOF2' | tee logs/quantized_comparison_plan.log
Comparison plan:
  A. prequantized_nvfp4       baseline, true FP4/NVFP4, no folding by us
  B. folded_gptq_int4         folded, GPTQ INT4 baseline, not FP4
  C. folded_trtllm_nvfp4      folded, true FP4/NVFP4, TensorRT-LLM target

Metrics to collect:
  - TTFT
  - end-to-end latency
  - tokens/sec
  - peak GPU memory
  - engine/checkpoint size
  - optional small quality metric / perplexity / task accuracy

Replace this placeholder with your actual TRT-LLM/vLLM/serving benchmark commands.
EOF2
