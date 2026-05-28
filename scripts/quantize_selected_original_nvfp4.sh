#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection
export INPUT_MODEL="${INPUT_MODEL:-$ORIGINAL_MODEL_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-NVFP4}"
export QUANT_LOG_DIR="${QUANT_LOG_DIR:-quantization_reports/${MODEL_TAG}-ORIGINAL-NVFP4}"
export QFORMAT="${QFORMAT:-nvfp4}"
export DTYPE="${DTYPE:-bfloat16}"
export CALIB_SIZE="${CALIB_SIZE:-128}"
export CALIB_MAX_SEQ_LENGTH="${CALIB_MAX_SEQ_LENGTH:-512}"
export TOKENIZER_MAX_SEQ_LENGTH="${TOKENIZER_MAX_SEQ_LENGTH:-2048}"
bash "$(dirname "$0")/quantize_trtllm_nvfp4.sh"
