#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

TARGET="${TARGET:-prequant_nvfp4}"
case "$TARGET" in
  prequant_nvfp4)
    MODEL_NAME="${MODEL_NAME:-Qwen3-Coder-480B-A35B-Instruct-NVFP4-prequantized}"
    PLAN_MODEL="${PLAN_MODEL:-$PREQUANT_NVFP4_MODEL_PATH}"
    QUANTIZATION="${QUANTIZATION:-prequantized_nvfp4}"
    DECODE_MODE="${DECODE_MODE:-baseline}"
    ;;
  folded_nvfp4)
    MODEL_NAME="${MODEL_NAME:-Qwen3-Coder-480B-A35B-Instruct-FOLDED-NVFP4}"
    PLAN_MODEL="${PLAN_MODEL:-$FOLDED_NVFP4_MODEL_PATH}"
    QUANTIZATION="${QUANTIZATION:-folded_nvfp4}"
    DECODE_MODE="${DECODE_MODE:-folded}"
    ;;
  folded_gptq_int4)
    MODEL_NAME="${MODEL_NAME:-Qwen3-Coder-480B-A35B-Instruct-FOLDED-GPTQ-INT4}"
    PLAN_MODEL="${PLAN_MODEL:-$FOLDED_GPTQ_INT4_MODEL_PATH}"
    QUANTIZATION="${QUANTIZATION:-folded_gptq_int4}"
    DECODE_MODE="${DECODE_MODE:-folded}"
    ;;
  small_folded_nvfp4)
    MODEL_NAME="${MODEL_NAME:-Qwen3-small-FOLDED-NVFP4}"
    PLAN_MODEL="${PLAN_MODEL:-$QWEN_SMALL_FOLDED_NVFP4_PATH}"
    QUANTIZATION="${QUANTIZATION:-small_folded_nvfp4}"
    DECODE_MODE="${DECODE_MODE:-folded}"
    ;;
  selected_original_nvfp4_engine)
    # Engine-serving target produced by scripts/run_selected_nvfp4_pair_engine_benchmark.sh.
    # PLAN_MODEL is only used by the benchmark client for tokenizer/planning metadata,
    # not as the server model path. The actual running model is ENGINE_DIR.
    MODEL_NAME="${MODEL_NAME:-${MODEL_TAG}-NVFP4-engine-tp${TP_SIZE}}"
    PLAN_MODEL="${PLAN_MODEL:-${ORIG_NVFP4_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-NVFP4}}"
    QUANTIZATION="${QUANTIZATION:-selected_original_nvfp4_engine}"
    DECODE_MODE="${DECODE_MODE:-baseline_engine}"
    ;;
  selected_folded_nvfp4_engine)
    MODEL_NAME="${MODEL_NAME:-${MODEL_TAG}-FOLDED-NVFP4-engine-tp${TP_SIZE}}"
    PLAN_MODEL="${PLAN_MODEL:-${FOLDED_NVFP4_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-FOLDED-NVFP4}}"
    QUANTIZATION="${QUANTIZATION:-selected_folded_nvfp4_engine}"
    DECODE_MODE="${DECODE_MODE:-folded_engine}"
    ;;
  selected_folded_nvfp4_plugin_engine)
    MODEL_NAME="${MODEL_NAME:-${MODEL_TAG}-FOLDED-NVFP4-rms-scale-swiglu-plugin-engine-tp${TP_SIZE}}"
    PLAN_MODEL="${PLAN_MODEL:-${FOLDED_NVFP4_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-FOLDED-NVFP4}}"
    QUANTIZATION="${QUANTIZATION:-selected_folded_nvfp4_plugin_engine}"
    DECODE_MODE="${DECODE_MODE:-folded_plugin_engine}"
    ;;
  *)
    echo "Unknown TARGET=$TARGET"
    echo "Known TARGET values: prequant_nvfp4, folded_nvfp4, folded_gptq_int4, small_folded_nvfp4, selected_original_nvfp4_engine, selected_folded_nvfp4_engine, selected_folded_nvfp4_plugin_engine"
    exit 2
    ;;
esac

OUT="${OUT:-results/${TARGET}_benchmark.csv}"
PLAN_OUT="${PLAN_OUT:-results/${TARGET}_safe_plan.json}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$PLAN_MODEL}"
SERVER_MAX_SEQ_LEN="${SERVER_MAX_SEQ_LEN:-$MAX_SEQ_LEN}"
SERVER_MAX_NUM_TOKENS="${SERVER_MAX_NUM_TOKENS:-$MAX_NUM_TOKENS}"
TIMEOUT_S="${TIMEOUT_S:-900}"
FIRST_TOKEN_TIMEOUT_S="${FIRST_TOKEN_TIMEOUT_S:-$TIMEOUT_S}"
REQUEST_READ_TIMEOUT_S="${REQUEST_READ_TIMEOUT_S:-$FIRST_TOKEN_TIMEOUT_S}"
CASE_TIMEOUT_S="${CASE_TIMEOUT_S:-$((TIMEOUT_S + 300))}"
STOP_ON_CASE_FAILURE="${STOP_ON_CASE_FAILURE:-0}"
SERVER_LOG="${SERVER_LOG:-runtime_logs/${TARGET}_server.log}"

export MODEL_NAME PLAN_MODEL DECODE_MODE QUANTIZATION OUT CONTEXTS CONCURRENCIES
export SERVER_MAX_SEQ_LEN SERVER_MAX_NUM_TOKENS MAX_NEW_TOKENS NUM_REQUESTS SAFETY_TOKENS PROMPT_TOKEN_RESERVE TOKENIZER_PATH
export TP_SIZE KV_DTYPE KV_MEMORY_FRACTION TIMEOUT_S FIRST_TOKEN_TIMEOUT_S REQUEST_READ_TIMEOUT_S CASE_TIMEOUT_S STOP_ON_CASE_FAILURE SERVER_LOG PLAN_OUT

bash scripts/run_benchmark_grid.sh
