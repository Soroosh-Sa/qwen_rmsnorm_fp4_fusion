#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

TARGET="${TARGET:-prequant_nvfp4}"
case "$TARGET" in
  prequant_nvfp4)
    MODEL_PATH="${MODEL_PATH:-$PREQUANT_NVFP4_MODEL_PATH}"
    CONFIG_PATH="${CONFIG_PATH:-configs/trtllm_prequant_nvfp4.yaml}"
    ;;
  folded_nvfp4)
    MODEL_PATH="${MODEL_PATH:-$FOLDED_NVFP4_MODEL_PATH}"
    CONFIG_PATH="${CONFIG_PATH:-configs/trtllm_folded_nvfp4.yaml}"
    ;;
  folded_gptq_int4)
    MODEL_PATH="${MODEL_PATH:-$FOLDED_GPTQ_INT4_MODEL_PATH}"
    CONFIG_PATH="${CONFIG_PATH:-configs/trtllm_folded_gptq_int4.yaml}"
    ;;
  small_folded_nvfp4)
    MODEL_PATH="${MODEL_PATH:-$QWEN_SMALL_FOLDED_NVFP4_PATH}"
    CONFIG_PATH="${CONFIG_PATH:-configs/trtllm_small_folded_nvfp4.yaml}"
    ;;
  selected_original)
    MODEL_PATH="${MODEL_PATH:-$ORIGINAL_MODEL_PATH}"
    CONFIG_PATH="${CONFIG_PATH:-configs/trtllm_prequant_nvfp4.yaml}"
    ;;
  selected_original_nvfp4)
    MODEL_PATH="${MODEL_PATH:-${ORIGINAL_NVFP4_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-NVFP4}}"
    CONFIG_PATH="${CONFIG_PATH:-configs/trtllm_small_folded_nvfp4.yaml}"
    ;;
  selected_folded_nvfp4)
    MODEL_PATH="${MODEL_PATH:-$FOLDED_NVFP4_PATH}"
    CONFIG_PATH="${CONFIG_PATH:-configs/trtllm_small_folded_nvfp4.yaml}"
    ;;
  selected_folded_gptq_int4)
    MODEL_PATH="${MODEL_PATH:-$FOLDED_GPTQ_INT4_PATH}"
    CONFIG_PATH="${CONFIG_PATH:-configs/trtllm_folded_gptq_int4.yaml}"
    ;;
  *)
    echo "Unknown TARGET=$TARGET. Use prequant_nvfp4, folded_nvfp4, folded_gptq_int4, small_folded_nvfp4, selected_original, selected_original_nvfp4, selected_folded_nvfp4, selected_folded_gptq_int4."
    exit 2
    ;;
esac

EXTRA_ARGS=(
  --backend pytorch
  --host "$HOST"
  --port "$PORT"
  --tp_size "$TP_SIZE"
  --max_seq_len "$MAX_SEQ_LEN"
)
append_trtllm_option_if_supported EXTRA_ARGS --max_num_tokens "$MAX_NUM_TOKENS"
append_trtllm_option_if_supported EXTRA_ARGS --max_input_len "$MAX_INPUT_LEN"
if [[ -n "$MAX_BATCH_SIZE" ]]; then
  append_trtllm_option_if_supported EXTRA_ARGS --max_batch_size "$MAX_BATCH_SIZE"
fi
if [[ -f "$CONFIG_PATH" ]]; then
  if trtllm_supports_option --extra_llm_api_options; then
    EXTRA_ARGS+=(--extra_llm_api_options "$CONFIG_PATH")
  elif trtllm_supports_option --config; then
    EXTRA_ARGS+=(--config "$CONFIG_PATH")
  else
    echo "WARNING: no config/extra_llm_api_options flag found; launching without config file."
  fi
else
  echo "WARNING: config file not found: $CONFIG_PATH; launching without config file."
fi

echo "Starting TensorRT-LLM target server"
echo "TARGET=$TARGET"
echo "MODEL_PATH=$MODEL_PATH"
echo "TP_SIZE=$TP_SIZE"
echo "MAX_SEQ_LEN=$MAX_SEQ_LEN"
echo "MAX_NUM_TOKENS=$MAX_NUM_TOKENS"
echo "MAX_INPUT_LEN=$MAX_INPUT_LEN"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "EXTRA_ARGS=${EXTRA_ARGS[*]}"

trtllm-serve serve "${EXTRA_ARGS[@]}" "$MODEL_PATH"
