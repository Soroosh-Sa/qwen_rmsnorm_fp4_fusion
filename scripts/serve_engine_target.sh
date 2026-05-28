#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

TARGET="${TARGET:-selected_original_nvfp4_engine}"
case "$TARGET" in
  selected_original_nvfp4_engine)
    ENGINE_DIR="${ENGINE_DIR:-${ORIG_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-NVFP4-engine-tp${TP_SIZE}}}"
    TOKENIZER_DIR="${TOKENIZER_DIR:-${ORIG_NVFP4_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-NVFP4}}"
    ;;
  selected_folded_nvfp4_engine)
    ENGINE_DIR="${ENGINE_DIR:-${FOLDED_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-engine-tp${TP_SIZE}}}"
    TOKENIZER_DIR="${TOKENIZER_DIR:-$FOLDED_NVFP4_PATH}"
    ;;
  *)
    echo "ERROR: Unknown engine TARGET=$TARGET" >&2
    exit 2
    ;;
esac

if [[ ! -d "$ENGINE_DIR" ]]; then
  echo "ERROR: ENGINE_DIR not found: $ENGINE_DIR" >&2
  echo "Build it first with: bash scripts/build_selected_nvfp4_pair_engines.sh" >&2
  exit 3
fi

# Fail early if this directory is not actually a serialized TensorRT-LLM engine.
python3 src/validate_trtllm_engine_dir.py --engine-dir "$ENGINE_DIR" --expected-tp "$TP_SIZE"

EXTRA_ARGS=(
  --host "$HOST"
  --port "$PORT"
)
# Current TensorRT-LLM docs list `pytorch | tensorrt | _autodeploy`; default is PyTorch,
# so force TensorRT when serving a built engine.
if trtllm_supports_option --backend; then
  EXTRA_ARGS+=(--backend tensorrt)
fi
append_trtllm_option_if_supported EXTRA_ARGS --max_seq_len "$MAX_SEQ_LEN"
append_trtllm_option_if_supported EXTRA_ARGS --max_num_tokens "$MAX_NUM_TOKENS"
append_trtllm_option_if_supported EXTRA_ARGS --max_input_len "$MAX_INPUT_LEN"
if [[ -n "$MAX_BATCH_SIZE" ]]; then
  append_trtllm_option_if_supported EXTRA_ARGS --max_batch_size "$MAX_BATCH_SIZE"
fi
if [[ -d "$TOKENIZER_DIR" ]]; then
  if trtllm_supports_option --tokenizer; then
    EXTRA_ARGS+=(--tokenizer "$TOKENIZER_DIR")
  elif trtllm_supports_option --tokenizer_dir; then
    EXTRA_ARGS+=(--tokenizer_dir "$TOKENIZER_DIR")
  fi
fi

echo "Starting TensorRT-LLM engine server"
echo "TARGET=$TARGET"
echo "ENGINE_DIR=$ENGINE_DIR"
echo "TOKENIZER_DIR=$TOKENIZER_DIR"
echo "MAX_SEQ_LEN=$MAX_SEQ_LEN"
echo "MAX_NUM_TOKENS=$MAX_NUM_TOKENS"
echo "EXTRA_ARGS=${EXTRA_ARGS[*]}"

# Important: force TensorRT engine path, never PyTorch checkpoint loading.
export TLLM_USE_TRT_ENGINE=1
trtllm-serve serve "${EXTRA_ARGS[@]}" "$ENGINE_DIR"
