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
# Engine serving note:
# In recent TensorRT-LLM versions, `trtllm-serve serve <engine_dir>` uses the C++/engine path by default.
# Passing `--backend pytorch` forces checkpoint loading, and on some NGC builds passing
# `--backend tensorrt` still sends the request through the LLM API checkpoint-loader path.
# Therefore the default for built engines is: DO NOT pass --backend at all.
# Override only for local debugging, e.g. TRTLLM_ENGINE_BACKEND=pytorch.
TRTLLM_ENGINE_BACKEND="${TRTLLM_ENGINE_BACKEND:-}"
if [[ -n "$TRTLLM_ENGINE_BACKEND" ]]; then
  if trtllm_supports_option --backend; then
    EXTRA_ARGS+=(--backend "$TRTLLM_ENGINE_BACKEND")
  else
    echo "WARNING: requested TRTLLM_ENGINE_BACKEND=$TRTLLM_ENGINE_BACKEND but trtllm-serve does not advertise --backend" >&2
  fi
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

# Important: for engine dirs, keep the engine directory as MODEL and avoid --backend by default.
# Syntax from TensorRT-LLM is: trtllm-serve serve [OPTIONS] MODEL, where MODEL may be a TensorRT engine path.
trtllm-serve serve "${EXTRA_ARGS[@]}" "$ENGINE_DIR"
