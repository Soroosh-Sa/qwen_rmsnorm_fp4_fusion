#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a TensorRT-LLM checkpoint directory, e.g. ...-NVFP4}"
ENGINE_DIR="${ENGINE_DIR:?Set ENGINE_DIR to the output engine directory}"
CLEAN_ENGINE_DIR="${CLEAN_ENGINE_DIR:-1}"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "ERROR: CHECKPOINT_DIR not found: $CHECKPOINT_DIR" >&2
  exit 2
fi
if ! command -v trtllm-build >/dev/null 2>&1; then
  echo "ERROR: trtllm-build not found in PATH." >&2
  exit 3
fi

# TensorRT-LLM quantize.py exports TensorRT-LLM checkpoints:
#   config.json + rank*.safetensors
# These are intended to be built into TensorRT engines before engine serving.
RANK_COUNT=$(find "$CHECKPOINT_DIR" -maxdepth 1 -name 'rank*.safetensors' | wc -l | tr -d ' ')
if [[ "$RANK_COUNT" == "0" ]]; then
  echo "WARNING: no rank*.safetensors files found in $CHECKPOINT_DIR. This may be an HF/pre-quantized checkpoint, not a TRT-LLM checkpoint." >&2
fi

echo "Building TensorRT-LLM engine"
echo "  CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "  ENGINE_DIR=$ENGINE_DIR"
echo "  RANK_COUNT=$RANK_COUNT"
echo "  MAX_SEQ_LEN=$MAX_SEQ_LEN"
echo "  MAX_INPUT_LEN=$MAX_INPUT_LEN"
echo "  MAX_NUM_TOKENS=$MAX_NUM_TOKENS"
echo "  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-unset}"

if [[ "$CLEAN_ENGINE_DIR" == "1" ]]; then
  rm -rf "$ENGINE_DIR"
fi
mkdir -p "$ENGINE_DIR"
mkdir -p runtime_logs

build_supports_option() {
  local opt="$1"
  trtllm-build --help 2>/dev/null | grep -q -- "$opt"
}

append_build_option_if_supported() {
  local array_name="$1"
  local opt="$2"
  local value="$3"
  local -n arr="$array_name"
  if build_supports_option "$opt"; then
    arr+=("$opt" "$value")
  else
    echo "WARNING: trtllm-build does not advertise option $opt; not passing it."
  fi
}

BUILD_ARGS=(--checkpoint_dir "$CHECKPOINT_DIR" --output_dir "$ENGINE_DIR")
append_build_option_if_supported BUILD_ARGS --max_seq_len "$MAX_SEQ_LEN"
append_build_option_if_supported BUILD_ARGS --max_input_len "$MAX_INPUT_LEN"
append_build_option_if_supported BUILD_ARGS --max_num_tokens "$MAX_NUM_TOKENS"
if [[ -n "${MAX_BATCH_SIZE:-}" ]]; then
  append_build_option_if_supported BUILD_ARGS --max_batch_size "$MAX_BATCH_SIZE"
fi

# Helpful defaults; only passed if available in this container.
if [[ -n "${MAX_BEAM_WIDTH:-}" ]]; then
  append_build_option_if_supported BUILD_ARGS --max_beam_width "$MAX_BEAM_WIDTH"
fi
if [[ -n "${WORKERS:-}" ]]; then
  append_build_option_if_supported BUILD_ARGS --workers "$WORKERS"
fi

echo "Command: trtllm-build ${BUILD_ARGS[*]}"
BUILD_LOG="${BUILD_LOG:-runtime_logs/build_$(basename "$ENGINE_DIR").log}"
echo "Build log: $BUILD_LOG"
# Use tee so errors remain visible and the full build output is saved.
trtllm-build "${BUILD_ARGS[@]}" 2>&1 | tee "$BUILD_LOG"

echo "Validating built engine directory..."
python3 src/validate_trtllm_engine_dir.py --engine-dir "$ENGINE_DIR" --expected-tp "${TP_SIZE:-}"

# Keep tokenizer/generation metadata beside the engine for trtllm-serve/OpenAI API convenience.
for f in tokenizer.json tokenizer_config.json generation_config.json special_tokens_map.json vocab.json merges.txt config.json; do
  if [[ -f "$CHECKPOINT_DIR/$f" ]]; then
    cp -f "$CHECKPOINT_DIR/$f" "$ENGINE_DIR/$f" || true
  fi
done

echo "Engine build finished and validated: $ENGINE_DIR"
