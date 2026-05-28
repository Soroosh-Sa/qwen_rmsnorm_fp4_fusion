#!/bin/bash
# Central model selector for folding/quantization/serving experiments.
# Source this file after common_env.sh or directly from scripts.

# Usage examples:
#   MODEL_PROFILE=qwen25_15b source scripts/model_profiles.sh
#   MODEL_PROFILE=qwen3_8b MODEL_ROOT=/workspace/models source scripts/model_profiles.sh
#   MODEL_ID=Qwen/Qwen2.5-7B-Instruct MODEL_TAG=my-qwen7b source scripts/model_profiles.sh

MODEL_PROFILE="${MODEL_PROFILE:-qwen25_15b}"

sanitize_model_tag() {
  echo "$1" | sed 's#/#-#g; s#:#-#g; s#[^A-Za-z0-9._-]#-#g'
}

# Explicit MODEL_ID always wins. Otherwise choose from a known profile.
case "$MODEL_PROFILE" in
  qwen25_05b|qwen2.5_0.5b|0.5b)
    DEFAULT_MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
    ;;
  qwen25_15b|qwen2.5_1.5b|1.5b)
    DEFAULT_MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
    ;;
  qwen25_7b|qwen2.5_7b|7b)
    DEFAULT_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
    ;;
  qwen3_17b|qwen3_1.7b|1.7b)
    DEFAULT_MODEL_ID="Qwen/Qwen3-1.7B"
    ;;
  qwen3_4b|4b)
    DEFAULT_MODEL_ID="Qwen/Qwen3-4B"
    ;;
  qwen3_8b|8b)
    DEFAULT_MODEL_ID="Qwen/Qwen3-8B"
    ;;
  qwen3_coder_480b_nvfp4|480b_nvfp4|prequant_nvfp4)
    DEFAULT_MODEL_ID="${PREQUANT_NVFP4_MODEL_ID:-nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4}"
    ;;
  custom)
    if [[ -z "${MODEL_ID:-}" ]]; then
      echo "ERROR: MODEL_PROFILE=custom requires MODEL_ID=..." >&2
      return 2 2>/dev/null || exit 2
    fi
    DEFAULT_MODEL_ID="$MODEL_ID"
    ;;
  *)
    echo "ERROR: Unknown MODEL_PROFILE=$MODEL_PROFILE" >&2
    echo "Known profiles: qwen25_05b, qwen25_15b, qwen25_7b, qwen3_17b, qwen3_4b, qwen3_8b, qwen3_coder_480b_nvfp4, custom" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

export QWEN_MODEL_ID="${MODEL_ID:-$DEFAULT_MODEL_ID}"
export MODEL_TAG="${MODEL_TAG:-$(sanitize_model_tag "$QWEN_MODEL_ID")}"
export ORIGINAL_MODEL_PATH="${ORIGINAL_MODEL_PATH:-${MODEL_ROOT:-/workspace/models}/$MODEL_TAG}"
export FOLDED_BF16_PATH="${FOLDED_BF16_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-FOLDED-BF16}"
export ORIGINAL_NVFP4_PATH="${ORIGINAL_NVFP4_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-NVFP4}"
export FOLDED_NVFP4_PATH="${FOLDED_NVFP4_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-FOLDED-NVFP4}"
export FOLDED_GPTQ_INT4_PATH="${FOLDED_GPTQ_INT4_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-FOLDED-GPTQ-INT4}"

print_model_selection() {
  cat <<EOM
Selected model profile
  MODEL_PROFILE=$MODEL_PROFILE
  QWEN_MODEL_ID=$QWEN_MODEL_ID
  MODEL_TAG=$MODEL_TAG
  ORIGINAL_MODEL_PATH=$ORIGINAL_MODEL_PATH
  FOLDED_BF16_PATH=$FOLDED_BF16_PATH
  ORIGINAL_NVFP4_PATH=$ORIGINAL_NVFP4_PATH
  FOLDED_NVFP4_PATH=$FOLDED_NVFP4_PATH
  FOLDED_GPTQ_INT4_PATH=$FOLDED_GPTQ_INT4_PATH
EOM
}
