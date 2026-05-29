#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection

# Correct folded NVFP4 workflow:
#   folded BF16/FP16 checkpoint with RMSNorm gamma already folded into MLP gate/up weights
#   -> TensorRT-LLM/ModelOpt NVFP4 checkpoint.
# Do not point INPUT_MODEL at the original unfused model for plugin experiments.
export INPUT_MODEL="${INPUT_MODEL:-$FOLDED_BF16_PATH}"
export OUTPUT_DIR="${OUTPUT_DIR:-$FOLDED_NVFP4_PATH}"
export QUANT_LOG_DIR="${QUANT_LOG_DIR:-quantization_reports/${MODEL_TAG}-FOLDED-NVFP4}"
export QFORMAT="${QFORMAT:-nvfp4}"
export DTYPE="${DTYPE:-bfloat16}"
export CALIB_SIZE="${CALIB_SIZE:-128}"
export CALIB_MAX_SEQ_LENGTH="${CALIB_MAX_SEQ_LENGTH:-512}"
export TOKENIZER_MAX_SEQ_LENGTH="${TOKENIZER_MAX_SEQ_LENGTH:-2048}"

if [[ ! -d "$INPUT_MODEL" ]]; then
  echo "ERROR: folded BF16 input checkpoint not found: $INPUT_MODEL" >&2
  echo "Expected folded path: $FOLDED_BF16_PATH" >&2
  echo "Create it with: MODEL_PROFILE=$MODEL_PROFILE bash scripts/fold_selected_model_sharded.sh" >&2
  exit 2
fi

if [[ "$INPUT_MODEL" == "$ORIGINAL_MODEL_PATH" ]]; then
  echo "ERROR: INPUT_MODEL points to the original model, not folded BF16: $INPUT_MODEL" >&2
  echo "For folded NVFP4 plugin experiments, INPUT_MODEL must be: $FOLDED_BF16_PATH" >&2
  exit 3
fi

if [[ "$OUTPUT_DIR" == "$INPUT_MODEL" ]]; then
  echo "ERROR: OUTPUT_DIR must be different from INPUT_MODEL." >&2
  exit 4
fi

# Lightweight sanity check: warn if the folder name does not look folded.
if [[ "$(basename "$INPUT_MODEL")" != *FOLDED* ]]; then
  echo "WARNING: INPUT_MODEL path name does not contain FOLDED: $INPUT_MODEL" >&2
  echo "Continuing because explicit INPUT_MODEL was provided, but verify the weights were folded first." >&2
fi

cat <<EOM
Folded NVFP4 quantization
  INPUT_MODEL=$INPUT_MODEL
  OUTPUT_DIR=$OUTPUT_DIR
  QFORMAT=$QFORMAT
  TP_SIZE=${TP_SIZE:-}
  CALIB_SIZE=$CALIB_SIZE
  CALIB_MAX_SEQ_LENGTH=$CALIB_MAX_SEQ_LENGTH
EOM

bash "$(dirname "$0")/quantize_trtllm_nvfp4.sh"
