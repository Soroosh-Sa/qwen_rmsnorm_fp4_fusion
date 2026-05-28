#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

# Generic TensorRT-LLM / ModelOpt NVFP4 quantization wrapper.
# Works with NVIDIA TensorRT-LLM release containers that include:
#   /app/tensorrt_llm/examples/quantization/quantize.py
# It exports a TensorRT-LLM checkpoint, not a plain Hugging Face checkpoint.

QUANTIZE_SCRIPT="${QUANTIZE_SCRIPT:-/app/tensorrt_llm/examples/quantization/quantize.py}"
if [[ ! -f "$QUANTIZE_SCRIPT" ]]; then
  # Fallback found in some containers.
  ALT="$(python - <<'PY'
import os
candidates = [
    '/app/tensorrt_llm/examples/quantization/quantize.py',
    '/usr/local/lib/python3.12/dist-packages/tensorrt_llm/quantization/quantize.py',
]
for p in candidates:
    if os.path.exists(p):
        print(p)
        raise SystemExit
print('')
PY
)"
  if [[ -n "$ALT" && -f "$ALT" ]]; then
    QUANTIZE_SCRIPT="$ALT"
  else
    echo "ERROR: TensorRT-LLM quantize.py not found." >&2
    echo "Set QUANTIZE_SCRIPT=/path/to/quantize.py" >&2
    exit 2
  fi
fi

INPUT_MODEL="${INPUT_MODEL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
if [[ -z "$INPUT_MODEL" || -z "$OUTPUT_DIR" ]]; then
  cat >&2 <<EOM
ERROR: INPUT_MODEL and OUTPUT_DIR are required.
Example:
  INPUT_MODEL=/workspace/models/Qwen-Qwen2.5-0.5B-Instruct \\
  OUTPUT_DIR=/workspace/models/Qwen-Qwen2.5-0.5B-Instruct-NVFP4 \\
  bash scripts/quantize_trtllm_nvfp4.sh
EOM
  exit 2
fi

QFORMAT="${QFORMAT:-nvfp4}"
DTYPE="${DTYPE:-bfloat16}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-${KV_DTYPE:-fp8}}"
CALIB_DATASET="${CALIB_DATASET:-cnn_dailymail}"
CALIB_SIZE="${CALIB_SIZE:-128}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-1}"
CALIB_MAX_SEQ_LENGTH="${CALIB_MAX_SEQ_LENGTH:-512}"
TOKENIZER_MAX_SEQ_LENGTH="${TOKENIZER_MAX_SEQ_LENGTH:-2048}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
SEED="${SEED:-1234}"
QUANT_LOG_DIR="${QUANT_LOG_DIR:-quantization_reports/$(basename "$OUTPUT_DIR") }"
QUANT_LOG_DIR="$(echo "$QUANT_LOG_DIR" | xargs)"
mkdir -p "$QUANT_LOG_DIR"

# IMPORTANT: TensorRT-LLM/ModelOpt writes rank*.safetensors files whose shapes
# depend on TP_SIZE. Reusing an output directory from a different TP size can
# leave stale rank shards and make trtllm-serve fail during weight loading.
# Generated output dirs are therefore cleaned by default.
CLEAN_OUTPUT_DIR="${CLEAN_OUTPUT_DIR:-1}"
if [[ -d "$OUTPUT_DIR" && "$CLEAN_OUTPUT_DIR" == "1" ]]; then
  echo "Cleaning existing quantization output directory: $OUTPUT_DIR" | tee -a "$QUANT_LOG_DIR/cleanup_output.txt"
  rm -rf "$OUTPUT_DIR"
elif [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "ERROR: OUTPUT_DIR exists and is not empty: $OUTPUT_DIR" >&2
  echo "Set CLEAN_OUTPUT_DIR=1 to remove and regenerate it, or choose a new OUTPUT_DIR." >&2
  exit 4
fi
mkdir -p "$OUTPUT_DIR"

# TensorRT-LLM requires tensor parallelism to divide model attention heads.
# Small Qwen models often cannot use TP_SIZE=4 (for example Qwen2.5-0.5B has
# 14 attention heads and 2 KV heads, so valid TP sizes are 1 or 2).
AUTO_ADJUST_TP_SIZE="${AUTO_ADJUST_TP_SIZE:-1}"
TP_CHECK_OUT="$(python src/check_tp_compatibility.py --model "$INPUT_MODEL" --tp-size "$TP_SIZE" --max-gpus "${MAX_GPUS_FOR_TP:-${GPU_COUNT:-16}}" --print-shell 2>/dev/null || true)"
eval "$TP_CHECK_OUT"
if [[ "${TP_COMPATIBLE:-1}" != "1" ]]; then
  echo "WARNING: Requested TP_SIZE=$TP_SIZE is not compatible with $INPUT_MODEL." | tee -a "$QUANT_LOG_DIR/tp_compatibility.txt"
  echo "  num_attention_heads=${MODEL_NUM_ATTENTION_HEADS:-unknown}" | tee -a "$QUANT_LOG_DIR/tp_compatibility.txt"
  echo "  num_key_value_heads=${MODEL_NUM_KEY_VALUE_HEADS:-unknown}" | tee -a "$QUANT_LOG_DIR/tp_compatibility.txt"
  echo "  valid TP sizes: ${VALID_TP_SIZES:-unknown}" | tee -a "$QUANT_LOG_DIR/tp_compatibility.txt"
  if [[ "$AUTO_ADJUST_TP_SIZE" == "1" && -n "${SUGGESTED_TP_SIZE:-}" ]]; then
    echo "Auto-adjusting TP_SIZE: $TP_SIZE -> $SUGGESTED_TP_SIZE" | tee -a "$QUANT_LOG_DIR/tp_compatibility.txt"
    TP_SIZE="$SUGGESTED_TP_SIZE"
  else
    echo "ERROR: Set TP_SIZE to one of: ${VALID_TP_SIZES:-unknown}, or set AUTO_ADJUST_TP_SIZE=1." >&2
    exit 3
  fi
fi

# Keep a machine-readable config record for reproducibility.
cat > "$QUANT_LOG_DIR/quantize_env.txt" <<EOM
QUANTIZE_SCRIPT=$QUANTIZE_SCRIPT
INPUT_MODEL=$INPUT_MODEL
OUTPUT_DIR=$OUTPUT_DIR
QFORMAT=$QFORMAT
DTYPE=$DTYPE
TP_SIZE=$TP_SIZE
PP_SIZE=$PP_SIZE
CP_SIZE=$CP_SIZE
KV_CACHE_DTYPE=$KV_CACHE_DTYPE
CALIB_DATASET=$CALIB_DATASET
CALIB_SIZE=$CALIB_SIZE
CALIB_BATCH_SIZE=$CALIB_BATCH_SIZE
CALIB_MAX_SEQ_LENGTH=$CALIB_MAX_SEQ_LENGTH
TOKENIZER_MAX_SEQ_LENGTH=$TOKENIZER_MAX_SEQ_LENGTH
DEVICE=$DEVICE
DEVICE_MAP=$DEVICE_MAP
SEED=$SEED
EOM

python - <<'PY' | tee "$QUANT_LOG_DIR/version_info.txt"
import sys
print('python:', sys.version)
for name in ['torch', 'transformers', 'modelopt', 'tensorrt_llm']:
    try:
        m = __import__(name)
        print(f'{name}:', getattr(m, '__version__', 'unknown'))
    except Exception as e:
        print(f'{name}: import failed: {e!r}')
PY

CMD=(
  python "$QUANTIZE_SCRIPT"
  --model_dir "$INPUT_MODEL"
  --output_dir "$OUTPUT_DIR"
  --qformat "$QFORMAT"
  --dtype "$DTYPE"
  --tp_size "$TP_SIZE"
  --pp_size "$PP_SIZE"
  --cp_size "$CP_SIZE"
  --device "$DEVICE"
  --device_map "$DEVICE_MAP"
  --calib_dataset "$CALIB_DATASET"
  --calib_size "$CALIB_SIZE"
  --batch_size "$CALIB_BATCH_SIZE"
  --calib_max_seq_length "$CALIB_MAX_SEQ_LENGTH"
  --tokenizer_max_seq_length "$TOKENIZER_MAX_SEQ_LENGTH"
  --seed "$SEED"
)

# NVFP4 usually benefits from FP8 KV cache in your previous TRT-LLM serving setup.
# Set KV_CACHE_DTYPE=none to skip this argument.
if [[ "$KV_CACHE_DTYPE" != "none" && -n "$KV_CACHE_DTYPE" ]]; then
  CMD+=(--kv_cache_dtype "$KV_CACHE_DTYPE")
fi

# User escape hatch for version-specific options.
if [[ -n "${EXTRA_QUANT_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARRAY=( $EXTRA_QUANT_ARGS )
  CMD+=("${EXTRA_ARRAY[@]}")
fi

printf '%q ' "${CMD[@]}" | tee "$QUANT_LOG_DIR/quantize_command.sh"
echo | tee -a "$QUANT_LOG_DIR/quantize_command.sh"

echo "Running TensorRT-LLM NVFP4 quantization..."
"${CMD[@]}" 2>&1 | tee "$QUANT_LOG_DIR/quantize.log"

# TensorRT-LLM/ModelOpt exports sometimes omit generation_config.json or
# write model_type=qwen for Qwen2.x, which Transformers in the container may
# not recognize. Repair metadata so trtllm-serve can load the exported checkpoint.
REPAIR_AFTER_QUANT="${REPAIR_AFTER_QUANT:-1}"
if [[ "$REPAIR_AFTER_QUANT" == "1" ]]; then
  echo "Repairing quantized checkpoint metadata..."
  python src/repair_trtllm_quantized_checkpoint.py \
    --source "$INPUT_MODEL" \
    --output "$OUTPUT_DIR" \
    --check-autoconfig \
    --report "$QUANT_LOG_DIR/repair_report.json" \
    2>&1 | tee "$QUANT_LOG_DIR/repair.log"
fi

echo "Validating TensorRT-LLM quantized checkpoint layout..."
python src/validate_trtllm_quantized_checkpoint.py   --model "$OUTPUT_DIR"   --expected-tp-size "$TP_SIZE"   --report "$QUANT_LOG_DIR/checkpoint_validation.json"   2>&1 | tee "$QUANT_LOG_DIR/checkpoint_validation.log"

echo "Quantization output: $OUTPUT_DIR"
find "$OUTPUT_DIR" -maxdepth 2 -type f | sort | tee "$QUANT_LOG_DIR/output_files.txt"
