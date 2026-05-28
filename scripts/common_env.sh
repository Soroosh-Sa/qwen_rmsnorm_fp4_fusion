#!/bin/bash
set -euo pipefail

# Shared environment for folded Qwen quantization + TensorRT-LLM benchmarking.
# Designed to match the previous Vast AI TensorRT-LLM workflow.

export PYTHON_BIN="${PYTHON_BIN:-python3}"

# Cache/model locations
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export MODEL_ROOT="${MODEL_ROOT:-/workspace/models}"
export ENGINE_ROOT="${ENGINE_ROOT:-/workspace/engines}"

# Public pre-quantized baseline from prior assignment.
export PREQUANT_NVFP4_MODEL_ID="${PREQUANT_NVFP4_MODEL_ID:-nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4}"
export PREQUANT_NVFP4_MODEL_PATH="${PREQUANT_NVFP4_MODEL_PATH:-$MODEL_ROOT/Qwen3-Coder-480B-A35B-Instruct-NVFP4}"

# Folded model paths produced by this project.
export FOLDED_BF16_MODEL_PATH="${FOLDED_BF16_MODEL_PATH:-$MODEL_ROOT/Qwen3-Coder-480B-A35B-Instruct-FOLDED-BF16}"
export FOLDED_NVFP4_MODEL_PATH="${FOLDED_NVFP4_MODEL_PATH:-$MODEL_ROOT/Qwen3-Coder-480B-A35B-Instruct-FOLDED-NVFP4}"
export FOLDED_GPTQ_INT4_MODEL_PATH="${FOLDED_GPTQ_INT4_MODEL_PATH:-$MODEL_ROOT/Qwen3-Coder-480B-A35B-Instruct-FOLDED-GPTQ-INT4}"

# Small models for smoke tests.
export QWEN_SMALL_ID="${QWEN_SMALL_ID:-Qwen/Qwen3-1.7B}"
export QWEN_SMALL_PATH="${QWEN_SMALL_PATH:-$MODEL_ROOT/Qwen3-1.7B}"
export QWEN_SMALL_FOLDED_PATH="${QWEN_SMALL_FOLDED_PATH:-$MODEL_ROOT/Qwen3-1.7B-FOLDED-BF16}"
export QWEN_SMALL_FOLDED_NVFP4_PATH="${QWEN_SMALL_FOLDED_NVFP4_PATH:-$MODEL_ROOT/Qwen3-1.7B-FOLDED-NVFP4}"
export QWEN_SMALL_FOLDED_GPTQ_INT4_PATH="${QWEN_SMALL_FOLDED_GPTQ_INT4_PATH:-$MODEL_ROOT/Qwen3-1.7B-FOLDED-GPTQ-INT4}"

# TensorRT-LLM server defaults.
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export TP_SIZE="${TP_SIZE:-4}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
export MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-$MAX_SEQ_LEN}"
export MAX_INPUT_LEN="${MAX_INPUT_LEN:-$MAX_SEQ_LEN}"
export MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-}"
export KV_DTYPE="${KV_DTYPE:-fp8}"
export KV_MEMORY_FRACTION="${KV_MEMORY_FRACTION:-0.70}"

# Benchmark defaults.
export CONTEXTS="${CONTEXTS:-1024 8192 32768}"
export CONCURRENCIES="${CONCURRENCIES:-1 2}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export NUM_REQUESTS="${NUM_REQUESTS:-8}"
export SAFETY_TOKENS="${SAFETY_TOKENS:-1024}"
export PROMPT_TOKEN_RESERVE="${PROMPT_TOKEN_RESERVE:-$SAFETY_TOKENS}"
export OPENAI_API_MODE="${OPENAI_API_MODE:-chat}"
export PROMPT_PROFILE="${PROMPT_PROFILE:-synthetic_code_context}"

# Dependency pins that worked better with TensorRT-LLM containers in the old repo.
export PIN_NUMPY="numpy>=1.26,<2"
export PIN_PANDAS="pandas>=2.3,<3"
export PIN_REQUESTS="requests>=2.32,<3"
export PIN_TQDM="tqdm>=4.66,<5"
export PIN_HF_HUB="huggingface_hub[cli]>=0.34,<1.0"

safe_install_benchmark_deps() {
  echo "Installing/repairing safe benchmark dependencies..."
  "$PYTHON_BIN" -m pip install \
    "$PIN_NUMPY" \
    "$PIN_PANDAS" \
    "$PIN_REQUESTS" \
    "$PIN_TQDM" \
    "$PIN_HF_HUB"
}

verify_core_versions() {
  "$PYTHON_BIN" - <<'PY'
import numpy
print('numpy:', numpy.__version__)
try:
    import huggingface_hub
    print('huggingface_hub:', huggingface_hub.__version__)
except Exception as e:
    print('huggingface_hub import failed:', repr(e))
try:
    import transformers
    print('transformers:', transformers.__version__)
except Exception as e:
    print('transformers import failed:', repr(e))
try:
    import tensorrt_llm
    print('TensorRT-LLM import: OK')
except Exception as e:
    print('TensorRT-LLM import failed:', repr(e))
PY
}

trtllm_supports_option() {
  local opt="$1"
  trtllm-serve serve --help 2>/dev/null | grep -q -- "$opt"
}

append_trtllm_option_if_supported() {
  local array_name="$1"
  local opt="$2"
  local value="$3"
  local -n arr="$array_name"
  if trtllm_supports_option "$opt"; then
    arr+=("$opt" "$value")
  else
    echo "WARNING: trtllm-serve does not advertise option $opt; not passing it."
  fi
}

# Optional unified model selector. This does not override the older target-specific
# variables, but it gives new scripts a consistent MODEL_PROFILE/MODEL_ID interface.
if [[ -f "$(dirname "${BASH_SOURCE[0]}")/model_profiles.sh" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/model_profiles.sh"
fi
