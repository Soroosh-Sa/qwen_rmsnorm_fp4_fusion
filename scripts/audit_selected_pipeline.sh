#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
print_model_selection

ORIG_NVFP4_PATH="${ORIG_NVFP4_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-NVFP4}"
FOLDED_NVFP4_PATH="${FOLDED_NVFP4_PATH:-${MODEL_ROOT:-/workspace/models}/${MODEL_TAG}-FOLDED-NVFP4}"

echo "\n[1/6] Shell/Python syntax check"
bash -n scripts/*.sh
python -m py_compile src/*.py benchmark/*.py

echo "\n[2/6] TP compatibility for selected original model"
bash scripts/check_selected_tp_compatibility.sh || true

echo "\n[3/6] Quantized checkpoint validation if outputs exist"
if [[ -d "$ORIG_NVFP4_PATH" ]]; then
  python src/validate_trtllm_quantized_checkpoint.py --model "$ORIG_NVFP4_PATH" --expected-tp-size "${TP_SIZE:-}" || true
else
  echo "SKIP: original NVFP4 checkpoint missing: $ORIG_NVFP4_PATH"
fi
if [[ -d "$FOLDED_NVFP4_PATH" ]]; then
  python src/validate_trtllm_quantized_checkpoint.py --model "$FOLDED_NVFP4_PATH" --expected-tp-size "${TP_SIZE:-}" || true
else
  echo "SKIP: folded NVFP4 checkpoint missing: $FOLDED_NVFP4_PATH"
fi

echo "\n[4/6] Engine directory validation if engines exist"
ORIG_ENGINE_DIR="${ORIG_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-NVFP4-engine-tp${TP_SIZE}}"
FOLDED_ENGINE_DIR="${FOLDED_ENGINE_DIR:-${ENGINE_ROOT:-/workspace/engines}/${MODEL_TAG}-FOLDED-NVFP4-engine-tp${TP_SIZE}}"
if [[ -d "$ORIG_ENGINE_DIR" ]]; then
  python src/validate_trtllm_engine_dir.py --engine-dir "$ORIG_ENGINE_DIR" --expected-tp "${TP_SIZE:-}"
  python src/validate_trtllm_engine_config.py --engine-dir "$ORIG_ENGINE_DIR"
else
  echo "SKIP: original engine missing: $ORIG_ENGINE_DIR"
fi
if [[ -d "$FOLDED_ENGINE_DIR" ]]; then
  python src/validate_trtllm_engine_dir.py --engine-dir "$FOLDED_ENGINE_DIR" --expected-tp "${TP_SIZE:-}"
  python src/validate_trtllm_engine_config.py --engine-dir "$FOLDED_ENGINE_DIR"
else
  echo "SKIP: folded engine missing: $FOLDED_ENGINE_DIR"
fi

echo "\n[5/6] TensorRT-LLM command availability"
command -v trtllm-serve && trtllm-serve serve --help | head -40 || true
command -v trtllm-build && trtllm-build --help | head -40 || true

echo "\n[6/6] Summary"
echo "Audit completed. Missing checkpoints/engines are OK before those stages run."
echo "For local TRT-LLM quantized checkpoints, use: scripts/run_selected_nvfp4_pair_engine_benchmark.sh"
