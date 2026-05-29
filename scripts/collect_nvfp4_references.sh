#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-runtime_logs/nvfp4_refs}"
TRTLLM_PKG="${TRTLLM_PKG:-/usr/local/lib/python3.12/dist-packages/tensorrt_llm}"
NATIVE_DIR="${NATIVE_DIR:-third_party/tensorrt_llm_native}"

mkdir -p "$OUT_DIR"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$OUT_DIR/$dst"
    echo "copied: $src -> $OUT_DIR/$dst"
  else
    echo "missing: $src" | tee -a "$OUT_DIR/missing.txt"
  fi
}

: > "$OUT_DIR/summary.txt"
: > "$OUT_DIR/missing.txt"

{
  echo "TensorRT-LLM package: $TRTLLM_PKG"
  echo "Native source dir:    $NATIVE_DIR"
  echo "Output dir:           $OUT_DIR"
  echo
  echo "== grep summary =="
} >> "$OUT_DIR/summary.txt"

# Installed Python NVFP4 / FP4 references.
copy_if_exists "$TRTLLM_PKG/_torch/auto_deploy/custom_ops/linear/swiglu.py" "swiglu.py"
copy_if_exists "$TRTLLM_PKG/_torch/auto_deploy/custom_ops/quantization/quant.py" "quant.py"
copy_if_exists "$TRTLLM_PKG/_torch/auto_deploy/custom_ops/quantization/torch_quant.py" "torch_quant.py"
copy_if_exists "$TRTLLM_PKG/_torch/auto_deploy/custom_ops/fused_moe/trtllm_moe.py" "trtllm_moe.py"
copy_if_exists "$TRTLLM_PKG/_torch/auto_deploy/config/default.yaml" "default.yaml"
copy_if_exists "$TRTLLM_PKG/_torch/auto_deploy/transform/library/fuse_relu2_quant_nvfp4.py" "fuse_relu2_quant_nvfp4.py"
copy_if_exists "$TRTLLM_PKG/_torch/auto_deploy/transform/library/fuse_swiglu.py" "fuse_swiglu.py"
copy_if_exists "$TRTLLM_PKG/_torch/auto_deploy/transform/library/fuse_quant.py" "fuse_quant.py"

# Native C++ / CUDA references if this repo has them.
copy_if_exists "$NATIVE_DIR/kernels/fusedGatedRMSNormQuant.cu" "fusedGatedRMSNormQuant.cu"
copy_if_exists "$NATIVE_DIR/kernels/fusedGatedRMSNormQuant.cuh" "fusedGatedRMSNormQuant.cuh"
copy_if_exists "$NATIVE_DIR/kernels/groupRmsNormKernels.cu" "groupRmsNormKernels.cu"
copy_if_exists "$NATIVE_DIR/kernels/groupRmsNormKernels.h" "groupRmsNormKernels.h"
copy_if_exists "$NATIVE_DIR/plugins/gemmSwigluPlugin.cpp" "gemmSwigluPlugin.cpp"
copy_if_exists "$NATIVE_DIR/plugins/gemmSwigluPlugin.cu" "gemmSwigluPlugin.cu"
copy_if_exists "$NATIVE_DIR/plugins/gemmSwigluPlugin.h" "gemmSwigluPlugin.h"

# Compact searchable snippets.
for f in "$OUT_DIR"/*.py "$OUT_DIR"/*.cu "$OUT_DIR"/*.cuh "$OUT_DIR"/*.h "$OUT_DIR"/*.cpp "$OUT_DIR"/*.yaml; do
  [[ -f "$f" ]] || continue
  echo >> "$OUT_DIR/summary.txt"
  echo "== $(basename "$f") :: NVFP4/FP4/SwiGLU/scale snippets ==" >> "$OUT_DIR/summary.txt"
  grep -nEi "NVFP4|nvfp4|FP4|fp4|swiglu|SwiGLU|scale|scaling|quantize|block" "$f" \
    | head -160 >> "$OUT_DIR/summary.txt" || true
done

echo
echo "Wrote NVFP4 reference bundle to: $OUT_DIR"
echo "Key file: $OUT_DIR/summary.txt"
