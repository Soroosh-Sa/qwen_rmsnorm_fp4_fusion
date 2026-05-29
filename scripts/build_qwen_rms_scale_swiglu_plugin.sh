#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

BUILD_DIR="${BUILD_DIR:-build/qwen_rms_scale_swiglu_plugin}"
PLUGIN_SO="$REPO_ROOT/$BUILD_DIR/libqwen_rms_scale_swiglu_plugin.so"

rm -rf "$BUILD_DIR"
cmake -S csrc/qwen_rms_scale_swiglu \
      -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
cmake --build "$BUILD_DIR" -j"${BUILD_JOBS:-$(nproc)}"

if [[ ! -f "$PLUGIN_SO" ]]; then
  echo "ERROR: plugin .so was not produced: $PLUGIN_SO" >&2
  exit 2
fi

echo "Built plugin: $PLUGIN_SO"

# Registration smoke test. Set SKIP_PLUGIN_REGISTRATION_TEST=1 in containers where
# Python TensorRT is unavailable but C++ TensorRT headers/libs exist.
if [[ "${SKIP_PLUGIN_REGISTRATION_TEST:-0}" != "1" ]]; then
  QWEN_RMS_SCALE_SWIGLU_PLUGIN_SO="$PLUGIN_SO" python - <<'PY'
import ctypes
import os
import tensorrt as trt

so = os.environ["QWEN_RMS_SCALE_SWIGLU_PLUGIN_SO"]
ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
creator = trt.get_plugin_registry().get_plugin_creator("QwenRmsScaleSwiglu", "1", "")
print("plugin_so:", so)
print("creator:", creator)
assert creator is not None, "QwenRmsScaleSwiglu plugin creator was not registered"
print("PASS: QwenRmsScaleSwiglu plugin registered")
PY
fi
