#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
mkdir -p results logs runtime_logs runs metrics checkpoints outputs engines trtllm quantization_reports "$HF_HOME" "$MODEL_ROOT"
safe_install_benchmark_deps
verify_core_versions || true
nvidia-smi || true
