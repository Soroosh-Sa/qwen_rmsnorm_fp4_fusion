#!/bin/bash
set -euo pipefail
# Backward-compatible Stage-6 engine wrapper.
# The final comparison is now the three-way engine benchmark:
#   A normal NVFP4 vs B folded NVFP4 base vs C folded NVFP4 plugin.
exec bash "$(dirname "$0")/run_nvfp4_normal_vs_folded_vs_plugin_benchmark.sh"
