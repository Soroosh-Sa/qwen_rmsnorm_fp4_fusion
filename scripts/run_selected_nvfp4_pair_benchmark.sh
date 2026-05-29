#!/bin/bash
set -euo pipefail
# Backward-compatible Stage-6 wrapper.
# Older versions of this script served NVFP4 checkpoints directly or only ran a
# two-way folded-base vs plugin comparison. The current final Stage-6 benchmark
# is the three-way engine comparison:
#   A normal NVFP4 vs B folded NVFP4 base vs C folded NVFP4 plugin.
exec bash "$(dirname "$0")/run_nvfp4_normal_vs_folded_vs_plugin_benchmark.sh"
