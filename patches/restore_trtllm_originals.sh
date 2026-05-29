#!/usr/bin/env bash
set -euo pipefail

TRTLLM_ROOT=$(python - <<'PY' 2>/dev/null | tail -n 1
import tensorrt_llm, os
print(os.path.dirname(tensorrt_llm.__file__))
PY
)

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

echo "[INFO] TRTLLM_ROOT=$TRTLLM_ROOT"

cp "$REPO_ROOT/patches/trtllm/original/qwen_model.py" \
   "$TRTLLM_ROOT/models/qwen/model.py"

cp "$REPO_ROOT/patches/trtllm/original/builder.py" \
   "$TRTLLM_ROOT/builder.py"

echo "[DONE] Restored original TensorRT-LLM files."
python -m py_compile "$TRTLLM_ROOT/models/qwen/model.py"
python -m py_compile "$TRTLLM_ROOT/builder.py"
echo "[DONE] Python syntax check passed."
