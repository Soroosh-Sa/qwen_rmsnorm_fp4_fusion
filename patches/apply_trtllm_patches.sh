#!/usr/bin/env bash
set -euo pipefail

TRTLLM_ROOT=$(python - <<'PY' 2>/dev/null | tail -n 1
import tensorrt_llm, os
print(os.path.dirname(tensorrt_llm.__file__))
PY
)

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

echo "[INFO] TRTLLM_ROOT=$TRTLLM_ROOT"
echo "[INFO] REPO_ROOT=$REPO_ROOT"

mkdir -p "$REPO_ROOT/patches/trtllm/container_backup"

backup_once() {
  local src="$1"
  local dst="$2"
  if [ ! -f "$dst" ]; then
    cp "$src" "$dst"
    echo "[BACKUP] $src -> $dst"
  else
    echo "[BACKUP exists] $dst"
  fi
}

backup_once \
  "$TRTLLM_ROOT/models/qwen/model.py" \
  "$REPO_ROOT/patches/trtllm/container_backup/qwen_model.py"

backup_once \
  "$TRTLLM_ROOT/builder.py" \
  "$REPO_ROOT/patches/trtllm/container_backup/builder.py"

cp "$REPO_ROOT/patches/trtllm/modified/qwen_model.py" \
   "$TRTLLM_ROOT/models/qwen/model.py"

cp "$REPO_ROOT/patches/trtllm/modified/builder.py" \
   "$TRTLLM_ROOT/builder.py"

echo "[DONE] Applied TensorRT-LLM patches."
python -m py_compile "$TRTLLM_ROOT/models/qwen/model.py"
python -m py_compile "$TRTLLM_ROOT/builder.py"
echo "[DONE] Python syntax check passed."
