#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

TRTLLM_ROOT=$(python - <<'PY'
import importlib.util, os
spec = importlib.util.find_spec("tensorrt_llm")
if spec is None or spec.origin is None:
    raise SystemExit("Could not locate installed tensorrt_llm package")
print(os.path.dirname(spec.origin))
PY
)

resolve_qwen_model_py() {
  python - <<'PY'
import importlib.util, os
from pathlib import Path
spec = importlib.util.find_spec("tensorrt_llm")
root = Path(os.path.dirname(spec.origin))

candidates = [
    root / "models" / "qwen" / "model.py",
    root / "models" / "qwen2" / "model.py",
    root / "models" / "qwen3" / "model.py",
]
for p in candidates:
    if p.exists():
        print(p)
        raise SystemExit

# Fallback: find the file defining QWenDecoderLayer.
for p in (root / "models").rglob("*.py"):
    try:
        text = p.read_text(errors="ignore")
    except Exception:
        continue
    if "class QWenDecoderLayer" in text or "class QwenDecoderLayer" in text:
        print(p)
        raise SystemExit

raise SystemExit(f"Could not find installed Qwen model.py under {root}")
PY
}

QWEN_MODEL_DST="$(resolve_qwen_model_py)"
BUILDER_DST="$TRTLLM_ROOT/builder.py"

if [[ ! -f "$QWEN_MODEL_DST" ]]; then
  echo "ERROR: resolved Qwen model file does not exist: $QWEN_MODEL_DST" >&2
  exit 2
fi
if [[ ! -f "$BUILDER_DST" ]]; then
  echo "ERROR: TensorRT-LLM builder.py does not exist: $BUILDER_DST" >&2
  exit 2
fi

echo "[INFO] TRTLLM_ROOT=$TRTLLM_ROOT"
echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] QWEN_MODEL_DST=$QWEN_MODEL_DST"
echo "[INFO] BUILDER_DST=$BUILDER_DST"

mkdir -p "$REPO_ROOT/patches/trtllm/container_backup"

backup_once() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "$dst" ]]; then
    cp "$src" "$dst"
    echo "[BACKUP] $src -> $dst"
  else
    echo "[BACKUP exists] $dst"
  fi
}

backup_once "$QWEN_MODEL_DST" "$REPO_ROOT/patches/trtllm/container_backup/qwen_model.py"
backup_once "$BUILDER_DST" "$REPO_ROOT/patches/trtllm/container_backup/builder.py"

cp "$REPO_ROOT/patches/trtllm/modified/qwen_model.py" "$QWEN_MODEL_DST"
cp "$REPO_ROOT/patches/trtllm/modified/builder.py" "$BUILDER_DST"

echo "[DONE] Applied TensorRT-LLM patches."
python -m py_compile "$QWEN_MODEL_DST"
python -m py_compile "$BUILDER_DST"
echo "[DONE] Python syntax check passed."
