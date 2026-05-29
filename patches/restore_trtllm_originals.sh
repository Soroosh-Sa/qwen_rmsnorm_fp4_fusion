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
for p in [root / "models" / "qwen" / "model.py", root / "models" / "qwen2" / "model.py", root / "models" / "qwen3" / "model.py"]:
    if p.exists():
        print(p)
        raise SystemExit
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

if [[ -f "$REPO_ROOT/patches/trtllm/container_backup/qwen_model.py" ]]; then
  cp "$REPO_ROOT/patches/trtllm/container_backup/qwen_model.py" "$QWEN_MODEL_DST"
  echo "[RESTORE] qwen_model.py -> $QWEN_MODEL_DST"
else
  echo "[SKIP] no backup for qwen_model.py"
fi

if [[ -f "$REPO_ROOT/patches/trtllm/container_backup/builder.py" ]]; then
  cp "$REPO_ROOT/patches/trtllm/container_backup/builder.py" "$BUILDER_DST"
  echo "[RESTORE] builder.py -> $BUILDER_DST"
else
  echo "[SKIP] no backup for builder.py"
fi
