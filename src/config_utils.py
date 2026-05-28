from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        if path.suffix == ".json":
            return json.load(f)
        raise ValueError(f"Unsupported config suffix: {path.suffix}")


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for key in ["metrics_dir", "logs_dir"]:
        if key in cfg and cfg[key]:
            Path(cfg[key]).mkdir(parents=True, exist_ok=True)
    if cfg.get("folded_output_dir"):
        Path(cfg["folded_output_dir"]).mkdir(parents=True, exist_ok=True)


def dtype_from_string(name: str):
    import torch

    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")
