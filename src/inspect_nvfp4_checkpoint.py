#!/usr/bin/env python3
"""Inspect a folded NVFP4 checkpoint without loading full tensors.

This script prints config files, safetensors keys, shapes, dtypes, and likely
NVFP4/scale tensors. It is intended to answer interface questions before adding
a true FP4-output plugin.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


def try_import_safe_open():
    try:
        from safetensors import safe_open  # type: ignore
        return safe_open
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "safetensors is required. Install or run in the TensorRT-LLM env. "
            f"Original error: {exc}"
        )


def should_highlight(name: str) -> bool:
    n = name.lower()
    needles = [
        "fp4", "nvfp4", "scale", "scaling", "alpha", "quant",
        "gate", "up", "down", "fc", "mlp", "expert", "moe",
    ]
    return any(k in n for k in needles)


def iter_safetensors_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*.safetensors"))
    yield from sorted(root.glob("**/*.safetensors"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    ap.add_argument("--max-keys", type=int, default=400)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    root = Path(args.checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {root}")

    lines: list[str] = []
    lines.append(f"checkpoint={root}")
    lines.append("")

    for name in ["config.json", "quant_config.json", "generation_config.json", "pretrained_config.json"]:
        p = root / name
        if p.exists():
            lines.append(f"== {name} ==")
            try:
                obj = json.loads(p.read_text())
                text = json.dumps(obj, indent=2, sort_keys=True)
            except Exception:
                text = p.read_text(errors="replace")
            lines.extend(text.splitlines()[:240])
            lines.append("")

    files = list(dict.fromkeys(iter_safetensors_files(root)))
    lines.append("== safetensors files ==")
    for f in files:
        try:
            size_mb = f.stat().st_size / (1024 * 1024)
        except OSError:
            size_mb = -1
        lines.append(f"{f.relative_to(root)}  size_mb={size_mb:.2f}")
    lines.append("")

    safe_open = try_import_safe_open()
    highlighted: list[str] = []
    total_keys = 0

    for f in files:
        lines.append(f"== keys: {f.relative_to(root)} ==")
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            keys = list(sf.keys())
            total_keys += len(keys)
            for i, k in enumerate(keys):
                if i >= args.max_keys:
                    lines.append(f"... truncated after {args.max_keys} keys in this file")
                    break
                t = sf.get_tensor(k)
                item = f"{k} | shape={tuple(t.shape)} | dtype={t.dtype}"
                lines.append(item)
                if should_highlight(k):
                    highlighted.append(item)
        lines.append("")

    lines.append("== highlighted likely NVFP4/scale/MLP tensors ==")
    for item in highlighted[: max(args.max_keys, 800)]:
        lines.append(item)
    if len(highlighted) > max(args.max_keys, 800):
        lines.append(f"... truncated highlighted list: {len(highlighted)} total")
    lines.append("")
    lines.append(f"total_safetensors_files={len(files)}")
    lines.append(f"total_keys_seen={total_keys}")
    lines.append(f"highlighted_keys={len(highlighted)}")

    text = "\n".join(lines) + "\n"
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"Wrote: {out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
