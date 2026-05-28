#!/usr/bin/env python3
"""Check/suggest TensorRT-LLM tensor-parallel size for a HF/TensorRT checkpoint.

For Qwen small models, tp_size must divide attention heads. In practice it is
also safest to require it to divide num_key_value_heads when present.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def load_config(model_dir: str) -> dict:
    p = Path(model_dir) / "config.json"
    if not p.exists():
        raise FileNotFoundError(f"config.json not found under {model_dir}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def valid_tps(cfg: dict, max_gpus: int) -> list[int]:
    n_heads = cfg.get("num_attention_heads") or cfg.get("n_head") or cfg.get("num_heads")
    n_kv = cfg.get("num_key_value_heads", n_heads)
    if n_heads is None:
        # Cannot infer; be conservative.
        return [1]
    vals = []
    for tp in range(1, max_gpus + 1):
        if n_heads % tp == 0 and (n_kv is None or n_kv % tp == 0):
            vals.append(tp)
    return vals or [1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Checkpoint/model directory containing config.json")
    ap.add_argument("--tp-size", type=int, default=None, help="Requested TP size to validate")
    ap.add_argument("--max-gpus", type=int, default=8)
    ap.add_argument("--print-shell", action="store_true", help="Print shell exports for AUTO_TP usage")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.model)
    n_heads = cfg.get("num_attention_heads") or cfg.get("n_head") or cfg.get("num_heads")
    n_kv = cfg.get("num_key_value_heads", n_heads)
    model_type = cfg.get("model_type", "unknown")
    vals = valid_tps(cfg, args.max_gpus)
    suggested = vals[-1]

    ok = True
    if args.tp_size is not None:
        ok = args.tp_size in vals

    if args.print_shell:
        print(f"export VALID_TP_SIZES='{','.join(map(str, vals))}'")
        print(f"export SUGGESTED_TP_SIZE='{suggested}'")
        print(f"export TP_COMPATIBLE='{'1' if ok else '0'}'")
        print(f"export MODEL_NUM_ATTENTION_HEADS='{n_heads}'")
        print(f"export MODEL_NUM_KEY_VALUE_HEADS='{n_kv}'")
        return 0 if ok else 3

    if not args.quiet:
        print("Tensor parallel compatibility")
        print(f"  model={args.model}")
        print(f"  model_type={model_type}")
        print(f"  num_attention_heads={n_heads}")
        print(f"  num_key_value_heads={n_kv}")
        print(f"  valid_tp_sizes<=max_gpus={vals}")
        print(f"  suggested_tp_size={suggested}")
        if args.tp_size is not None:
            print(f"  requested_tp_size={args.tp_size}")
            print(f"  compatible={ok}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
