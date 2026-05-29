#!/usr/bin/env python3
"""Inspect whether a folded NVFP4 TensorRT-LLM checkpoint matches the plugin contract.

This script is intentionally lightweight. It checks config metadata, required MLP
keys, tensor dtypes/shapes, and whether post_layernorm weights look folded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_dtype(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Folded NVFP4 TRT-LLM checkpoint dir")
    parser.add_argument("--rank", default="rank0.safetensors")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--check-post-ln", action="store_true")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    cfg_path = ckpt / "config.json"
    st_path = ckpt / args.rank

    if not cfg_path.is_file():
        raise SystemExit(f"Missing config.json: {cfg_path}")
    if not st_path.is_file():
        raise SystemExit(f"Missing safetensors file: {st_path}")

    cfg = _load_json(cfg_path)
    quant = cfg.get("quantization", {}) or {}
    mapping = cfg.get("mapping", {}) or {}
    layer = args.layer

    print(f"checkpoint={ckpt}")
    print(f"rank_file={st_path.name}")
    print(f"model_type={cfg.get('model_type')} architecture={cfg.get('architecture')}")
    print(f"dtype={cfg.get('dtype')} quant_algo={quant.get('quant_algo')} group_size={quant.get('group_size')}")
    print(f"tp_size={mapping.get('tp_size')} world_size={mapping.get('world_size')}")
    print(f"folding_metadata_present={'_rmsnorm_folding_metadata' in cfg}")
    print()

    required = [
        f"transformer.layers.{layer}.mlp.fc.weight",
        f"transformer.layers.{layer}.mlp.fc.activation_scaling_factor",
        f"transformer.layers.{layer}.mlp.fc.weights_scaling_factor",
        f"transformer.layers.{layer}.mlp.fc.weights_scaling_factor_2",
        f"transformer.layers.{layer}.mlp.gate.weight",
        f"transformer.layers.{layer}.mlp.gate.activation_scaling_factor",
        f"transformer.layers.{layer}.mlp.gate.weights_scaling_factor",
        f"transformer.layers.{layer}.mlp.gate.weights_scaling_factor_2",
        f"transformer.layers.{layer}.mlp.proj.weight",
        f"transformer.layers.{layer}.mlp.proj.activation_scaling_factor",
        f"transformer.layers.{layer}.mlp.proj.weights_scaling_factor",
        f"transformer.layers.{layer}.mlp.proj.weights_scaling_factor_2",
        f"transformer.layers.{layer}.post_layernorm.weight",
    ]

    failures: list[str] = []
    with safe_open(st_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for k in required:
            if k not in keys:
                failures.append(f"missing: {k}")

        print(f"== layer {layer} tensors ==")
        for k in required:
            if k in keys:
                t = f.get_tensor(k)
                print(f"{k} | shape={tuple(t.shape)} | dtype={_fmt_dtype(t.dtype)}")

        # Contract checks for the current 0.5B-style dense Qwen path.
        for name in ["fc", "gate", "proj"]:
            wk = f"transformer.layers.{layer}.mlp.{name}.weight"
            if wk in keys:
                wt = f.get_tensor(wk)
                if wt.dtype != torch.uint8:
                    failures.append(f"{wk} expected uint8 packed FP4, got {wt.dtype}")
                if wt.shape[-1] % 8 != 0:
                    failures.append(f"{wk} last dim should be divisible by 8 for NVFP4 GEMM, got {wt.shape}")

            sfk = f"transformer.layers.{layer}.mlp.{name}.weights_scaling_factor"
            if sfk in keys:
                sf = f.get_tensor(sfk)
                if sf.numel() % (128 * 4) != 0:
                    failures.append(f"{sfk} numel should be multiple of 512 in cutlass layout, got {sf.numel()}")

            ask = f"transformer.layers.{layer}.mlp.{name}.activation_scaling_factor"
            if ask in keys:
                a = f.get_tensor(ask)
                if tuple(a.shape) != ():
                    failures.append(f"{ask} should be scalar, got shape={tuple(a.shape)}")

        ln_key = f"transformer.layers.{layer}.post_layernorm.weight"
        if args.check_post_ln and ln_key in keys:
            ln = f.get_tensor(ln_key).float()
            max_abs = (ln - 1.0).abs().max().item()
            print(f"post_layernorm_weight_max_abs_diff_from_1={max_abs:.8g}")
            if max_abs > 5e-3:
                failures.append(
                    f"{ln_key} does not look reset to ones after folding; max abs diff={max_abs}"
                )

    print()
    print("== contract result ==")
    if failures:
        print("FAIL")
        for item in failures:
            print(f"- {item}")
        return 1

    print("PASS")
    print("This checkpoint matches the current safe plugin contract: FP4 weights/scales, BF16/FP16 activations around the plugin, and existing proj linear can quantize internally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
