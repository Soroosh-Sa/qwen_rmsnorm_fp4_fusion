#!/usr/bin/env python3
"""Validate a TensorRT-LLM/ModelOpt quantized checkpoint layout.

This catches a common failure mode: re-quantizing into an existing output
folder with a different TP size. In that case stale rank*.safetensors files or
rank tensor shapes can remain and trtllm-serve may fail during weight loading.
"""
import argparse
import json
import os
import re
from pathlib import Path

from safetensors.torch import load_file


def load_config(model_dir: Path) -> dict:
    p = model_dir / "config.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing config.json under {model_dir}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def rank_files(model_dir: Path) -> list[Path]:
    def key(p: Path):
        m = re.match(r"rank(\d+)\.safetensors$", p.name)
        return int(m.group(1)) if m else 10**9
    return sorted(model_dir.glob("rank*.safetensors"), key=key)


def tensor_shape_in_any_rank(model_dir: Path, tensor_name: str):
    for f in rank_files(model_dir):
        tensors = load_file(str(f), device="cpu")
        if tensor_name in tensors:
            return f.name, tuple(tensors[tensor_name].shape), str(tensors[tensor_name].dtype)
    return None, None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="TensorRT-LLM quantized checkpoint directory")
    ap.add_argument("--expected-tp-size", type=int, default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--strict", action="store_true", help="Exit nonzero if any warning is found")
    args = ap.parse_args()

    model_dir = Path(args.model)
    cfg = load_config(model_dir)
    mapping = cfg.get("mapping", {}) or {}
    quant = cfg.get("quantization", {}) or {}
    tp = int(mapping.get("tp_size") or mapping.get("attn_tp_size") or 1)
    world = int(mapping.get("world_size") or tp)
    inter = int(cfg.get("intermediate_size") or 0)
    hidden = int(cfg.get("hidden_size") or 0)
    n_heads = cfg.get("num_attention_heads")
    n_kv = cfg.get("num_key_value_heads")
    files = rank_files(model_dir)
    warnings = []
    errors = []

    if args.expected_tp_size is not None and tp != args.expected_tp_size:
        errors.append(f"config mapping tp_size={tp}, but expected {args.expected_tp_size}")

    if len(files) != world:
        errors.append(f"found {len(files)} rank*.safetensors files, but config mapping world_size={world}")

    # Common Qwen TRT-LLM names after ModelOpt export.
    gate_name = "transformer.layers.0.mlp.gate.weight"
    fc_name = "transformer.layers.0.mlp.fc.weight"
    proj_name = "transformer.layers.0.mlp.proj.weight"
    gate_file, gate_shape, gate_dtype = tensor_shape_in_any_rank(model_dir, gate_name)
    fc_file, fc_shape, fc_dtype = tensor_shape_in_any_rank(model_dir, fc_name)
    proj_file, proj_shape, proj_dtype = tensor_shape_in_any_rank(model_dir, proj_name)

    expected_mlp_rows = inter // tp if inter and tp else None
    if gate_shape and expected_mlp_rows and gate_shape[0] != expected_mlp_rows:
        errors.append(
            f"{gate_name} first dim is {gate_shape[0]} in {gate_file}, but expected intermediate_size/tp_size={inter}/{tp}={expected_mlp_rows}. "
            "This usually means the output directory contains stale rank shards from a different TP size."
        )
    if fc_shape and expected_mlp_rows and fc_shape[0] != expected_mlp_rows:
        errors.append(
            f"{fc_name} first dim is {fc_shape[0]} in {fc_file}, but expected intermediate_size/tp_size={inter}/{tp}={expected_mlp_rows}."
        )

    # Packed NVFP4 usually halves the input dimension in uint8 storage.
    expected_packed_hidden = hidden // 2 if hidden else None
    if gate_shape and expected_packed_hidden and len(gate_shape) > 1 and gate_shape[1] != expected_packed_hidden:
        warnings.append(
            f"{gate_name} second dim is {gate_shape[1]}, expected hidden_size/2={expected_packed_hidden} for packed NVFP4 uint8 weights."
        )

    status = "ok" if not errors and not warnings else ("error" if errors else "warning")
    report = {
        "model_dir": str(model_dir),
        "status": status,
        "model_type": cfg.get("model_type"),
        "architecture": cfg.get("architecture"),
        "architectures": cfg.get("architectures"),
        "mapping": mapping,
        "quantization": quant,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "hidden_size": hidden,
        "intermediate_size": inter,
        "rank_files": [f.name for f in files],
        "rank_file_count": len(files),
        "sample_tensors": {
            gate_name: {"file": gate_file, "shape": gate_shape, "dtype": gate_dtype},
            fc_name: {"file": fc_file, "shape": fc_shape, "dtype": fc_dtype},
            proj_name: {"file": proj_file, "shape": proj_shape, "dtype": proj_dtype},
        },
        "warnings": warnings,
        "errors": errors,
    }

    print("TensorRT-LLM quantized checkpoint validation")
    print(f"  model_dir: {model_dir}")
    print(f"  status: {status}")
    print(f"  tp_size/world_size: {tp}/{world}")
    print(f"  rank files: {len(files)} -> {[f.name for f in files]}")
    print(f"  quant_algo: {quant.get('quant_algo')}")
    print(f"  kv_cache_quant_algo: {quant.get('kv_cache_quant_algo')}")
    if gate_shape:
        print(f"  {gate_name}: {gate_shape} {gate_dtype} in {gate_file}")
    if fc_shape:
        print(f"  {fc_name}: {fc_shape} {fc_dtype} in {fc_file}")
    if proj_shape:
        print(f"  {proj_name}: {proj_shape} {proj_dtype} in {proj_file}")
    for e in errors:
        print(f"ERROR: {e}")
    for w in warnings:
        print(f"WARNING: {w}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    if errors or (args.strict and warnings):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
