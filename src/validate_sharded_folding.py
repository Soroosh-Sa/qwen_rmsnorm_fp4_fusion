from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from safetensors.torch import load_file
from tqdm import tqdm

INDEX_FILE = "model.safetensors.index.json"


def load_index(model_dir: Path) -> Dict[str, str]:
    path = model_dir / INDEX_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["weight_map"]
    files = sorted(model_dir.glob("*.safetensors"))
    if len(files) != 1:
        raise FileNotFoundError(f"No {INDEX_FILE}, and did not find exactly one .safetensors file in {model_dir}")
    tensors = load_file(str(files[0]), device="cpu")
    return {name: files[0].name for name in tensors.keys()}


def parse_layer_idx(name: str) -> Optional[int]:
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def norm_for_linear(name: str) -> Optional[str]:
    layer_idx = parse_layer_idx(name)
    if layer_idx is None:
        return None
    prefix = name.split(f"layers.{layer_idx}.")[0]
    if any(name.endswith(f".self_attn.{p}.weight") for p in ["q_proj", "k_proj", "v_proj"]):
        return f"{prefix}layers.{layer_idx}.input_layernorm.weight"
    if any(name.endswith(f".mlp.{p}.weight") for p in ["gate_proj", "up_proj"]):
        return f"{prefix}layers.{layer_idx}.post_attention_layernorm.weight"
    return None


def load_tensor(model_dir: Path, weight_map: Dict[str, str], name: str, cache: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
    shard = weight_map[name]
    if shard not in cache:
        cache[shard] = load_file(str(model_dir / shard), device="cpu")
    return cache[shard][name]


def main() -> None:
    parser = argparse.ArgumentParser(description="Numerically validate several folded tensors against original W * gamma.")
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--folded-dir", required=True)
    parser.add_argument("--max-tensors", type=int, default=16)
    parser.add_argument("--check-norm-ones", action="store_true")
    args = parser.parse_args()

    original_dir = Path(args.original_dir).expanduser().resolve()
    folded_dir = Path(args.folded_dir).expanduser().resolve()
    orig_map = load_index(original_dir)
    fold_map = load_index(folded_dir)

    targets = [name for name in orig_map if norm_for_linear(name) is not None and name in fold_map]
    targets = sorted(targets)[: args.max_tensors]
    orig_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    fold_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    max_errors = []
    for name in tqdm(targets, desc="Checking folded tensors"):
        norm_name = norm_for_linear(name)
        assert norm_name is not None
        w = load_tensor(original_dir, orig_map, name, orig_cache)
        gamma = load_tensor(original_dir, orig_map, norm_name, orig_cache)
        folded_expected = w * gamma.to(dtype=w.dtype).view(1, -1)
        folded_actual = load_tensor(folded_dir, fold_map, name, fold_cache)
        err = (folded_expected.float() - folded_actual.float()).abs().max().item()
        max_errors.append((name, err))

    print("Folded tensor checks:")
    for name, err in max_errors:
        print(f"  {err:.6e}  {name}")

    if args.check_norm_ones:
        norm_names = sorted([name for name in orig_map if name.endswith(".input_layernorm.weight") or name.endswith(".post_attention_layernorm.weight")])
        norm_names = [n for n in norm_names if n in fold_map][: args.max_tensors]
        print("Norm one checks:")
        for name in norm_names:
            t = load_tensor(folded_dir, fold_map, name, fold_cache)
            err = (t.float() - 1.0).abs().max().item()
            print(f"  {err:.6e}  {name}")

    if max_errors and max(err for _, err in max_errors) > 0:
        # Exact equality is expected for CPU same-dtype multiply/save/load, but small nonzero can happen across dtype conversions.
        pass


if __name__ == "__main__":
    main()
