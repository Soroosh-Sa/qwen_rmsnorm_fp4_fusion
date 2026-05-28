from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


ATTN_LINEARS_DEFAULT = ("q_proj", "k_proj", "v_proj")
MLP_LINEARS_DEFAULT = ("gate_proj", "up_proj")
INDEX_FILE = "model.safetensors.index.json"


@dataclass
class FoldShardRecord:
    shard: str
    tensor_name: str
    layer_idx: int
    norm_tensor_name: str
    linear_kind: str
    shape: str
    dtype: str
    max_abs_before: float
    max_abs_gamma: float
    max_abs_after: float
    action: str


@dataclass
class NormRewriteRecord:
    shard: str
    tensor_name: str
    layer_idx: int
    shape: str
    dtype: str
    action: str


def read_index(model_dir: Path) -> Tuple[Dict[str, str], Dict]:
    index_path = model_dir / INDEX_FILE
    if not index_path.exists():
        raise FileNotFoundError(
            f"Could not find {index_path}. This script expects a sharded safetensors HF checkpoint. "
            "For a single-file safetensors checkpoint, use --allow-single-file."
        )
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map", {})
    if not weight_map:
        raise ValueError(f"{index_path} does not contain a non-empty weight_map")
    return weight_map, data


def find_single_safetensors(model_dir: Path) -> Optional[Path]:
    files = sorted(model_dir.glob("*.safetensors"))
    if len(files) == 1:
        return files[0]
    return None


def build_weight_map_for_single_file(model_dir: Path, single_file: Path) -> Tuple[Dict[str, str], Dict]:
    tensors = load_file(str(single_file), device="cpu")
    weight_map = {name: single_file.name for name in tensors.keys()}
    index_data = {"metadata": {"format": "pt"}, "weight_map": weight_map}
    return weight_map, index_data


def parse_layer_idx(tensor_name: str) -> Optional[int]:
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", tensor_name)
    if not m:
        return None
    return int(m.group(1))


def is_attention_linear_weight(name: str, attn_linears: Iterable[str]) -> Optional[str]:
    for proj in attn_linears:
        if name.endswith(f".self_attn.{proj}.weight"):
            return proj
    return None


def is_mlp_linear_weight(name: str, mlp_linears: Iterable[str]) -> Optional[str]:
    for proj in mlp_linears:
        if name.endswith(f".mlp.{proj}.weight"):
            return proj
    return None


def norm_name_for_linear(name: str, attn_linears: Iterable[str], mlp_linears: Iterable[str]) -> Optional[str]:
    layer_idx = parse_layer_idx(name)
    if layer_idx is None:
        return None
    if is_attention_linear_weight(name, attn_linears) is not None:
        prefix = name.split(f"layers.{layer_idx}.")[0]
        return f"{prefix}layers.{layer_idx}.input_layernorm.weight"
    if is_mlp_linear_weight(name, mlp_linears) is not None:
        prefix = name.split(f"layers.{layer_idx}.")[0]
        return f"{prefix}layers.{layer_idx}.post_attention_layernorm.weight"
    return None


def is_target_norm(name: str) -> bool:
    return name.endswith(".input_layernorm.weight") or name.endswith(".post_attention_layernorm.weight")


def copy_sidecar_files(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    skip_suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    skip_names = {INDEX_FILE}
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in input_dir.iterdir():
        if src.name in skip_names:
            continue
        if src.is_file() and src.suffix not in skip_suffixes:
            dst = output_dir / src.name
            if dst.exists() and not overwrite:
                continue
            shutil.copy2(src, dst)
        elif src.is_dir() and src.name not in {".git", "__pycache__"}:
            dst = output_dir / src.name
            if dst.exists() and overwrite:
                shutil.rmtree(dst)
            if not dst.exists():
                shutil.copytree(src, dst)


def load_all_norm_gammas(model_dir: Path, weight_map: Dict[str, str]) -> Dict[str, torch.Tensor]:
    norm_names = sorted([name for name in weight_map if is_target_norm(name)])
    shard_to_norms: Dict[str, List[str]] = {}
    for name in norm_names:
        shard_to_norms.setdefault(weight_map[name], []).append(name)

    gammas: Dict[str, torch.Tensor] = {}
    for shard_name, names in tqdm(shard_to_norms.items(), desc="Loading RMSNorm gamma tensors"):
        shard_path = model_dir / shard_name
        tensors = load_file(str(shard_path), device="cpu")
        for name in names:
            if name not in tensors:
                raise KeyError(f"Index expected {name} in {shard_path}, but it was not found")
            gammas[name] = tensors[name].detach().clone().cpu()
    return gammas


def fold_weight_tensor(weight: torch.Tensor, gamma: torch.Tensor, tensor_name: str, norm_name: str) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D linear weight for {tensor_name}, got shape {tuple(weight.shape)}")
    if gamma.ndim != 1:
        raise ValueError(f"Expected 1D RMSNorm gamma for {norm_name}, got shape {tuple(gamma.shape)}")
    if weight.shape[1] != gamma.shape[0]:
        raise ValueError(
            f"Shape mismatch for {tensor_name}: W shape {tuple(weight.shape)}, gamma {norm_name} shape {tuple(gamma.shape)}. "
            "For PyTorch Linear, expected W.shape[1] == gamma.shape[0]."
        )
    folded = weight * gamma.to(dtype=weight.dtype).view(1, -1)
    return folded


def process_shards(
    input_dir: Path,
    output_dir: Path,
    weight_map: Dict[str, str],
    index_data: Dict,
    gammas: Dict[str, torch.Tensor],
    attn_linears: Iterable[str],
    mlp_linears: Iterable[str],
    set_norms_to_one: bool,
    dry_run: bool,
    overwrite: bool,
    max_shards: Optional[int],
) -> Tuple[List[FoldShardRecord], List[NormRewriteRecord]]:
    shards = sorted(set(weight_map.values()))
    if max_shards is not None:
        shards = shards[:max_shards]

    fold_records: List[FoldShardRecord] = []
    norm_records: List[NormRewriteRecord] = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for shard_name in tqdm(shards, desc="Processing checkpoint shards"):
        in_path = input_dir / shard_name
        out_path = output_dir / shard_name
        if out_path.exists() and not overwrite and not dry_run:
            raise FileExistsError(f"Output shard already exists: {out_path}. Use --overwrite to replace it.")

        tensors = load_file(str(in_path), device="cpu")
        modified = False

        for name in list(tensors.keys()):
            norm_name = norm_name_for_linear(name, attn_linears, mlp_linears)
            if norm_name is None:
                continue
            layer_idx = parse_layer_idx(name)
            if layer_idx is None:
                continue
            if norm_name not in gammas:
                raise KeyError(f"Could not find required gamma tensor {norm_name} for {name}")

            gamma = gammas[norm_name]
            weight = tensors[name]
            linear_kind = is_attention_linear_weight(name, attn_linears) or is_mlp_linear_weight(name, mlp_linears) or "unknown"
            before = weight.detach().abs().max().float().item()
            gamma_max = gamma.detach().abs().max().float().item()
            folded = fold_weight_tensor(weight, gamma, name, norm_name)
            after = folded.detach().abs().max().float().item()

            fold_records.append(
                FoldShardRecord(
                    shard=shard_name,
                    tensor_name=name,
                    layer_idx=layer_idx,
                    norm_tensor_name=norm_name,
                    linear_kind=linear_kind,
                    shape=str(tuple(weight.shape)),
                    dtype=str(weight.dtype),
                    max_abs_before=before,
                    max_abs_gamma=gamma_max,
                    max_abs_after=after,
                    action="dry_run_fold" if dry_run else "folded",
                )
            )

            if not dry_run:
                tensors[name] = folded
                modified = True

        if set_norms_to_one:
            for name in list(tensors.keys()):
                if not is_target_norm(name):
                    continue
                layer_idx = parse_layer_idx(name)
                if layer_idx is None:
                    continue
                norm_records.append(
                    NormRewriteRecord(
                        shard=shard_name,
                        tensor_name=name,
                        layer_idx=layer_idx,
                        shape=str(tuple(tensors[name].shape)),
                        dtype=str(tensors[name].dtype),
                        action="dry_run_set_to_one" if dry_run else "set_to_one",
                    )
                )
                if not dry_run:
                    tensors[name] = torch.ones_like(tensors[name])
                    modified = True

        if not dry_run:
            # Save every processed shard, not only modified ones, so output checkpoint is complete.
            save_file(tensors, str(out_path), metadata={"format": "pt"})

        del tensors

    if not dry_run:
        new_index = dict(index_data)
        new_index["weight_map"] = dict(weight_map)
        (output_dir / INDEX_FILE).write_text(json.dumps(new_index, indent=2, sort_keys=True), encoding="utf-8")

    return fold_records, norm_records


def write_records(records: List[object], path_csv: Path, path_json: Path) -> None:
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in records]
    with path_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    if rows:
        with path_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        path_csv.write_text("", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fold Qwen RMSNorm gamma into Linear weights shard-by-shard.")
    parser.add_argument("--input-dir", required=True, help="Original HF checkpoint directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for folded checkpoint")
    parser.add_argument("--metrics-dir", default="metrics", help="Where to save folding CSV/JSON reports")
    parser.add_argument("--attention-linears", default=",".join(ATTN_LINEARS_DEFAULT))
    parser.add_argument("--mlp-linears", default=",".join(MLP_LINEARS_DEFAULT))
    parser.add_argument("--no-set-norms-to-one", action="store_true", help="Do not rewrite RMSNorm gamma vectors to ones")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report what would be modified, without writing shards")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory/shards")
    parser.add_argument("--max-shards", type=int, default=None, help="Process only the first N shards for debugging")
    parser.add_argument("--allow-single-file", action="store_true", help="Allow checkpoints with exactly one .safetensors file and no index JSON")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    metrics_dir = Path(args.metrics_dir).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}. Use --overwrite.")
    if output_dir.exists() and args.overwrite and not args.dry_run:
        # Keep directory but let per-file writes replace files; remove stale index/sidecars from earlier runs.
        pass

    attn_linears = tuple(x.strip() for x in args.attention_linears.split(",") if x.strip())
    mlp_linears = tuple(x.strip() for x in args.mlp_linears.split(",") if x.strip())

    try:
        weight_map, index_data = read_index(input_dir)
    except FileNotFoundError:
        if not args.allow_single_file:
            raise
        single_file = find_single_safetensors(input_dir)
        if single_file is None:
            raise FileNotFoundError("No index JSON found and did not find exactly one .safetensors file.")
        print(f"No index found. Using single-file checkpoint: {single_file.name}")
        weight_map, index_data = build_weight_map_for_single_file(input_dir, single_file)

    print(f"Input checkpoint: {input_dir}")
    print(f"Output checkpoint: {output_dir}")
    print(f"Total tensors in index: {len(weight_map)}")
    print(f"Total shards: {len(set(weight_map.values()))}")
    print(f"Attention linears: {attn_linears}")
    print(f"MLP linears: {mlp_linears}")
    print(f"Set RMSNorm weights to one: {not args.no_set_norms_to_one}")
    print(f"Dry run: {args.dry_run}")

    if not args.dry_run:
        copy_sidecar_files(input_dir, output_dir, overwrite=args.overwrite)

    gammas = load_all_norm_gammas(input_dir, weight_map)
    print(f"Loaded {len(gammas)} RMSNorm gamma tensors")

    fold_records, norm_records = process_shards(
        input_dir=input_dir,
        output_dir=output_dir,
        weight_map=weight_map,
        index_data=index_data,
        gammas=gammas,
        attn_linears=attn_linears,
        mlp_linears=mlp_linears,
        set_norms_to_one=not args.no_set_norms_to_one,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        max_shards=args.max_shards,
    )

    suffix = "dry_run" if args.dry_run else "folded"
    write_records(
        fold_records,
        metrics_dir / f"sharded_fold_records_{suffix}.csv",
        metrics_dir / f"sharded_fold_records_{suffix}.json",
    )
    write_records(
        norm_records,
        metrics_dir / f"sharded_norm_rewrite_records_{suffix}.csv",
        metrics_dir / f"sharded_norm_rewrite_records_{suffix}.json",
    )

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_index_tensors": len(weight_map),
        "num_shards": len(set(weight_map.values())),
        "num_folded_linear_tensors": len(fold_records),
        "num_rewritten_norm_tensors": len(norm_records),
        "attention_linears": list(attn_linears),
        "mlp_linears": list(mlp_linears),
        "set_norms_to_one": not args.no_set_norms_to_one,
        "dry_run": args.dry_run,
        "max_shards": args.max_shards,
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"sharded_fold_manifest_{suffix}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Folded Linear tensors: {len(fold_records)}")
    print(f"Rewritten RMSNorm tensors: {len(norm_records)}")
    print(f"Metrics written under: {metrics_dir}")


if __name__ == "__main__":
    main()
