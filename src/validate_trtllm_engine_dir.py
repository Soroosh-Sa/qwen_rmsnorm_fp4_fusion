#!/usr/bin/env python3
"""Validate that a TensorRT-LLM engine directory really contains serialized engines.

This catches the common mistake where an engine output directory exists only because
metadata files were copied there, but trtllm-build did not produce rank*.engine.
"""
import argparse
from pathlib import Path
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--expected-tp", type=int, default=None)
    args = ap.parse_args()
    d = Path(args.engine_dir)
    if not d.exists():
        print(f"ERROR: engine dir does not exist: {d}")
        return 2
    if not d.is_dir():
        print(f"ERROR: engine path is not a directory: {d}")
        return 2

    files = sorted(p.name for p in d.iterdir())
    engine_files = sorted([p for p in d.iterdir() if p.is_file() and (p.name.endswith('.engine') or p.name.endswith('.plan'))])
    rank_engine_files = sorted([p for p in engine_files if p.name.startswith('rank')])

    print(f"ENGINE_DIR={d}")
    print(f"num_files={len(files)}")
    print("files:")
    for name in files:
        print(f"  {name}")
    print(f"num_engine_or_plan_files={len(engine_files)}")
    for p in engine_files:
        print(f"  engine: {p.name} size={p.stat().st_size}")

    # Most TRT-LLM engine dirs contain config.json plus rank*.engine. Some versions
    # use slightly different names, so accept any .engine/.plan but warn if rank count mismatches.
    if not engine_files:
        print("ERROR: no .engine or .plan files found. This is not a usable TensorRT-LLM engine dir.")
        print("Likely causes:")
        print("  1. trtllm-build did not run or failed before producing engines")
        print("  2. the script served the output directory before validating the build")
        print("  3. this directory contains copied tokenizer/config metadata only")
        return 3

    if args.expected_tp is not None and rank_engine_files and len(rank_engine_files) != args.expected_tp:
        print(f"ERROR: expected {args.expected_tp} rank*.engine files, found {len(rank_engine_files)}")
        return 4

    cfg = d / "config.json"
    if cfg.exists():
        try:
            c = json.loads(cfg.read_text())
            if isinstance(c, dict):
                print("config.json: present")
                for key in ["version", "pretrained_config", "build_config", "mapping"]:
                    if key in c:
                        print(f"  {key}: present")
        except Exception as e:
            print(f"WARNING: could not parse config.json: {e}")
    else:
        print("WARNING: config.json is missing from engine dir")

    print("OK: TensorRT-LLM engine directory looks usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
