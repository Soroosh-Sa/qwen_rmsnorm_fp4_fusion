from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def main():
    p = argparse.ArgumentParser(description="Create a manifest for the requested quantized-model comparison.")
    p.add_argument("--out", default="metrics/comparison_manifest.json")
    p.add_argument("--csv", default="metrics/comparison_targets.csv")
    args = p.parse_args()

    rows = [
        {
            "name": "prequantized_nvfp4",
            "role": "baseline",
            "precision": "nvfp4",
            "is_true_fp4": True,
            "folded": False,
            "runtime": "TensorRT-LLM",
            "checkpoint_or_engine": env("PREQUANTIZED_NVFP4_DIR", "TODO"),
            "notes": "Use this instead of original BF16 for speed baseline.",
        },
        {
            "name": "folded_gptq_int4",
            "role": "comparison",
            "precision": "gptq_int4",
            "is_true_fp4": False,
            "folded": True,
            "runtime": env("GPTQ_RUNTIME", "vLLM/AutoGPTQ/GPTQModel/TBD"),
            "checkpoint_or_engine": env("FOLDED_GPTQ_INT4_DIR", "TODO"),
            "notes": "INT4 baseline after RMSNorm folding; not FP4.",
        },
        {
            "name": "folded_trtllm_nvfp4",
            "role": "main_target",
            "precision": "nvfp4",
            "is_true_fp4": True,
            "folded": True,
            "runtime": "TensorRT-LLM",
            "checkpoint_or_engine": env("FOLDED_TRTLLM_NVFP4_DIR", "TODO"),
            "notes": "Main Blackwell FP4 path after RMSNorm folding.",
        },
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"comparison_targets": rows}, indent=2), encoding="utf-8")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
