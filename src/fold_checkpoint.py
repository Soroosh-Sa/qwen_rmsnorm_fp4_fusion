from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer

from config_utils import dtype_from_string, ensure_dirs, load_config
from folding import fold_qwen_rmsnorms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen_small.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    dtype = dtype_from_string(cfg.get("dtype", "bfloat16"))

    model_name = cfg["model_name_or_path"]
    output_dir = Path(cfg["folded_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=cfg.get("device_map", "auto"),
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
    )
    model.eval()

    print("Folding RMSNorm gamma into selected Linear weights...")
    records = fold_qwen_rmsnorms(model, cfg)

    print(f"Saving folded model to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=bool(cfg.get("trust_remote_code", True)))
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"Tokenizer save skipped: {e}")

    rows = [asdict(r) for r in records]
    metrics_dir = Path(cfg.get("metrics_dir", "metrics"))
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / "fold_checkpoint_records.csv"
    json_path = metrics_dir / "fold_checkpoint_records.json"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "model_name_or_path": model_name,
        "folded_output_dir": str(output_dir),
        "dtype": cfg.get("dtype", "bfloat16"),
        "num_folded_linears": len(records),
        "set_norm_weights_to_one": cfg.get("folding", {}).get("set_norm_weights_to_one", True),
    }
    with (metrics_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Folded {len(records)} Linear weights.")
    print(f"Records: {csv_path}")


if __name__ == "__main__":
    main()
