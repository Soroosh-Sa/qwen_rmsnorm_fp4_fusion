#!/usr/bin/env python3
"""Repair metadata of TensorRT-LLM/ModelOpt quantized checkpoints.

Some TensorRT-LLM quantization exports may produce a config.json with
`model_type: qwen`, which current Transformers may not recognize for Qwen2.x
checkpoints. This script copies missing tokenizer/generation files from the
source checkpoint and, by default, restores source HF config identity fields
(model_type/architectures) while preserving quantization fields from the output.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def copy_if_exists(src_dir: Path, out_dir: Path, name: str, overwrite: bool = False) -> bool:
    src = src_dir / name
    dst = out_dir / name
    if not src.exists():
        return False
    if dst.exists() and not overwrite:
        return False
    if src.is_dir():
        if dst.exists() and overwrite:
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Original HF checkpoint used as quantization input")
    ap.add_argument("--output", required=True, help="Quantized TensorRT-LLM checkpoint output directory")
    ap.add_argument("--trust-source-config", action="store_true", default=True,
                    help="Restore model_type/architectures from source config. Default true.")
    ap.add_argument("--no-trust-source-config", action="store_false", dest="trust_source_config")
    ap.add_argument("--check-autoconfig", action="store_true", help="Try transformers.AutoConfig.from_pretrained(output)")
    ap.add_argument("--report", default=None, help="Optional report JSON path")
    args = ap.parse_args()

    src_dir = Path(args.source)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_config_path = src_dir / "config.json"
    out_config_path = out_dir / "config.json"
    if not src_config_path.exists():
        raise FileNotFoundError(f"Missing source config.json: {src_config_path}")
    if not out_config_path.exists():
        raise FileNotFoundError(f"Missing output config.json: {out_config_path}")

    src_cfg = load_json(src_config_path)
    out_cfg = load_json(out_config_path)
    before = {
        "output_model_type": out_cfg.get("model_type"),
        "output_architectures": out_cfg.get("architectures"),
        "source_model_type": src_cfg.get("model_type"),
        "source_architectures": src_cfg.get("architectures"),
    }

    changed = []
    if args.trust_source_config:
        for key in [
            "model_type",
            "architectures",
            "auto_map",
            "tokenizer_class",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "tie_word_embeddings",
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_theta",
            "max_position_embeddings",
            "sliding_window",
            "use_sliding_window",
        ]:
            if key in src_cfg and out_cfg.get(key) != src_cfg.get(key):
                out_cfg[key] = src_cfg[key]
                changed.append(key)

    # Preserve quantization-related metadata from output; if the quantizer wrote it,
    # it remains in out_cfg. Add a small marker for reproducibility.
    out_cfg.setdefault("_rmsnorm_folding_metadata", {})
    out_cfg["_rmsnorm_folding_metadata"].update({
        "metadata_repaired_from_source_config": str(src_config_path),
        "source_model_type": src_cfg.get("model_type"),
        "previous_output_model_type": before["output_model_type"],
    })
    save_json(out_config_path, out_cfg)

    # Copy generation/tokenizer files that TRT-LLM serving expects. Do not overwrite
    # tokenizer files already produced by quantization unless explicitly missing.
    copied = []
    for name in [
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "preprocessor_config.json",
        "chat_template.jinja",
    ]:
        if copy_if_exists(src_dir, out_dir, name, overwrite=False):
            copied.append(name)

    report = {
        "source": str(src_dir),
        "output": str(out_dir),
        "before": before,
        "after": {
            "output_model_type": out_cfg.get("model_type"),
            "output_architectures": out_cfg.get("architectures"),
        },
        "changed_config_keys": changed,
        "copied_files": copied,
        "autoconfig_ok": None,
        "autoconfig_error": None,
    }

    if args.check_autoconfig:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(out_dir, trust_remote_code=True)
            report["autoconfig_ok"] = True
            report["autoconfig_class"] = cfg.__class__.__name__
        except Exception as e:  # noqa: BLE001
            report["autoconfig_ok"] = False
            report["autoconfig_error"] = repr(e)

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(report_path, report)


if __name__ == "__main__":
    main()
