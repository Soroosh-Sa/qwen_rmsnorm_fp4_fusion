from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config_utils import dtype_from_string, ensure_dirs, load_config


@torch.no_grad()
def run_logits(model, tokenizer, prompts):
    device = next(iter(model.parameters())).device
    results = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        out = model(**encoded)
        logits = out.logits.detach().float().cpu()
        results.append({
            "prompt": prompt,
            "shape": list(logits.shape),
            "logits": logits,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen_small.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    dtype = dtype_from_string(cfg.get("dtype", "bfloat16"))
    prompts = cfg.get("verify", {}).get("prompts", ["Hello"])

    original_path = cfg["model_name_or_path"]
    folded_path = cfg["folded_output_dir"]

    tokenizer = AutoTokenizer.from_pretrained(original_path, trust_remote_code=bool(cfg.get("trust_remote_code", True)))

    print(f"Loading original model: {original_path}")
    original = AutoModelForCausalLM.from_pretrained(
        original_path,
        torch_dtype=dtype,
        device_map=cfg.get("device_map", "auto"),
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
    ).eval()

    print(f"Loading folded model: {folded_path}")
    folded = AutoModelForCausalLM.from_pretrained(
        folded_path,
        torch_dtype=dtype,
        device_map=cfg.get("device_map", "auto"),
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
    ).eval()

    summary = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        dev1 = next(iter(original.parameters())).device
        enc1 = {k: v.to(dev1) for k, v in enc.items()}
        logits1 = original(**enc1).logits.detach().float().cpu()

        dev2 = next(iter(folded.parameters())).device
        enc2 = {k: v.to(dev2) for k, v in enc.items()}
        logits2 = folded(**enc2).logits.detach().float().cpu()

        diff = logits1 - logits2
        denom = logits1.abs().clamp_min(1e-8)
        summary.append({
            "prompt": prompt,
            "shape": list(logits1.shape),
            "max_abs_error": diff.abs().max().item(),
            "mean_abs_error": diff.abs().mean().item(),
            "max_rel_error": (diff.abs() / denom).max().item(),
            "mean_rel_error": (diff.abs() / denom).mean().item(),
            "top1_original_last_token": int(logits1[0, -1].argmax().item()),
            "top1_folded_last_token": int(logits2[0, -1].argmax().item()),
        })

    metrics_dir = Path(cfg.get("metrics_dir", "metrics"))
    out_path = metrics_dir / "model_logits_check.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote model logits check to {out_path}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
