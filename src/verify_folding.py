from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from config_utils import dtype_from_string, ensure_dirs, load_config
from folding import direct_rmsnorm_linear_check


def get_eps(norm_module) -> float:
    for attr in ["variance_epsilon", "eps"]:
        if hasattr(norm_module, attr):
            return float(getattr(norm_module, attr))
    return 1e-6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen_small.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    dtype = dtype_from_string(cfg.get("dtype", "bfloat16"))

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=dtype,
        device_map=cfg.get("device_map", "auto"),
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
    )
    model.eval()

    verify_cfg = cfg.get("verify", {})
    batch_size = int(verify_cfg.get("batch_size", 2))
    seq_len = int(verify_cfg.get("seq_len", 16))
    num_layers = int(verify_cfg.get("num_layers_to_check", 4))

    rows = []
    folding_cfg = cfg.get("folding", {})
    attn_names = folding_cfg.get("attention_linears", ["q_proj", "k_proj", "v_proj"])
    mlp_names = folding_cfg.get("mlp_linears", ["gate_proj", "up_proj"])

    for layer_idx, layer in enumerate(model.model.layers[:num_layers]):
        hidden_size = layer.input_layernorm.weight.numel()
        device = layer.input_layernorm.weight.device
        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

        # input_layernorm -> attention q/k/v
        eps = get_eps(layer.input_layernorm)
        gamma = layer.input_layernorm.weight.detach()
        for name in attn_names:
            if hasattr(layer.self_attn, name):
                linear = getattr(layer.self_attn, name)
                stats = direct_rmsnorm_linear_check(x, gamma, linear, eps)
                rows.append({
                    "layer_idx": layer_idx,
                    "norm": "input_layernorm",
                    "linear": f"self_attn.{name}",
                    "eps": eps,
                    **stats,
                })

        # post_attention_layernorm -> mlp gate/up
        eps = get_eps(layer.post_attention_layernorm)
        gamma = layer.post_attention_layernorm.weight.detach()
        for name in mlp_names:
            if hasattr(layer.mlp, name):
                linear = getattr(layer.mlp, name)
                stats = direct_rmsnorm_linear_check(x, gamma, linear, eps)
                rows.append({
                    "layer_idx": layer_idx,
                    "norm": "post_attention_layernorm",
                    "linear": f"mlp.{name}",
                    "eps": eps,
                    **stats,
                })

    metrics_dir = Path(cfg.get("metrics_dir", "metrics"))
    csv_path = metrics_dir / "folding_layer_checks.csv"
    json_path = metrics_dir / "folding_layer_checks.json"

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

    print(f"Wrote {len(rows)} layer checks to {csv_path}")
    if rows:
        max_abs = max(r["max_abs_error"] for r in rows)
        max_rel = max(r["max_rel_error"] for r in rows)
        print(f"max_abs_error={max_abs:.6e}")
        print(f"max_rel_error={max_rel:.6e}")


if __name__ == "__main__":
    main()
