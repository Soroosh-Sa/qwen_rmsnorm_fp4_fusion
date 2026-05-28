from __future__ import annotations

import argparse
from transformers import AutoModelForCausalLM

from config_utils import dtype_from_string, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen_small.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dtype = dtype_from_string(cfg.get("dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=dtype,
        device_map=cfg.get("device_map", "auto"),
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
    )

    print(model.__class__)
    print("Number of layers:", len(model.model.layers))
    layer = model.model.layers[0]
    print("Layer 0:")
    print(layer)
    print("\nNamed modules containing norm/proj:")
    for name, module in layer.named_modules():
        if any(k in name for k in ["norm", "proj"]):
            print(name, "=>", module.__class__.__name__)


if __name__ == "__main__":
    main()
