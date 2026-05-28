#!/usr/bin/env python3
"""
Plan safe TensorRT-LLM benchmark cases.

This script does not start/stop a server and does not modify any run scripts.
It estimates which (context_len, concurrency) pairs are safe for the currently
running server configuration.

Safety checks:
  1. Sequence safety:
       context_len + max_new_tokens + safety_tokens <= server_max_seq_len

  1b. Request-token safety:
       context_len + max_new_tokens + safety_tokens <= server_max_num_tokens
       This matters for TensorRT-LLM PyTorch backend because max_seq_len can be
       large while max_num_tokens remains at a smaller default (for example 8192).

  2. Approximate KV-cache safety:
       estimated_kv_gb <= min_free_gpu_gb * kv_memory_fraction

The KV estimate is intentionally conservative and approximate. It is useful for
skipping obviously unsafe cases, but the final source of truth is the actual
TensorRT-LLM server behavior/logs.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

try:
    from transformers import AutoConfig
except Exception:
    AutoConfig = None


def parse_int_list(value: str):
    return [int(x.strip()) for x in value.replace(",", " ").split() if x.strip()]


def get_gpu_memory_gb():
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip().splitlines()
    except Exception:
        return []

    gpus = []
    for line in out:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            name, total_mb, free_mb, used_mb = parts[:4]
            gpus.append(
                {
                    "name": name,
                    "total_gb": float(total_mb) / 1024.0,
                    "free_gb": float(free_mb) / 1024.0,
                    "used_gb": float(used_mb) / 1024.0,
                }
            )
    return gpus


def load_config(model_path_or_id):
    if AutoConfig is None:
        return None
    try:
        return AutoConfig.from_pretrained(model_path_or_id, trust_remote_code=True)
    except Exception:
        return None


def get_attr(cfg, names, default=None):
    for name in names:
        if hasattr(cfg, name):
            value = getattr(cfg, name)
            if value is not None:
                return value
    return default


def estimate_kv_bytes_per_token(cfg, tp_size=1, kv_dtype="bf16"):
    if cfg is None:
        return None

    num_layers = get_attr(cfg, ["num_hidden_layers", "n_layer", "num_layers"])
    hidden_size = get_attr(cfg, ["hidden_size", "n_embd"])
    num_attention_heads = get_attr(cfg, ["num_attention_heads", "n_head"])
    num_kv_heads = get_attr(
        cfg,
        ["num_key_value_heads", "num_kv_heads", "multi_query_group_num"],
        default=num_attention_heads,
    )

    if not all([num_layers, hidden_size, num_attention_heads, num_kv_heads]):
        return None

    head_dim = hidden_size // num_attention_heads

    # With tensor parallelism, KV heads are generally distributed across ranks.
    kv_heads_per_gpu = max(1, math.ceil(num_kv_heads / max(1, tp_size)))

    if kv_dtype.lower() in {"fp8", "int8"}:
        bytes_per_element = 1
    else:
        bytes_per_element = 2  # fp16/bf16

    # K and V per token.
    return num_layers * 2 * kv_heads_per_gpu * head_dim * bytes_per_element


def build_plan(args):
    contexts = parse_int_list(args.contexts)
    concurrencies = parse_int_list(args.concurrency)

    gpus = get_gpu_memory_gb()
    min_free_gb = min((g["free_gb"] for g in gpus), default=None)
    min_total_gb = min((g["total_gb"] for g in gpus), default=None)

    cfg = load_config(args.model)
    kv_bytes_per_token = estimate_kv_bytes_per_token(
        cfg, tp_size=args.tp_size, kv_dtype=args.kv_dtype
    )

    usable_kv_gb = None
    if min_free_gb is not None:
        usable_kv_gb = min_free_gb * args.kv_memory_fraction

    tests = []
    for context in contexts:
        estimated_total_tokens = context + args.max_new_tokens + args.safety_tokens
        seq_safe = estimated_total_tokens <= args.server_max_seq_len
        num_tokens_safe = estimated_total_tokens <= args.server_max_num_tokens

        for conc in concurrencies:
            estimated_kv_gb = None
            if kv_bytes_per_token is not None:
                estimated_kv_gb = (
                    conc * estimated_total_tokens * kv_bytes_per_token
                ) / (1024**3)

            if usable_kv_gb is None or estimated_kv_gb is None:
                memory_safe = True
                memory_reason = "kv estimate unavailable; sequence check only"
            else:
                memory_safe = estimated_kv_gb <= usable_kv_gb
                memory_reason = (
                    f"estimated_kv_gb={estimated_kv_gb:.4f}, "
                    f"usable_kv_gb={usable_kv_gb:.4f}"
                )

            run = bool(seq_safe and num_tokens_safe and memory_safe)
            if not seq_safe:
                reason = (
                    f"total_tokens={estimated_total_tokens} > "
                    f"server_max_seq_len={args.server_max_seq_len}"
                )
            elif not num_tokens_safe:
                reason = (
                    f"total_tokens={estimated_total_tokens} > "
                    f"server_max_num_tokens={args.server_max_num_tokens}"
                )
            elif not memory_safe:
                reason = memory_reason
            else:
                reason = "safe"

            tests.append(
                {
                    "context_len": context,
                    "concurrency": conc,
                    "max_new_tokens": args.max_new_tokens,
                    "safety_tokens": args.safety_tokens,
                    "estimated_total_tokens": estimated_total_tokens,
                    "server_max_seq_len": args.server_max_seq_len,
                    "server_max_num_tokens": args.server_max_num_tokens,
                    "seq_safe": seq_safe,
                    "num_tokens_safe": num_tokens_safe,
                    "estimated_kv_gb": None
                    if estimated_kv_gb is None
                    else round(estimated_kv_gb, 4),
                    "usable_kv_gb": None
                    if usable_kv_gb is None
                    else round(usable_kv_gb, 4),
                    "memory_safe": memory_safe,
                    "run": run,
                    "reason": reason,
                }
            )

    return {
        "gpu_summary": gpus,
        "min_total_gb": min_total_gb,
        "min_free_gb": min_free_gb,
        "model": args.model,
        "tp_size": args.tp_size,
        "kv_dtype": args.kv_dtype,
        "kv_bytes_per_token": kv_bytes_per_token,
        "kv_memory_fraction": args.kv_memory_fraction,
        "usable_kv_gb": usable_kv_gb,
        "server_max_seq_len": args.server_max_seq_len,
        "server_max_num_tokens": args.server_max_num_tokens,
        "max_new_tokens": args.max_new_tokens,
        "safety_tokens": args.safety_tokens,
        "tests": tests,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local model path")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--server-max-seq-len", type=int, default=512)
    parser.add_argument("--server-max-num-tokens", type=int, default=None, help="TensorRT-LLM max_num_tokens; defaults to server-max-seq-len")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--kv-dtype", default="bf16", choices=["bf16", "fp16", "fp8", "int8"])
    parser.add_argument("--safety-tokens", type=int, default=64)
    parser.add_argument("--kv-memory-fraction", type=float, default=0.20)
    parser.add_argument("--contexts", default="128,256,512,1024,2048,4096,8192")
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--format",
        choices=["json", "tsv", "summary"],
        default="json",
        help="json prints full plan; tsv prints runnable context/concurrency rows; summary prints human-readable plan.",
    )
    args = parser.parse_args()
    if args.server_max_num_tokens is None:
        args.server_max_num_tokens = args.server_max_seq_len

    plan = build_plan(args)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(plan, f, indent=2)

    if args.format == "json":
        print(json.dumps(plan, indent=2))
    elif args.format == "tsv":
        for test in plan["tests"]:
            if test["run"]:
                print(
                    f"{test['context_len']}\t{test['concurrency']}\t"
                    f"{test['estimated_total_tokens']}\t"
                    f"{test['estimated_kv_gb'] if test['estimated_kv_gb'] is not None else 'NA'}"
                )
    else:
        print("GPU summary:")
        for gpu in plan["gpu_summary"]:
            print(
                f"  {gpu['name']}: total={gpu['total_gb']:.2f} GB, "
                f"free={gpu['free_gb']:.2f} GB, used={gpu['used_gb']:.2f} GB"
            )
        print(f"Model for planning: {plan['model']}")
        print(f"Server max seq len: {plan['server_max_seq_len']}")
        print(f"Server max num tokens: {plan['server_max_num_tokens']}")
        print(f"Max new tokens: {plan['max_new_tokens']}")
        print(f"Safety tokens: {plan['safety_tokens']}")
        print()
        for test in plan["tests"]:
            status = "RUN" if test["run"] else "SKIP"
            print(
                f"{status}: context={test['context_len']}, "
                f"concurrency={test['concurrency']}, "
                f"total_tokens={test['estimated_total_tokens']}, "
                f"reason={test['reason']}"
            )


if __name__ == "__main__":
    main()
