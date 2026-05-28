#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess


def gpu_snapshot():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True)
        rows = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                rows.append({
                    "gpu_index": parts[0],
                    "gpu_name": parts[1],
                    "mem_used_mb": float(parts[2]),
                    "mem_total_mb": float(parts[3]),
                    "gpu_util_percent": float(parts[4]),
                })
        return rows
    except Exception:
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--framework", default="tensorrt-llm")
    p.add_argument("--model", required=True)
    p.add_argument("--quantization", default="unknown")
    p.add_argument("--decode-mode", default="baseline")
    p.add_argument("--context-len", type=int, required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--num-requests", type=int, required=True)
    p.add_argument("--error-message", required=True)
    p.add_argument("--scenario-name", default=os.environ.get("SCENARIO_NAME", "unspecified"))
    p.add_argument("--workload-type", default=os.environ.get("WORKLOAD_TYPE", "unspecified"))
    p.add_argument("--prompt-profile", default=os.environ.get("PROMPT_PROFILE", "synthetic_code_context"))
    p.add_argument("--api-mode", default=os.environ.get("OPENAI_API_MODE", "unknown"))
    p.add_argument("--duration-s", default=os.environ.get("DURATION_S", "0"))
    args = p.parse_args()

    gpus = gpu_snapshot()
    vram_gb = sum(g.get("mem_used_mb", 0.0) for g in gpus) / 1024.0
    gpu_util = None
    utils = [g.get("gpu_util_percent") for g in gpus if "gpu_util_percent" in g]
    if utils:
        gpu_util = sum(utils) / len(utils)

    row = {
        "scenario_name": args.scenario_name,
        "workload_type": args.workload_type,
        "prompt_profile": args.prompt_profile,
        "api_mode": args.api_mode,
        "duration_s_requested": args.duration_s,
        "framework": args.framework,
        "model": args.model,
        "quantization": args.quantization,
        "decode_mode": args.decode_mode,
        "gpu_type": "; ".join(sorted(set(g.get("gpu_name", "unknown") for g in gpus))) if gpus else "unknown",
        "num_gpus": len(gpus),
        "context_len": args.context_len,
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_tokens,
        "target_prompt_tokens": "",
        "prompt_tokens_est": "",
        "prompt_tokens_reported": "",
        "num_requests": args.num_requests,
        "successful_requests": 0,
        "failed_requests": args.num_requests,
        "ttft_mean_ms": "",
        "ttft_p50_ms": "",
        "ttft_p99_ms": "",
        "tps_mean": "",
        "tps_p50": "",
        "tps_p99": "",
        "aggregate_tps": 0.0,
        "total_output_tokens": 0,
        "total_time_s": "",
        "vram_idle_gb": vram_gb,
        "vram_load_gb": vram_gb,
        "kv_cache_growth_gb": 0.0,
        "gpu_util_mean_after": gpu_util if gpu_util is not None else "",
        "runtime_stability": "fail",
        "error_count": args.num_requests,
        "error_messages": args.error_message,
    }

    fieldnames = list(row.keys())
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_header = not os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Appended failure row to {args.output}: {args.error_message}")


if __name__ == "__main__":
    main()
