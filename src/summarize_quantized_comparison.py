#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def read_existing(paths):
    frames = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"[WARN] Missing result file: {path}")
            continue
        if path.stat().st_size == 0:
            print(f"[WARN] Empty result file: {path}")
            continue
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No input benchmark CSV files found")
    return pd.concat(frames, ignore_index=True, sort=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", default="results/quantized_comparison_summary.csv")
    p.add_argument("--pivot-output", default="results/quantized_comparison_pivot.csv")
    args = p.parse_args()

    df = read_existing(args.inputs)
    for col, default in [
        ("quantization", "unknown"),
        ("decode_mode", "unknown"),
        ("context_len", -1),
        ("concurrency", -1),
        ("runtime_stability", "unknown"),
        ("successful_requests", 0),
        ("failed_requests", 0),
    ]:
        if col not in df.columns:
            df[col] = default

    keep = [
        "model", "quantization", "decode_mode", "framework", "gpu_type", "num_gpus",
        "context_len", "concurrency", "max_new_tokens", "num_requests",
        "successful_requests", "failed_requests", "runtime_stability", "error_count", "error_messages",
        "ttft_mean_ms", "ttft_p50_ms", "ttft_p99_ms",
        "tps_mean", "tps_p50", "tps_p99", "aggregate_tps",
        "vram_idle_gb", "vram_load_gb", "kv_cache_growth_gb", "gpu_util_mean_after",
        "source_file",
    ]
    keep = [c for c in keep if c in df.columns]
    summary = df[keep].copy()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print("Saved summary:", args.output)
    print(summary.to_string(index=False))

    metric_cols = [c for c in ["ttft_mean_ms", "ttft_p99_ms", "tps_mean", "tps_p99", "aggregate_tps", "vram_load_gb", "kv_cache_growth_gb"] if c in df.columns]
    if metric_cols:
        group_cols = ["context_len", "concurrency", "quantization"]
        pivot = df.groupby(group_cols, dropna=False)[metric_cols].mean(numeric_only=True).reset_index()
        pivot.to_csv(args.pivot_output, index=False)
        print("Saved pivot:", args.pivot_output)
        print(pivot.to_string(index=False))

    # Compact pass/fail coverage by quantization target.
    cov = df.groupby("quantization", dropna=False).agg(
        rows=("quantization", "size"),
        passing=("runtime_stability", lambda s: int((s.astype(str) == "pass").sum())),
        total_successful_requests=("successful_requests", "sum"),
        total_failed_requests=("failed_requests", "sum"),
    ).reset_index()
    cov_path = Path(args.output).with_name("quantized_comparison_coverage.csv")
    cov.to_csv(cov_path, index=False)
    print("Saved coverage:", cov_path)
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
