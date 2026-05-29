#!/usr/bin/env python3
"""Summarize Stage-6 NVFP4 three-way benchmark results.

A = normal/original NVFP4 TensorRT-LLM engine
B = folded-weight NVFP4 base engine, no custom plugin
C = folded-weight NVFP4 plugin engine

The key final comparison is C vs A.  B is an ablation.
"""
import argparse
import csv
import math
from pathlib import Path

KEYS = ["context_len", "concurrency", "max_new_tokens", "num_requests"]
METRICS = [
    "tps_mean",
    "aggregate_tps",
    "ttft_mean_ms",
    "total_output_tokens",
    "successful_requests",
    "failed_requests",
    "vram_load_gb",
    "kv_cache_growth_gb",
]


def _to_float(v):
    if v is None or v == "":
        return math.nan
    try:
        return float(v)
    except Exception:
        return math.nan


def _fmt(v):
    if math.isnan(v):
        return ""
    return f"{v:.6f}"


def _pct(new, old):
    if math.isnan(new) or math.isnan(old) or old == 0:
        return ""
    return f"{(new - old) / old * 100.0:.6f}"


def _read_rows(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def _key(row):
    return tuple(str(row.get(k, "")) for k in KEYS)


def _sort_key(k):
    out = []
    for x in k:
        try:
            out.append(int(x))
        except Exception:
            out.append(x)
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", required=True, help="A: normal/original NVFP4 benchmark CSV")
    ap.add_argument("--folded-base", required=True, help="B: folded NVFP4 base benchmark CSV")
    ap.add_argument("--plugin", required=True, help="C: folded NVFP4 plugin benchmark CSV")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    normal_rows = _read_rows(args.normal)
    folded_rows = _read_rows(args.folded_base)
    plugin_rows = _read_rows(args.plugin)

    normal_by_key = {_key(r): r for r in normal_rows}
    folded_by_key = {_key(r): r for r in folded_rows}
    plugin_by_key = {_key(r): r for r in plugin_rows}

    all_keys = sorted(
        set(normal_by_key) & set(folded_by_key) & set(plugin_by_key),
        key=_sort_key,
    )

    out_rows = []
    for k in all_keys:
        a = normal_by_key[k]
        b = folded_by_key[k]
        c = plugin_by_key[k]
        row = {name: val for name, val in zip(KEYS, k)}
        row["normal_model"] = a.get("model", "")
        row["folded_base_model"] = b.get("model", "")
        row["plugin_model"] = c.get("model", "")
        row["normal_runtime_stability"] = a.get("runtime_stability", "")
        row["folded_base_runtime_stability"] = b.get("runtime_stability", "")
        row["plugin_runtime_stability"] = c.get("runtime_stability", "")

        for m in METRICS:
            av = _to_float(a.get(m))
            bv = _to_float(b.get(m))
            cv = _to_float(c.get(m))
            row[f"normal_{m}"] = _fmt(av)
            row[f"folded_base_{m}"] = _fmt(bv)
            row[f"plugin_{m}"] = _fmt(cv)

            if m in {"tps_mean", "aggregate_tps"}:
                row[f"plugin_vs_normal_{m}_change_pct"] = _pct(cv, av)  # C vs A, final comparison
                row[f"plugin_vs_folded_base_{m}_change_pct"] = _pct(cv, bv)  # C vs B, plugin ablation
                row[f"folded_base_vs_normal_{m}_change_pct"] = _pct(bv, av)  # B vs A, folding ablation
            elif m == "ttft_mean_ms":
                # Lower is better for TTFT. Negative means the numerator is faster.
                row["plugin_vs_normal_ttft_mean_ms_change_pct"] = _pct(cv, av)
                row["plugin_vs_folded_base_ttft_mean_ms_change_pct"] = _pct(cv, bv)
                row["folded_base_vs_normal_ttft_mean_ms_change_pct"] = _pct(bv, av)
        out_rows.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out_rows:
        fieldnames = [*KEYS, "error"]
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow({"error": "no_matching_context_concurrency_rows_across_three_csvs"})
        return

    fieldnames = list(out_rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)


if __name__ == "__main__":
    main()
