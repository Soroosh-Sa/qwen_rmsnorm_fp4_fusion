#!/usr/bin/env python3
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


def _read_rows(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def _key(row):
    return tuple(str(row.get(k, "")) for k in KEYS)


def _pct(plugin, base):
    if math.isnan(plugin) or math.isnan(base) or base == 0:
        return ""
    return f"{(plugin - base) / base * 100.0:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--plugin", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    base_rows = _read_rows(args.base)
    plugin_rows = _read_rows(args.plugin)
    base_by_key = {_key(r): r for r in base_rows}
    plugin_by_key = {_key(r): r for r in plugin_rows}
    keys = sorted(set(base_by_key) & set(plugin_by_key), key=lambda k: tuple(int(x) if str(x).isdigit() else x for x in k))

    out_rows = []
    for k in keys:
        b = base_by_key[k]
        p = plugin_by_key[k]
        row = {name: val for name, val in zip(KEYS, k)}
        row["base_model"] = b.get("model", "")
        row["plugin_model"] = p.get("model", "")
        row["base_runtime_stability"] = b.get("runtime_stability", "")
        row["plugin_runtime_stability"] = p.get("runtime_stability", "")
        for m in METRICS:
            bv = _to_float(b.get(m))
            pv = _to_float(p.get(m))
            row[f"base_{m}"] = "" if math.isnan(bv) else f"{bv:.6f}"
            row[f"plugin_{m}"] = "" if math.isnan(pv) else f"{pv:.6f}"
            if m in {"tps_mean", "aggregate_tps"}:
                row[f"plugin_{m}_change_pct"] = _pct(pv, bv)
            if m == "ttft_mean_ms":
                # Lower is better for TTFT, so negative means plugin is faster.
                row["plugin_ttft_mean_ms_change_pct"] = _pct(pv, bv)
        out_rows.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out_rows:
        # Still write a small diagnostic CSV.
        fieldnames = [*KEYS, "error"]
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow({"error": "no_matching_context_concurrency_rows"})
        return

    fieldnames = list(out_rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)


if __name__ == "__main__":
    main()
