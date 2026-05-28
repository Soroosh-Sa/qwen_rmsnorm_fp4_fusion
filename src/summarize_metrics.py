from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    metrics = Path("metrics")
    print("Available metric files:")
    for p in sorted(metrics.glob("*")):
        print(" -", p)

    csv_path = metrics / "folding_layer_checks.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print("\nFolding layer check summary:")
        print(df[["max_abs_error", "mean_abs_error", "max_rel_error", "mean_rel_error"]].describe())

    json_path = metrics / "model_logits_check.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        print("\nModel logits checks:")
        for row in data:
            print(row)


if __name__ == "__main__":
    main()
