"""
Quick analysis script for BA-Trees experiments.

Usage (from project root, with .venv activated):

    python src/analyze_results.py

This will:
  - Load src/born_again_dp/results/summary.csv,
  - Print aggregate stats per dataset/method,
  - Show a few basic plots (one window per dataset).
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = ROOT / "src" / "born_again_dp" / "results" / "summary.csv"


def main():
    if not SUMMARY_PATH.exists():
        raise SystemExit(f"summary.csv not found at {SUMMARY_PATH}. Run src/run_experiments.py first.")

    df = pd.read_csv(SUMMARY_PATH)
    print(f"Loaded {len(df)} rows from {SUMMARY_PATH}")

    # Aggregate per dataset/method
    agg = (
        df.groupby(["dataset", "method"])[
            ["depth", "leaves", "cpu_time", "rf_acc", "ba_acc", "rf_ba_agreement"]
        ]
        .mean()
        .reset_index()
    )
    print("\nAverage metrics per dataset/method:")
    print(agg.to_string(index=False))

    # Simple plots per dataset
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        means = (
            sub.groupby("method")[["leaves", "cpu_time", "rf_ba_agreement"]]
            .mean()
            .reset_index()
        )

        # Bar plot: avg #leaves per method
        plt.figure(figsize=(6, 4))
        plt.bar(means["method"], means["leaves"])
        plt.title(f"{dataset} – average #leaves per method")
        plt.ylabel("#leaves")
        plt.tight_layout()
        plt.show()

        # Bar plot: avg CPU time per method (log scale)
        plt.figure(figsize=(6, 4))
        plt.bar(means["method"], means["cpu_time"])
        plt.yscale("log")
        plt.title(f"{dataset} – average CPU time (s) per method (log scale)")
        plt.ylabel("CPU time (s, log)")
        plt.tight_layout()
        plt.show()

        # Scatter: leaves vs RF–BA agreement
        plt.figure(figsize=(6, 4))
        for method in ["dp", "greedy", "beam"]:
            msub = sub[sub["method"] == method]
            plt.scatter(
                msub["leaves"],
                msub["rf_ba_agreement"],
                label=method,
                alpha=0.7,
            )
        plt.title(f"{dataset} – #leaves vs RF–BA agreement")
        plt.xlabel("#leaves")
        plt.ylabel("RF–BA agreement")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

