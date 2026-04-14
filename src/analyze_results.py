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
OUTPUT_DIR = ROOT / "src" / "analysis_plots"


def main():
    if not SUMMARY_PATH.exists():
        raise SystemExit(f"summary.csv not found at {SUMMARY_PATH}. Run src/run_experiments.py first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    agg_path = OUTPUT_DIR / "aggregate_metrics.csv"
    agg.to_csv(agg_path, index=False)
    print(f"\nSaved aggregate metrics to: {agg_path}")

    # Simple plots per dataset
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        means = (
            sub.groupby("method")[["leaves", "cpu_time", "rf_ba_agreement"]]
            .mean()
            .reset_index()
        )
        dataset_slug = dataset.lower().replace(" ", "_").replace("-", "_")
        means_path = OUTPUT_DIR / f"{dataset_slug}_method_means.csv"
        means.to_csv(means_path, index=False)
        print(f"Saved method means for {dataset} to: {means_path}")

        # Bar plot: avg #leaves per method
        plt.figure(figsize=(6, 4))
        plt.bar(means["method"], means["leaves"])
        plt.title(f"{dataset} – average #leaves per method")
        plt.ylabel("#leaves")
        plt.tight_layout()
        leaves_plot_path = OUTPUT_DIR / f"{dataset_slug}_avg_leaves.png"
        plt.savefig(leaves_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {leaves_plot_path}")

        # Bar plot: avg CPU time per method (log scale)
        plt.figure(figsize=(6, 4))
        plt.bar(means["method"], means["cpu_time"])
        plt.yscale("log")
        plt.title(f"{dataset} – average CPU time (s) per method (log scale)")
        plt.ylabel("CPU time (s, log)")
        plt.tight_layout()
        cpu_plot_path = OUTPUT_DIR / f"{dataset_slug}_avg_cpu_time_log.png"
        plt.savefig(cpu_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {cpu_plot_path}")

        # Scatter: leaves vs RF–BA agreement
        plt.figure(figsize=(6, 4))
        for method in ["dp", "greedy", "beam", "beam_lookahead", "beam_balance"]:
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
        scatter_plot_path = OUTPUT_DIR / f"{dataset_slug}_leaves_vs_agreement.png"
        plt.savefig(scatter_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {scatter_plot_path}")


if __name__ == "__main__":
    main()

