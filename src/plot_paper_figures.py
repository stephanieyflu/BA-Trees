"""
Generate the three paper figures:
- Fig 1+3 combined: leaves vs RF-BA agreement and leaves vs CPU time (log), side-by-side with one legend.
- Fig 2: RF accuracy, BA accuracy, RF-BA agreement bar charts.

Usage (from project root):

    python src/plot_paper_figures.py

Reads: src/born_again_dp/results/summary.csv
Writes: docs/mie424-figs/fig_scatter.png, fig2.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = ROOT / "src" / "born_again_dp" / "results" / "summary.csv"
FIG_DIR = ROOT / "docs" / "mie424-figs"

# Dataset order for bar charts and consistent labeling
DATASET_ORDER = [
    "Breast-Cancer-Wisconsin",
    "COMPAS-ProPublica",
    "FICO",
    "HTRU2",
    "Pima-Diabetes",
    "Seeds",
]

# Short display labels for x-axes (keep full names in text/tables)
DISPLAY_LABELS = {
    "Breast-Cancer-Wisconsin": "Breast",
    "COMPAS-ProPublica": "COMPAS",
    "FICO": "FICO",
    "HTRU2": "HTRU2",
    "Pima-Diabetes": "Pima",
    "Seeds": "Seeds",
}

# Method order and colors (match paper: dp=blue, greedy=orange, beam=green)
METHOD_ORDER = ["dp", "greedy", "beam"]
METHOD_COLORS = {"dp": "#1f77b4", "greedy": "#ff7f0e", "beam": "#2ca02c"}

# Marker per dataset for scatter plots
DATASET_MARKERS = {
    "Breast-Cancer-Wisconsin": "o",
    "COMPAS-ProPublica": "x",
    "FICO": "s",
    "HTRU2": "P",  # plus
    "Pima-Diabetes": "*",
    "Seeds": "^",
}


def load_data():
    if not SUMMARY_PATH.exists():
        raise SystemExit(
            f"summary.csv not found at {SUMMARY_PATH}. Run src/run_experiments.py first."
        )
    df = pd.read_csv(SUMMARY_PATH)
    return df


def fig2_bar_charts(df, out_path):
    """Three side-by-side bar charts: RF accuracy, BA-tree accuracy, RF-BA agreement.
    X = dataset, grouped bars = method (beam, dp, greedy). Uses mean over folds."""
    agg = (
        df.groupby(["dataset", "method"])[["rf_acc", "ba_acc", "rf_ba_agreement"]]
        .mean()
        .reset_index()
    )

    # Pivot so we have dataset x method for each metric
    datasets = [d for d in DATASET_ORDER if d in agg["dataset"].values]
    x = np.arange(len(datasets))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(9, 4.6), sharey=True)

    for ax, (metric, ylabel) in zip(
        axes,
        [
            ("rf_acc", "RF accuracy"),
            ("ba_acc", "BA-tree accuracy"),
            ("rf_ba_agreement", "RF-BA agreement"),
        ],
    ):
        for i, method in enumerate(["beam", "dp", "greedy"]):
            vals = [
                agg[(agg["dataset"] == d) & (agg["method"] == method)][metric].values
                for d in datasets
            ]
            vals = [v[0] if len(v) else np.nan for v in vals]
            offset = (i - 1) * width
            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=method,
                color=METHOD_COLORS[method],
                edgecolor="k",
                linewidth=0.5,
            )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [DISPLAY_LABELS[d] for d in datasets],
            rotation=25,
            ha="right",
        )
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title("RF accuracy")
    axes[1].set_title("BA-tree accuracy")
    axes[2].set_title("RF-BA agreement")
    fig.suptitle("RF accuracy, BA-tree accuracy, and RF-BA agreement", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_scatter_pair(df, out_path):
    """Side-by-side scatters: (left) leaves vs RF-BA agreement, (right) leaves vs CPU time (log), shared legend."""
    from matplotlib.lines import Line2D

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.8), sharex=True)

    for method in METHOD_ORDER:
        for dataset in DATASET_ORDER:
            sub = df[(df["method"] == method) & (df["dataset"] == dataset)]
            if sub.empty:
                continue
            kw = dict(
                c=METHOD_COLORS[method],
                marker=DATASET_MARKERS[dataset],
                s=50,
                alpha=0.8,
            )
            if DATASET_MARKERS[dataset] != "x":
                kw["edgecolors"] = "k"
                kw["linewidths"] = 0.5
            ax1.scatter(sub["leaves"], sub["rf_ba_agreement"], **kw)
            ax2.scatter(sub["leaves"], sub["cpu_time"], **kw)

    # Axis labels / titles
    ax1.set_xlabel("Number of leaves")
    ax1.set_ylabel("RF-BA agreement")
    ax1.set_title("Tree size vs RF-BA agreement")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Number of leaves")
    ax2.set_ylabel("CPU time (seconds)")
    ax2.set_yscale("log")
    ax2.set_title("Runtime versus tree size (log scale)")
    ax2.grid(True, alpha=0.3, which="both")

    # Combined legend (methods by color, datasets by marker)
    legend_elements = []
    for method in METHOD_ORDER:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=METHOD_COLORS[method],
                markersize=10,
                label=method,
            )
        )
    for dataset in DATASET_ORDER:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=DATASET_MARKERS[dataset],
                color="w",
                markerfacecolor="gray",
                markeredgecolor="k",
                markersize=8,
                label=DISPLAY_LABELS.get(dataset, dataset),
            )
        )

    fig.legend(
        handles=legend_elements,
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        borderaxespad=0.5,
    )

    plt.tight_layout(rect=(0.0, 0.0, 0.85, 1.0))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_data()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig_scatter_pair(df, FIG_DIR / "fig_scatter.png")
    print(f"Saved {FIG_DIR / 'fig_scatter.png'}")

    fig2_bar_charts(df, FIG_DIR / "fig2.png")
    print(f"Saved {FIG_DIR / 'fig2.png'}")


if __name__ == "__main__":
    main()
