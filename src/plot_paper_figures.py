"""
Generate the three paper figures:
- Fig 1+3 combined: leaves vs RF-BA agreement and leaves vs CPU time (log), side-by-side with one legend.
- Fig 2: RF accuracy, BA accuracy, RF-BA agreement bar charts.
- Additional informative figures (saved to the same output folder).

Usage (from project root):

    python src/plot_paper_figures.py

Reads: src/born_again_dp/results/summary.csv
Writes: docs/mie424-figs/*.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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


def _legend_method_and_dataset():
    """Legend handles for methods (colors) and datasets (markers)."""
    handles = []
    for method in METHOD_ORDER:
        handles.append(
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
        handles.append(
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
    return handles


def _scatter(ax, df, x_col, y_col):
    """Scatter with method color + dataset marker."""
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
            ax.scatter(sub[x_col], sub[y_col], **kw)


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.8), sharex=True)

    _scatter(ax1, df, "leaves", "rf_ba_agreement")
    _scatter(ax2, df, "leaves", "cpu_time")

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
    fig.legend(
        handles=_legend_method_and_dataset(),
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        borderaxespad=0.5,
    )

    plt.tight_layout(rect=(0.0, 0.0, 0.85, 1.0))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_accuracy_gap(df, out_path):
    """Scatter: RF-BA agreement vs BA accuracy gap (BA acc - RF acc)."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    df = df.copy()
    df["ba_minus_rf"] = df["ba_acc"] - df["rf_acc"]
    _scatter(ax, df, "rf_ba_agreement", "ba_minus_rf")

    ax.axhline(0.0, color="k", linewidth=1, alpha=0.5)
    ax.set_xlabel("RF-BA agreement")
    ax.set_ylabel("BA accuracy − RF accuracy")
    ax.set_title("Agreement vs accuracy gap")
    ax.grid(True, alpha=0.3)

    fig.legend(
        handles=_legend_method_and_dataset(),
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        borderaxespad=0.5,
    )
    plt.tight_layout(rect=(0.0, 0.0, 0.85, 1.0))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_efficiency_frontier(df, out_path):
    """Scatter: RF-BA agreement vs CPU time (log)."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    _scatter(ax, df, "cpu_time", "rf_ba_agreement")

    ax.set_xscale("log")
    ax.set_xlabel("CPU time (seconds, log scale)")
    ax.set_ylabel("RF-BA agreement")
    ax.set_title("Fidelity vs runtime")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, which="both")

    fig.legend(
        handles=_legend_method_and_dataset(),
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        borderaxespad=0.5,
    )
    plt.tight_layout(rect=(0.0, 0.0, 0.85, 1.0))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_min_leaves_for_threshold(df, out_path, threshold=0.95):
    """Bar chart: for each dataset+method, minimum leaves among folds meeting agreement threshold."""
    df = df.copy()
    df_good = df[df["rf_ba_agreement"] >= threshold]
    mins = (
        df_good.groupby(["dataset", "method"])["leaves"]
        .min()
        .reset_index()
        .rename(columns={"leaves": "min_leaves"})
    )

    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    methods = ["dp", "greedy", "beam"]
    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, method in enumerate(methods):
        vals = []
        for d in datasets:
            v = mins[(mins["dataset"] == d) & (mins["method"] == method)]["min_leaves"].values
            vals.append(v[0] if len(v) else np.nan)
        ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=method,
            color=METHOD_COLORS[method],
            edgecolor="k",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABELS[d] for d in datasets], rotation=25, ha="right")
    ax.set_ylabel(f"Min #leaves with RF-BA agreement ≥ {threshold:.2f}")
    ax.set_title("Tree size needed for high agreement")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_stability_boxplots(df, out_path):
    """Per-dataset boxplots for leaves / cpu_time / agreement by method (fold variability)."""
    metrics = [
        ("leaves", "#leaves"),
        ("cpu_time", "CPU time (s)"),
        ("rf_ba_agreement", "RF-BA agreement"),
    ]

    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    fig, axes = plt.subplots(len(datasets), 3, figsize=(10.5, 2.0 * len(datasets)), sharey="col")

    if len(datasets) == 1:
        axes = np.array([axes])  # ensure 2D

    for r, dataset in enumerate(datasets):
        sub = df[df["dataset"] == dataset]
        for c, (metric, ylabel) in enumerate(metrics):
            ax = axes[r, c]
            data = [sub[sub["method"] == m][metric].values for m in ["dp", "greedy", "beam"]]
            bp = ax.boxplot(
                data,
                patch_artist=True,
                tick_labels=["dp", "greedy", "beam"],
            )
            for patch, m in zip(bp["boxes"], ["dp", "greedy", "beam"]):
                patch.set_facecolor(METHOD_COLORS[m])
                patch.set_alpha(0.6)
            ax.grid(True, axis="y", alpha=0.25)
            if metric == "cpu_time":
                ax.set_yscale("log")
            if r == 0:
                ax.set_title(ylabel)
            if c == 0:
                ax.set_ylabel(DISPLAY_LABELS.get(dataset, dataset))

    fig.suptitle("Fold-to-fold stability (boxplots)", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pareto_frontier(points):
    """Return Pareto-efficient points for minimizing x and maximizing y."""
    pts = sorted(points, key=lambda t: (t[0], -t[1]))
    frontier = []
    best_y = -1.0
    for x, y in pts:
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


def fig_pareto_frontier(df, out_path):
    """Pareto plot in leaves vs agreement space (highlight non-dominated points per method)."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    # scatter all points lightly
    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        ax.scatter(sub["leaves"], sub["rf_ba_agreement"], color=METHOD_COLORS[method], alpha=0.25, s=25, label=f"{method} (all)")

        pts = list(zip(sub["leaves"].values, sub["rf_ba_agreement"].values))
        frontier = _pareto_frontier(pts)
        if frontier:
            fx, fy = zip(*frontier)
            ax.plot(fx, fy, color=METHOD_COLORS[method], linewidth=2.0, label=f"{method} (Pareto)")

    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("RF-BA agreement")
    ax.set_title("Pareto frontier: size vs fidelity")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    # Keep legend out of the plotting area
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_depth_vs_leaves(df, out_path):
    """Scatter: depth vs leaves to show tree shape differences."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    _scatter(ax, df, "depth", "leaves")
    ax.set_xlabel("Tree depth")
    ax.set_ylabel("Number of leaves")
    ax.set_title("Tree shape: depth vs leaves")
    ax.grid(True, alpha=0.3)
    fig.legend(
        handles=_legend_method_and_dataset(),
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        borderaxespad=0.5,
    )
    plt.tight_layout(rect=(0.0, 0.0, 0.85, 1.0))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_dataset_difficulty(df, out_path):
    """Dataset-level summary: best agreement achieved vs typical size (median leaves) per method."""
    agg = (
        df.groupby(["dataset", "method"])
        .agg(
            best_agreement=("rf_ba_agreement", "max"),
            median_leaves=("leaves", "median"),
        )
        .reset_index()
    )
    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for method in METHOD_ORDER:
        sub = agg[agg["method"] == method].set_index("dataset").reindex(datasets).reset_index()
        ax.scatter(
            sub["median_leaves"],
            sub["best_agreement"],
            color=METHOD_COLORS[method],
            s=80,
            label=method,
            edgecolors="k",
            linewidths=0.5,
            alpha=0.9,
        )
        for _, row in sub.iterrows():
            if pd.isna(row["median_leaves"]) or pd.isna(row["best_agreement"]):
                continue
            ax.annotate(
                DISPLAY_LABELS.get(row["dataset"], row["dataset"]),
                (row["median_leaves"], row["best_agreement"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
            )

    ax.set_xlabel("Median #leaves (over folds)")
    ax.set_ylabel("Best RF-BA agreement (over folds)")
    ax.set_title("Dataset difficulty summary")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_failure_rate(df, out_path, threshold=0.6):
    """Bar chart: fraction of folds with agreement below threshold, per dataset and method."""
    df = df.copy()
    df["fail"] = df["rf_ba_agreement"] < threshold
    rate = df.groupby(["dataset", "method"])["fail"].mean().reset_index().rename(columns={"fail": "fail_rate"})

    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, method in enumerate(["dp", "greedy", "beam"]):
        vals = []
        for d in datasets:
            v = rate[(rate["dataset"] == d) & (rate["method"] == method)]["fail_rate"].values
            vals.append(v[0] if len(v) else np.nan)
        ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=method,
            color=METHOD_COLORS[method],
            edgecolor="k",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABELS[d] for d in datasets], rotation=25, ha="right")
    ax.set_ylabel(f"Fraction of folds with RF-BA agreement < {threshold:.2f}")
    ax.set_title("Low-fidelity failure rate by dataset")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_data()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig_scatter_pair(df, FIG_DIR / "fig_scatter.png")
    print(f"Saved {FIG_DIR / 'fig_scatter.png'}")

    fig2_bar_charts(df, FIG_DIR / "fig2.png")
    print(f"Saved {FIG_DIR / 'fig2.png'}")

    # Additional informative figures
    fig_accuracy_gap(df, FIG_DIR / "fig_accuracy_gap.png")
    print(f"Saved {FIG_DIR / 'fig_accuracy_gap.png'}")

    fig_efficiency_frontier(df, FIG_DIR / "fig_efficiency.png")
    print(f"Saved {FIG_DIR / 'fig_efficiency.png'}")

    fig_min_leaves_for_threshold(df, FIG_DIR / "fig_min_leaves_095.png", threshold=0.95)
    print(f"Saved {FIG_DIR / 'fig_min_leaves_095.png'}")

    fig_stability_boxplots(df, FIG_DIR / "fig_stability_boxplots.png")
    print(f"Saved {FIG_DIR / 'fig_stability_boxplots.png'}")

    fig_pareto_frontier(df, FIG_DIR / "fig_pareto.png")
    print(f"Saved {FIG_DIR / 'fig_pareto.png'}")

    fig_depth_vs_leaves(df, FIG_DIR / "fig_depth_vs_leaves.png")
    print(f"Saved {FIG_DIR / 'fig_depth_vs_leaves.png'}")

    fig_dataset_difficulty(df, FIG_DIR / "fig_dataset_difficulty.png")
    print(f"Saved {FIG_DIR / 'fig_dataset_difficulty.png'}")

    fig_failure_rate(df, FIG_DIR / "fig_failure_rate_060.png", threshold=0.60)
    print(f"Saved {FIG_DIR / 'fig_failure_rate_060.png'}")


if __name__ == "__main__":
    main()
