"""
Diagnose why beam search underperforms in BA-Trees experiments.

Usage:
    python src/diagnose_beam_search.py
    python src/diagnose_beam_search.py --summary path/to/summary.csv --outdir path/to/output

This script reads the experiment summary and produces:
  - A fold-level comparison table (beam vs dp/greedy)
  - A textual diagnostic report with likely failure modes
  - Diagnostic plots to inspect relationships between beam gaps and complexity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY = ROOT / "src" / "born_again_dp" / "results" / "summary.csv"
DEFAULT_OUTDIR = ROOT / "src" / "analysis_plots" / "beam_diagnostics"


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return np.nan
    sx = x[valid]
    sy = y[valid]
    if np.isclose(sx.std(ddof=0), 0.0) or np.isclose(sy.std(ddof=0), 0.0):
        return np.nan
    return float(sx.corr(sy))


def build_fold_level_table(df: pd.DataFrame) -> pd.DataFrame:
    idx = ["dataset", "fold"]
    metrics = ["ba_acc", "rf_ba_agreement", "leaves", "depth", "cpu_time"]

    wide = (
        df.pivot_table(index=idx, columns="method", values=metrics, aggfunc="first")
        .sort_index(axis=1)
        .copy()
    )
    wide.columns = [f"{m}_{method}" for m, method in wide.columns]
    wide = wide.reset_index()

    # Core performance gaps
    wide["beam_minus_dp_acc"] = wide["ba_acc_beam"] - wide["ba_acc_dp"]
    wide["beam_minus_greedy_acc"] = wide["ba_acc_beam"] - wide["ba_acc_greedy"]
    wide["beam_minus_dp_agreement"] = (
        wide["rf_ba_agreement_beam"] - wide["rf_ba_agreement_dp"]
    )
    wide["beam_minus_greedy_agreement"] = (
        wide["rf_ba_agreement_beam"] - wide["rf_ba_agreement_greedy"]
    )

    # Complexity deltas
    wide["beam_minus_dp_leaves"] = wide["leaves_beam"] - wide["leaves_dp"]
    wide["beam_minus_greedy_leaves"] = wide["leaves_beam"] - wide["leaves_greedy"]
    wide["beam_minus_dp_depth"] = wide["depth_beam"] - wide["depth_dp"]
    wide["beam_minus_greedy_depth"] = wide["depth_beam"] - wide["depth_greedy"]

    # Ratios are useful to detect relative compression/expansion
    wide["beam_to_dp_leaves_ratio"] = wide["leaves_beam"] / wide["leaves_dp"].replace(0, np.nan)
    wide["beam_to_dp_depth_ratio"] = wide["depth_beam"] / wide["depth_dp"].replace(0, np.nan)
    wide["beam_to_dp_cpu_ratio"] = wide["cpu_time_beam"] / wide["cpu_time_dp"].replace(0, np.nan)
    wide["beam_to_greedy_cpu_ratio"] = (
        wide["cpu_time_beam"] / wide["cpu_time_greedy"].replace(0, np.nan)
    )

    # Failure flags: large drop in BA accuracy relative to baselines
    wide["catastrophic_vs_dp"] = wide["beam_minus_dp_acc"] <= -0.15
    wide["catastrophic_vs_greedy"] = wide["beam_minus_greedy_acc"] <= -0.15

    return wide


def make_report(folds: pd.DataFrame) -> str:
    n = len(folds)
    if n == 0:
        return "No fold-level rows available."

    lines = []
    lines.append("# Beam Search Diagnostic Report")
    lines.append("")
    lines.append(f"Total folds analyzed: **{n}**")
    lines.append("")

    # Overall summary
    mean_gap_dp = folds["beam_minus_dp_acc"].mean()
    mean_gap_gr = folds["beam_minus_greedy_acc"].mean()
    win_dp = (folds["beam_minus_dp_acc"] > 0).mean()
    win_gr = (folds["beam_minus_greedy_acc"] > 0).mean()
    catastrophic = folds["catastrophic_vs_dp"].mean()

    lines.append("## Overall Performance")
    lines.append(f"- Mean BA accuracy gap vs DP: **{mean_gap_dp:+.4f}**")
    lines.append(f"- Mean BA accuracy gap vs Greedy: **{mean_gap_gr:+.4f}**")
    lines.append(f"- Beam win rate vs DP: **{win_dp:.1%}**")
    lines.append(f"- Beam win rate vs Greedy: **{win_gr:.1%}**")
    lines.append(f"- Catastrophic drop rate (<= -0.15 vs DP): **{catastrophic:.1%}**")
    lines.append("")

    # By dataset
    by_ds = (
        folds.groupby("dataset")
        .agg(
            folds=("fold", "count"),
            mean_gap_dp=("beam_minus_dp_acc", "mean"),
            mean_gap_greedy=("beam_minus_greedy_acc", "mean"),
            mean_agree_gap_dp=("beam_minus_dp_agreement", "mean"),
            catastrophic_rate=("catastrophic_vs_dp", "mean"),
            mean_leaves_ratio=("beam_to_dp_leaves_ratio", "mean"),
            mean_depth_ratio=("beam_to_dp_depth_ratio", "mean"),
        )
        .sort_values("mean_gap_dp")
    )

    lines.append("## Dataset-Level Signals (worst to best by gap vs DP)")
    # to_markdown requires optional dependency "tabulate"; fall back if missing.
    try:
        lines.append(by_ds.round(4).to_markdown())
    except ImportError:
        lines.append(by_ds.round(4).to_string())
    lines.append("")

    # Correlations to infer likely causes
    corr_gap_leaves = _safe_corr(folds["beam_minus_dp_acc"], folds["beam_to_dp_leaves_ratio"])
    corr_gap_depth = _safe_corr(folds["beam_minus_dp_acc"], folds["beam_to_dp_depth_ratio"])
    corr_gap_agree = _safe_corr(folds["beam_minus_dp_acc"], folds["beam_minus_dp_agreement"])

    lines.append("## Correlations (fold-level, beam vs DP)")
    lines.append(f"- corr(acc gap, leaves ratio): **{corr_gap_leaves:+.3f}**")
    lines.append(f"- corr(acc gap, depth ratio): **{corr_gap_depth:+.3f}**")
    lines.append(f"- corr(acc gap, agreement gap): **{corr_gap_agree:+.3f}**")
    lines.append("")

    # Heuristic interpretation rules
    lines.append("## Likely Failure Modes")
    if np.isfinite(corr_gap_leaves) and corr_gap_leaves > 0.25:
        lines.append(
            "- Beam tends to lose accuracy when it returns relatively smaller trees, suggesting the score favors early simplification too aggressively."
        )
    if np.isfinite(corr_gap_leaves) and corr_gap_leaves < -0.25:
        lines.append(
            "- Larger beam trees are not translating to better accuracy, suggesting split ranking quality is the bottleneck rather than pure search capacity."
        )
    if np.isfinite(corr_gap_agree) and corr_gap_agree > 0.6:
        lines.append(
            "- Accuracy losses closely track lower RF agreement, indicating beam states are drifting away from faithful RF behavior."
        )
    if catastrophic > 0.2:
        lines.append(
            "- High catastrophic-drop rate indicates instability across folds; fixed beam width/top-k expansion likely fails on harder folds."
        )
    if (by_ds["mean_gap_dp"].std() if len(by_ds) > 1 else 0.0) > 0.08:
        lines.append(
            "- Strong dataset heterogeneity suggests a single beam setting (width=5, top-3 children) is not robust across problem difficulty."
        )
    if lines[-1] == "## Likely Failure Modes":
        lines.append(
            "- No dominant single failure mode detected; collect additional run-time traces (candidate split quality by depth, retained states per iteration) to disambiguate."
        )

    lines.append("")
    lines.append("## Recommended Next Experiments")
    lines.append("- Sweep beam width (e.g., 3, 5, 10, 20) and candidate children per state (3, 5, 10).")
    lines.append("- Compare current state score with alternatives that weight impurity more than split count.")
    lines.append("- Add per-iteration logging of pruned states and best discarded candidate score.")

    return "\n".join(lines)


def save_plots(folds: pd.DataFrame, outdir: Path) -> None:
    # 1) Accuracy gap distribution by dataset
    ds_order = (
        folds.groupby("dataset")["beam_minus_dp_acc"]
        .mean()
        .sort_values()
        .index.tolist()
    )

    plt.figure(figsize=(10, 5))
    data = [folds.loc[folds["dataset"] == d, "beam_minus_dp_acc"].values for d in ds_order]
    plt.boxplot(data, labels=ds_order, vert=True, showmeans=True)
    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Beam - DP BA accuracy")
    plt.title("Beam under/over-performance by dataset")
    plt.tight_layout()
    plt.savefig(outdir / "beam_vs_dp_acc_gap_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2) Gap vs leaves ratio
    plt.figure(figsize=(6, 5))
    plt.scatter(
        folds["beam_to_dp_leaves_ratio"],
        folds["beam_minus_dp_acc"],
        alpha=0.7,
    )
    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.axvline(1.0, color="gray", linewidth=1, linestyle=":")
    plt.xlabel("Beam / DP leaves ratio")
    plt.ylabel("Beam - DP BA accuracy")
    plt.title("Accuracy gap vs tree-size compression")
    plt.tight_layout()
    plt.savefig(outdir / "acc_gap_vs_leaves_ratio.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3) Gap vs agreement gap
    plt.figure(figsize=(6, 5))
    plt.scatter(
        folds["beam_minus_dp_agreement"],
        folds["beam_minus_dp_acc"],
        alpha=0.7,
    )
    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.axvline(0.0, color="gray", linewidth=1, linestyle=":")
    plt.xlabel("Beam - DP RF agreement")
    plt.ylabel("Beam - DP BA accuracy")
    plt.title("Accuracy gap tracks RF-faithfulness loss?")
    plt.tight_layout()
    plt.savefig(outdir / "acc_gap_vs_agreement_gap.png", dpi=150, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose beam-search underperformance.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Path to summary.csv produced by run_experiments.py",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory for diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.summary.exists():
        raise SystemExit(f"summary file not found: {args.summary}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary)
    needed = {"dataset", "fold", "method", "ba_acc", "rf_ba_agreement", "leaves", "depth", "cpu_time"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"summary file missing required columns: {sorted(missing)}")

    methods_present = set(df["method"].unique())
    expected = {"dp", "greedy", "beam"}
    if not expected.issubset(methods_present):
        raise SystemExit(
            f"summary must include methods {sorted(expected)}; got {sorted(methods_present)}"
        )

    folds = build_fold_level_table(df)
    folds_csv = args.outdir / "beam_fold_diagnostics.csv"
    folds.to_csv(folds_csv, index=False)

    report = make_report(folds)
    report_path = args.outdir / "beam_diagnostic_report.md"
    report_path.write_text(report)

    save_plots(folds, args.outdir)

    print(f"Saved fold-level diagnostics: {folds_csv}")
    print(f"Saved diagnostic report: {report_path}")
    print(f"Saved plots in: {args.outdir}")


if __name__ == "__main__":
    main()
