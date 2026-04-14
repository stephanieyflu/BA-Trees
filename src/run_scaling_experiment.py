"""
Scaling experiment: vary the number of RF trees used by the BA solver
and measure how runtime, tree size, and accuracy scale for each method.

Uses 3 folds (not 10) and a per-run timeout to keep total runtime
manageable. DP is expected to time out at higher tree counts on
harder datasets -- that's the point of the experiment.

Produces: src/born_again_dp/results/scaling/scaling_summary.csv
"""

import csv
import os
import subprocess
import time
from pathlib import Path

import numpy as np

import datasets
import random_forests
import persistence

ROOT = Path(__file__).resolve().parent.parent
BA_BIN = ROOT / "src" / "born_again_dp" / "bornAgain"
RESULTS_ROOT = ROOT / "src" / "born_again_dp" / "results" / "scaling"

METHODS = {
    "dp": 1,
    "greedy": 6,
    "beam": 7,
    "beam_lookahead": 7,
    "beam_balance": 7,
}

TREE_COUNTS = [2, 4, 6, 8, 10]
FOLDS = list(range(1, 3))          # 2 folds keeps runtime ~5x shorter
TIMEOUT_SECONDS = 120              # kill any single run after 2 min
DATASETS = None


def run_solver(forest_file, out_prefix, method_name, objective, max_trees):
    """Returns True on success, False on timeout."""
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BA_BIN),
        str(forest_file),
        str(out_prefix),
        "-trees", str(max_trees),
        "-obj", str(objective),
    ]
    if method_name == "beam_lookahead":
        cmd += ["-bh", "1"]
    elif method_name == "beam_balance":
        cmd += ["-bh", "2"]
    elif method_name == "beam":
        cmd += ["-bh", "0"]

    if os.environ.get("BA_PRINT_CMD"):
        print("  cmd:", " ".join(cmd), flush=True)

    try:
        subprocess.run(cmd, check=True, timeout=TIMEOUT_SECONDS)
        return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT ({TIMEOUT_SECONDS}s)")
        return False


def parse_out_file(path):
    with path.open("r") as f:
        line = f.readline().strip()
    parts = line.split(",")
    return {
        "depth": int(parts[6]),
        "splits": int(parts[7]),
        "leaves": int(parts[8]),
        "cpu_time": float(parts[10]),
        "nb_cells": int(parts[11]),
    }


def compute_row(dataset, fold, method, n_trees_ba):
    """Returns a dict of metrics, or None if the solver timed out."""
    df_train, df_test, info = datasets.load(dataset, fold)
    feature_cols = list(info["features"].values())
    target_col = df_train.columns[-1]

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values
    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values

    rf_full, forest_file = random_forests.load(
        X_train, y_train, dataset, fold, n_trees=10, return_file=True
    )
    rf_sub = persistence.classifier_from_file(
        forest_file, X_train, y_train, pruning=True, num_trees=n_trees_ba
    )

    tag = f"{dataset}_fold{fold}_{method}_t{n_trees_ba}"
    out_prefix = RESULTS_ROOT / method / tag

    ok = run_solver(
        Path(forest_file), out_prefix, method, METHODS[method], n_trees_ba
    )
    if not ok:
        return None

    out_info = parse_out_file(out_prefix.with_suffix(".out"))

    ba_clf = persistence.classifier_from_file(
        str(out_prefix) + ".tree",
        X_train, y_train,
        pruning=False, compute_score=False, num_trees=1,
    )

    rf_full_pred = rf_full.predict(X_test)
    rf_sub_pred = rf_sub.predict(X_test)
    ba_pred = ba_clf.predict(X_test)

    return {
        "dataset": dataset,
        "fold": fold,
        "method": method,
        "n_trees_ba": n_trees_ba,
        "depth": out_info["depth"],
        "leaves": out_info["leaves"],
        "cpu_time": out_info["cpu_time"],
        "nb_cells": out_info["nb_cells"],
        "rf_full_test_acc": float(np.mean(rf_full_pred == y_test)),
        "rf_sub_test_acc": float(np.mean(rf_sub_pred == y_test)),
        "ba_test_acc": float(np.mean(ba_pred == y_test)),
        "ba_train_acc": float(np.mean(ba_clf.predict(X_train) == y_train)),
        "agreement_vs_full": float(np.mean(ba_pred == rf_full_pred)),
        "agreement_vs_sub": float(np.mean(ba_pred == rf_sub_pred)),
    }


def main():
    if not BA_BIN.exists():
        raise SystemExit(
            f"bornAgain binary not found at {BA_BIN}. "
            "Build it with `make` in src/born_again_dp."
        )

    global DATASETS
    if DATASETS is None:
        DATASETS = list(datasets.dataset_names)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    for m in METHODS:
        (RESULTS_ROOT / m).mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS_ROOT / "scaling_summary.csv"
    fieldnames = [
        "dataset", "fold", "method", "n_trees_ba",
        "depth", "leaves", "cpu_time", "nb_cells",
        "rf_full_test_acc", "rf_sub_test_acc",
        "ba_test_acc", "ba_train_acc",
        "agreement_vs_full", "agreement_vs_sub",
    ]

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total = len(DATASETS) * len(TREE_COUNTS) * len(FOLDS) * len(METHODS)
        done = 0
        t0 = time.time()

        for dataset in DATASETS:
            for n_trees_ba in TREE_COUNTS:
                for fold in FOLDS:
                    for method in METHODS:
                        done += 1
                        elapsed = time.time() - t0
                        eta = (elapsed / done) * (total - done) if done else 0
                        print(
                            f"[{done}/{total}  ETA {eta/60:.0f}m] "
                            f"{dataset} | trees={n_trees_ba} | "
                            f"fold={fold} | {method}",
                            flush=True,
                        )
                        try:
                            row = compute_row(dataset, fold, method, n_trees_ba)
                            if row is not None:
                                writer.writerow(row)
                                f.flush()
                        except Exception as e:
                            print(f"  FAILED: {e}")

    print(f"\nScaling summary written to {summary_path}")


if __name__ == "__main__":
    main()
