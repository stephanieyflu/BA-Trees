"""
Run BA-tree experiments (DP, Greedy, Beam) on all datasets/folds and
summarize results including agreement with the original random forest.

This script assumes:
  - The C++ binary `born_again_dp/bornAgain` has been built (via `make`).
  - The Python environment satisfies `docs/requirements.txt`.

It will:
  1. Loop over all datasets and folds.
  2. Run bornAgain with objectives:
       - 1 = NbLeaves (DP, optimal),
       - 6 = GreedyExactCells,
       - 7 = BeamSearchExactCells.
  3. Parse the `.out` files and compute:
       - RF accuracy on test set,
       - BA-tree accuracy on test set,
       - RF–BA agreement (%) on test set.
  4. Write a CSV summary to `src/born_again_dp/results/summary.csv`.
"""

import csv
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import datasets
import random_forests
import persistence


ROOT = Path(__file__).resolve().parent.parent
BA_BIN = ROOT / "src" / "born_again_dp" / "bornAgain"
RESULTS_ROOT = ROOT / "src" / "born_again_dp" / "results"


METHODS = {
    "dp": 1,          # dynamic programming, NbLeaves
    "greedy": 6,      # GreedyExactCells
    "beam": 7,        # BeamSearchExactCells
    "beam_lookahead": 7, # BeamSearchExactCells with Lookahead Heuristic, bh=1
    "beam_balance": 7, # BeamSearchExactCells with Depth Penalty Heuristic, bh=2
}

# Datasets / folds to run.
# Use all datasets defined in datasets.py, and all 10 folds.
DATASETS = None  # filled from datasets.dataset_names in main()
FOLDS = list(range(1, 11))

# To keep runtimes manageable across all datasets, we limit the number of trees
# read by the C++ solver. You can increase this (up to 10) if your machine can handle it.
MAX_TREES_PER_RUN = 4


def ensure_result_dirs():
    for m in METHODS.keys():
        (RESULTS_ROOT / m).mkdir(parents=True, exist_ok=True)


def run_solver(forest_file: Path, out_prefix: Path, method_name: str, objective: int, max_trees: int):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BA_BIN),
        str(forest_file),
        str(out_prefix),
        "-trees", str(max_trees),
        "-obj", str(objective),
    ]
    
    # Map method name to the heuristic ID
    if method_name == "beam_lookahead":
        cmd += ["-bh", "1"]
    elif method_name == "beam_balance":
        cmd += ["-bh", "2"]
    elif method_name == "beam":
        cmd += ["-bh", "0"]

    if os.environ.get("BA_PRINT_CMD"):
        print("bornAgain:", " ".join(cmd), flush=True)

    subprocess.run(cmd, check=True)


def parse_out_file(path: Path):
    """
    Parse a single-line .out file from bornAgain.

    Format (from exportRunStatistics):
      datasetName,ensembleMethod,nbTrees,nbFeatures,nbClasses,objective,
      finalDepth,finalSplits,finalLeaves,1,cpuTime,nbCells,regionsMemorizedDP,iterationsDP
    """
    with path.open("r") as f:
        line = f.readline().strip()
    parts = line.split(",")
    return {
        "dataset_name": parts[0],
        "ensemble_method": parts[1],
        "nb_trees": int(parts[2]),
        "nb_features": int(parts[3]),
        "nb_classes": int(parts[4]),
        "objective": int(parts[5]),
        "depth": int(parts[6]),
        "splits": int(parts[7]),
        "leaves": int(parts[8]),
        "cpu_time": float(parts[10]),
        "nb_cells": int(parts[11]),
        "nb_subproblems": float(parts[12]),
        "nb_recursive_calls": float(parts[13]),
    }


def compute_metrics_for_run(dataset: str, fold: int, method: str):
    """
    For a given dataset/fold/method:
      - Run bornAgain if needed.
      - Compute RF accuracy, BA accuracy, and agreement on the test set.
      - Return a dict of metrics (including tree stats from .out).
    """
    # 1) Load train/test data
    df_train, df_test, info = datasets.load(dataset, fold)
    feature_cols = list(info["features"].values())
    target_col = df_train.columns[-1]

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values
    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values

    # 2) Load RF and get forest file
    rf_clf, forest_file = random_forests.load(
        X_train, y_train, dataset, fold, n_trees=10, return_file=True
    )

    # 3) Run solver 
    out_prefix = RESULTS_ROOT / method / f"{dataset}_fold{fold}_{method}"     # Use the 'method' string (e.g., "beam_lookahead") to define the path
    run_solver(Path(forest_file), out_prefix, method, METHODS[method], MAX_TREES_PER_RUN)  # Run the solver with the updated pathing
    
    out_info = parse_out_file(out_prefix.with_suffix(".out"))

    # 4) Load BA-tree as classifier
    ba_clf = persistence.classifier_from_file(
        str(out_prefix) + ".tree",
        X_train,
        y_train,
        pruning=False,
        compute_score=False,
        num_trees=1,
    )

    # 5) Compute metrics on test set
    rf_pred = rf_clf.predict(X_test)
    ba_pred = ba_clf.predict(X_test)

    rf_acc = np.mean(rf_pred == y_test)
    ba_acc = np.mean(ba_pred == y_test)
    agreement = np.mean(rf_pred == ba_pred)

    row = {
        "dataset": dataset,
        "fold": fold,
        "method": method,
        "objective": out_info["objective"],
        "depth": out_info["depth"],
        "splits": out_info["splits"],
        "leaves": out_info["leaves"],
        "cpu_time": out_info["cpu_time"],
        "nb_cells": out_info["nb_cells"],
        "nb_subproblems": out_info["nb_subproblems"],
        "nb_recursive_calls": out_info["nb_recursive_calls"],
        "rf_acc": rf_acc,
        "ba_acc": ba_acc,
        "rf_ba_agreement": agreement,
    }
    return row


def main():
    if not BA_BIN.exists():
        raise SystemExit(f"bornAgain binary not found at {BA_BIN}. Build it with `make` in src/born_again_dp.")

    # Initialize dataset list from datasets.py
    global DATASETS
    if DATASETS is None:
        DATASETS = list(datasets.dataset_names)

    ensure_result_dirs()

    summary_path = RESULTS_ROOT / "summary.csv"
    fieldnames = [
        "dataset",
        "fold",
        "method",
        "objective",
        "depth",
        "splits",
        "leaves",
        "cpu_time",
        "nb_cells",
        "nb_subproblems",
        "nb_recursive_calls",
        "rf_acc",
        "ba_acc",
        "rf_ba_agreement",
    ]

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for dataset in DATASETS:
            for fold in FOLDS:
                for method in METHODS.keys():
                    print(f"Running {method} on {dataset}, fold {fold}...")
                    row = compute_metrics_for_run(dataset, fold, method)
                    writer.writerow(row)
                    f.flush()

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()

