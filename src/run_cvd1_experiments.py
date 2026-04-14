"""
Run DP/Greedy/Beam experiments for CVD-1 only and append to summary.csv.

Usage:
    python src/run_cvd1_experiments.py
"""

import csv
from pathlib import Path

from run_experiments import (
    BA_BIN,
    RESULTS_ROOT,
    METHODS,
    FOLDS,
    ensure_result_dirs,
    compute_metrics_for_run,
)


DATASET = "CVD-1"


def _complete_row(row: dict, fieldnames: list) -> dict:
    """Fill missing columns (e.g. after extending the schema) for DictWriter."""
    return {k: row.get(k, "") for k in fieldnames}


def main():
    if not BA_BIN.exists():
        raise SystemExit(f"bornAgain binary not found at {BA_BIN}. Build it with `make` in src/born_again_dp.")

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
        "rf_train_acc",
        "rf_test_acc",
        "ba_train_acc",
        "ba_test_acc",
        "ba_gen_gap",
        "rf_ba_agreement_train",
        "rf_ba_agreement_test",
    ]

    # Load existing rows if present, and remove old CVD-1 rows so reruns stay clean.
    rows = []
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("dataset") != DATASET]

    # Run new CVD-1 rows.
    new_rows = []
    for fold in FOLDS:
        for method in METHODS.keys():
            print(f"Running {method} on {DATASET}, fold {fold}...")
            row = compute_metrics_for_run(DATASET, fold, method)
            new_rows.append(row)

    # Write back full summary (old non-CVD + new CVD).
    all_rows = rows + new_rows
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(_complete_row(row, fieldnames))

    print(f"\nUpdated summary written to {summary_path}")
    print(f"Added/updated {len(new_rows)} rows for {DATASET}.")


if __name__ == "__main__":
    main()

