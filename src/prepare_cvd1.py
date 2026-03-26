"""
Prepare CVD-1 for BA-Trees experiments:
1) Build 10 stratified train/test folds in resources/datasets/CVD-1
2) Train 10 random forests (one per fold)
3) Export forests in the BA-Trees text format to resources/forests/CVD-1

Usage:
    python src/prepare_cvd1.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "resources" / "datasets" / "CVD-1"
FOREST_DIR = ROOT / "resources" / "forests" / "CVD-1"
RAW_CSV = DATASET_DIR / "Cardiovascular_Disease_Dataset.csv"

DATASET_NAME = "CVD-1"
N_FOLDS = 10
N_TREES = 10
MAX_DEPTH = 3
RANDOM_SEED = 1


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw CVD data to the format expected by this repo."""
    df = df.copy()

    # Drop identifier column if present.
    if "patientid" in df.columns:
        df = df.drop(columns=["patientid"])

    if "target" not in df.columns:
        raise ValueError("Expected a 'target' column in CVD-1 dataset.")

    # Coerce all fields to numeric and drop rows with missing/non-numeric entries.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    # Target must be binary 0/1 for current experiment setup.
    df["target"] = df["target"].astype(int)
    uniq = sorted(df["target"].unique().tolist())
    if uniq != [0, 1]:
        raise ValueError(f"Expected binary target labels [0, 1], got {uniq}")

    # Keep target as last column.
    feature_cols = [c for c in df.columns if c != "target"]
    df = df[feature_cols + ["target"]]
    return df


def _tree_depths(children_left: np.ndarray, children_right: np.ndarray) -> list[int]:
    depths = [0] * len(children_left)
    stack = [(0, 0)]
    while stack:
        node, d = stack.pop()
        depths[node] = d
        l = children_left[node]
        r = children_right[node]
        if l != -1:
            stack.append((l, d + 1))
        if r != -1:
            stack.append((r, d + 1))
    return depths


def export_forest_txt(clf: RandomForestClassifier, output_file: Path, dataset_file_name: str):
    """Export sklearn random forest to BA-Trees text format."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    n_features = clf.n_features_in_
    classes = sorted(int(c) for c in clf.classes_)
    n_classes = len(classes)
    max_depth = max(est.tree_.max_depth for est in clf.estimators_)

    with output_file.open("w", encoding="utf-8") as f:
        f.write(f"DATASET_NAME: {dataset_file_name}\n")
        f.write("ENSEMBLE: RF\n")
        f.write(f"NB_TREES: {len(clf.estimators_)}\n")
        f.write(f"NB_FEATURES: {n_features}\n")
        f.write(f"NB_CLASSES: {n_classes}\n")
        f.write(f"MAX_TREE_DEPTH: {max_depth}\n")
        f.write(
            "Format: node / node type (LN - leave node, IN - internal node) "
            "left child / right child / feature / threshold / node_depth / majority class "
            "(starts with index 0)\n\n"
        )

        for t_idx, est in enumerate(clf.estimators_):
            tree_ = est.tree_
            left = tree_.children_left
            right = tree_.children_right
            feature = tree_.feature
            threshold = tree_.threshold
            value = tree_.value
            depths = _tree_depths(left, right)

            f.write(f"[TREE {t_idx}]\n")
            f.write(f"NB_NODES: {tree_.node_count}\n")
            for n in range(tree_.node_count):
                is_leaf = left[n] == right[n] == -1
                if is_leaf:
                    cls = int(np.argmax(value[n][0]))
                    f.write(f"{n} LN -1 -1 -1 -1 {depths[n]} {cls}\n")
                else:
                    thr = float(threshold[n])
                    f.write(
                        f"{n} IN {int(left[n])} {int(right[n])} "
                        f"{int(feature[n])} {thr:.12g} {depths[n]} -1\n"
                    )
            f.write("\n")


def main():
    if not RAW_CSV.exists():
        raise SystemExit(f"Raw dataset not found: {RAW_CSV}")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    FOREST_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(RAW_CSV)
    df = _validate_and_clean(raw_df)

    # Save "full" dataset and feature list in the same style as existing datasets.
    df.to_csv(DATASET_DIR / f"{DATASET_NAME}.csv", index=False)
    df.to_csv(DATASET_DIR / f"{DATASET_NAME}.full.csv", index=False)
    pd.DataFrame({"feature": [c for c in df.columns if c != "target"]}).to_csv(
        DATASET_DIR / f"{DATASET_NAME}.featurelist.csv", index=False
    )

    X = df.drop(columns=["target"]).values
    y = df["target"].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        test_df = df.iloc[te_idx].reset_index(drop=True)

        train_path = DATASET_DIR / f"{DATASET_NAME}.train{fold}.csv"
        test_path = DATASET_DIR / f"{DATASET_NAME}.test{fold}.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        rf = RandomForestClassifier(
            n_estimators=N_TREES,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_SEED + fold,
        )
        rf.fit(train_df.drop(columns=["target"]).values, train_df["target"].values)

        forest_path = FOREST_DIR / f"{DATASET_NAME}.RF{fold}.txt"
        export_forest_txt(rf, forest_path, dataset_file_name=f"{DATASET_NAME}.train{fold}.csv")
        print(f"Prepared fold {fold}: {train_path.name}, {test_path.name}, {forest_path.name}")

    print("\nCVD-1 preprocessing + forest export complete.")


if __name__ == "__main__":
    main()

