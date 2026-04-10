#!/usr/bin/env python3
"""
Build Task 2 chained datasets using Task 1A prediction outputs.

For each Task 2 panel row, this script adds a per-ZIP feature carrying the
Task 1A model's predicted wildfire risk for the target year (2021). The value
is replicated across all historical rows for that ZIP so sequence models that
consume 2018-2020 inputs can access the chained risk signal.

For ZIPs that are not covered by the supplied prediction CSV, the script falls
back to the actual 2021 avg_fire_risk_score from the Task 2 dataset. This keeps
full dataset coverage while preserving the Task 1A test-split predictions where
they exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK2_DIR = ROOT / "data" / "preprocessed" / "task2"
TASK1A_PRED_DIR = ROOT / "output" / "task1a_tslib_benchmarks" / "predictions"

DATASET_SPECS = {
    "minimal": {
        "task2_path": TASK2_DIR / "insurance_minimal.csv",
        "prediction_path": TASK1A_PRED_DIR / "minimal__QCrossformer.csv",
    },
    "extended": {
        "task2_path": TASK2_DIR / "insurance_extended.csv",
        "prediction_path": TASK1A_PRED_DIR / "extended__QCrossformer.csv",
    },
}


def build_updated_dataset(
    dataset_name: str,
    task2_path: Path,
    prediction_path: Path,
    out_suffix: str,
) -> Path:
    if not task2_path.exists():
        raise FileNotFoundError(f"Task 2 dataset not found: {task2_path}")
    if not prediction_path.exists():
        raise FileNotFoundError(f"Task 1A prediction file not found: {prediction_path}")

    df = pd.read_csv(task2_path).sort_values(["ZIP", "Year"]).reset_index(drop=True)
    preds = pd.read_csv(prediction_path)

    feature_name = f"task1a_{out_suffix}_risk_2021"
    if feature_name in df.columns:
        raise ValueError(f"Output feature already exists in dataset: {feature_name}")

    actual_2021 = (
        df.loc[df["Year"] == 2021, ["ZIP", "avg_fire_risk_score"]]
        .drop_duplicates("ZIP")
        .rename(columns={"avg_fire_risk_score": feature_name})
    )
    chained = actual_2021.set_index("ZIP")[feature_name].to_dict()

    pred_map = preds.set_index("ZIP")["prediction"].to_dict()
    chained.update(pred_map)

    df[feature_name] = df["ZIP"].map(chained)

    source_name = f"task1a_{out_suffix}_risk_is_predicted"
    pred_zips = set(pred_map)
    df[source_name] = df["ZIP"].map(lambda zip_code: 1 if zip_code in pred_zips else 0)

    anchor_col = "avg_fire_risk_score"
    insert_at = df.columns.get_loc(anchor_col) + 1
    ordered_cols = list(df.columns)
    for col in [feature_name, source_name]:
        ordered_cols.remove(col)
    ordered_cols[insert_at:insert_at] = [feature_name, source_name]
    df = df[ordered_cols]

    out_path = task2_path.with_name(f"{task2_path.stem}_{out_suffix}.csv")
    df.to_csv(out_path, index=False)

    print(f"[write] {dataset_name:<8} -> {out_path}")
    print(
        f"         chained feature={feature_name}, "
        f"predicted ZIPs={len(pred_zips)}, total ZIPs={df['ZIP'].nunique()}"
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Task 2 chained datasets from Task 1A QCrossformer predictions."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS),
        default=["minimal", "extended"],
        help="Which Task 2 dataset variants to update.",
    )
    parser.add_argument(
        "--suffix",
        default="qcrossformer",
        help="Suffix to append to output dataset filenames and chained feature names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset_name in args.datasets:
        spec = DATASET_SPECS[dataset_name]
        build_updated_dataset(
            dataset_name=dataset_name,
            task2_path=spec["task2_path"],
            prediction_path=spec["prediction_path"],
            out_suffix=args.suffix,
        )


if __name__ == "__main__":
    main()
