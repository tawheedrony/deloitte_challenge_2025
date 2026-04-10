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
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK2_DIR = ROOT / "data" / "preprocessed" / "task2"
TASK1A_OUTPUT_ROOT = ROOT / "output" / "task1a_tslib_benchmarks"
TASK2_BENCHMARK_OUTPUT_ROOT = ROOT / "output" / "task2_tslib_benchmarks"

DATASET_SPECS = {
    "minimal": {
        "task2_path": TASK2_DIR / "insurance_minimal.csv",
    },
    "extended": {
        "task2_path": TASK2_DIR / "insurance_extended.csv",
    },
}


def sanitize_name(value: str, separator: str = "_") -> str:
    cleaned = "".join(char.lower() if char.isalnum() else separator for char in value.strip())
    doubled = separator * 2
    while doubled in cleaned:
        cleaned = cleaned.replace(doubled, separator)
    return cleaned.strip(separator) or "run"


def prediction_dir_for(path: Path) -> Path | None:
    if path.name == "predictions" and path.is_dir():
        return path
    prediction_dir = path / "predictions"
    if prediction_dir.is_dir():
        return prediction_dir
    return None


def infer_run_tag(path: Path) -> str:
    if path.name == "predictions":
        run_root = path.parent
        if (run_root / "run_config.json").exists():
            return run_root.name
        return "legacy"
    if (path / "run_config.json").exists():
        return path.name
    return "legacy"


def list_run_roots(prediction_root: Path) -> list[Path]:
    candidates: list[Path] = []
    direct_prediction_dir = prediction_dir_for(prediction_root)
    if direct_prediction_dir is not None:
        candidates.append(prediction_root)

    runs_dir = prediction_root / "runs"
    if runs_dir.is_dir():
        candidates.extend(
            run_dir
            for run_dir in sorted(runs_dir.iterdir())
            if run_dir.is_dir() and prediction_dir_for(run_dir) is not None
        )
    return candidates


def has_required_predictions(run_root: Path, dataset_names: list[str], task1a_model: str) -> bool:
    prediction_dir = prediction_dir_for(run_root)
    if prediction_dir is None:
        return False
    return all(
        (prediction_dir / f"{dataset_name}__{task1a_model}.csv").exists()
        for dataset_name in dataset_names
    )


def resolve_task1a_run_root(
    prediction_root: Path,
    dataset_names: list[str],
    task1a_model: str,
    run_tag: str | None,
) -> tuple[Path, str]:
    if run_tag:
        normalized_tag = sanitize_name(run_tag, separator="-")
        explicit_candidates = [
            prediction_root / "runs" / normalized_tag,
            prediction_root / normalized_tag,
            prediction_root,
        ]
        for candidate in explicit_candidates:
            if not candidate.exists():
                continue
            if candidate == prediction_root:
                if candidate.name != normalized_tag and not (candidate / "run_config.json").exists():
                    continue
            elif candidate.name != normalized_tag:
                continue
            if has_required_predictions(candidate, dataset_names, task1a_model):
                return candidate, normalized_tag
        raise FileNotFoundError(
            f"No Task 1A run '{normalized_tag}' under {prediction_root} with predictions "
            f"for datasets={dataset_names} and model={task1a_model}"
        )

    latest_run_file = prediction_root / "latest_run.txt"
    if latest_run_file.exists():
        latest_tag = latest_run_file.read_text(encoding="utf-8").strip()
        latest_candidate = prediction_root / "runs" / latest_tag
        if latest_tag and has_required_predictions(latest_candidate, dataset_names, task1a_model):
            return latest_candidate, latest_tag

    candidates = [
        candidate
        for candidate in list_run_roots(prediction_root)
        if has_required_predictions(candidate, dataset_names, task1a_model)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No Task 1A outputs under {prediction_root} provide predictions for "
            f"datasets={dataset_names} and model={task1a_model}"
        )

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    selected = candidates[0]
    selected_tag = infer_run_tag(selected)
    return selected, selected_tag


def resolve_prediction_paths(
    run_root: Path,
    dataset_names: list[str],
    task1a_model: str,
) -> dict[str, Path]:
    prediction_dir = prediction_dir_for(run_root)
    if prediction_dir is None:
        raise FileNotFoundError(f"No predictions directory found under {run_root}")
    return {
        dataset_name: prediction_dir / f"{dataset_name}__{task1a_model}.csv"
        for dataset_name in dataset_names
    }


def build_updated_dataset(
    dataset_name: str,
    task2_path: Path,
    prediction_path: Path,
    out_suffix: str,
    output_dir: Path,
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

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{task2_path.stem}_{out_suffix}.csv"
    df.to_csv(out_path, index=False)

    print(f"[write] {dataset_name:<8} -> {out_path}")
    print(
        f"         chained feature={feature_name}, "
        f"predicted ZIPs={len(pred_zips)}, total ZIPs={df['ZIP'].nunique()}"
    )
    return out_path


def run_task2_benchmarks(
    dataset_paths: list[Path],
    args: argparse.Namespace,
    output_suffix: str,
) -> None:
    benchmark_output_dir = Path(args.benchmark_output_dir).expanduser().resolve()
    benchmark_run_tag = args.benchmark_run_tag or f"task2_{output_suffix}"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "task1a.py"),
        "--datasets",
        *[str(path) for path in dataset_paths],
        "--target",
        "earned_premium",
        "--models",
        *args.benchmark_models,
        "--split-mode",
        args.benchmark_split_mode,
        "--device",
        args.benchmark_device,
        "--quantum-backend",
        args.benchmark_quantum_backend,
        "--output-dir",
        str(benchmark_output_dir),
        "--run-tag",
        benchmark_run_tag,
    ]
    if args.benchmark_include_quantum:
        command.append("--include-quantum")
    if args.benchmark_quantum_only:
        command.append("--quantum-only")
    if args.benchmark_preflight_only:
        command.append("--preflight-only")
    if args.benchmark_data_reupload:
        command.append("--data-reupload")

    command.extend(
        [
            "--n-qubits",
            str(args.benchmark_n_qubits),
            "--n-qlayers",
            str(args.benchmark_n_qlayers),
            "--n-esteps",
            str(args.benchmark_n_esteps),
        ]
    )

    print(f"[bench] launching -> {' '.join(command)}")
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Task 2 chained datasets from Task 1A prediction outputs."
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
        default=None,
        help="Optional suffix for output dataset filenames and chained feature names.",
    )
    parser.add_argument(
        "--task1a-model",
        default="QCrossformer",
        help="Task 1A model name whose predictions should be chained into Task 2.",
    )
    parser.add_argument(
        "--prediction-root",
        default=str(TASK1A_OUTPUT_ROOT),
        help="Task 1A output root, run directory, or predictions directory.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Specific Task 1A run tag to consume. Defaults to the latest compatible run.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(TASK2_DIR),
        help="Directory where the chained Task 2 datasets will be written.",
    )
    parser.add_argument(
        "--run-benchmarks",
        action="store_true",
        help="After building the chained Task 2 CSVs, launch the Task 2 TSLib benchmark runner.",
    )
    parser.add_argument(
        "--benchmark-models",
        nargs="+",
        default=["all"],
        help="Model list forwarded to scripts/task1a.py when --run-benchmarks is set.",
    )
    parser.add_argument(
        "--benchmark-output-dir",
        default=str(TASK2_BENCHMARK_OUTPUT_ROOT),
        help="Output directory for the downstream Task 2 benchmark run.",
    )
    parser.add_argument(
        "--benchmark-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device forwarded to the downstream Task 2 benchmark run.",
    )
    parser.add_argument(
        "--benchmark-quantum-backend",
        default="default.qubit",
        help="Quantum backend forwarded to the downstream Task 2 benchmark run.",
    )
    parser.add_argument(
        "--benchmark-run-tag",
        default=None,
        help="Optional run tag for the downstream Task 2 benchmark run.",
    )
    parser.add_argument(
        "--benchmark-split-mode",
        choices=["full_2021", "zip_holdout"],
        default="full_2021",
        help="Split mode forwarded to the downstream Task 2 benchmark run.",
    )
    parser.add_argument(
        "--benchmark-include-quantum",
        action="store_true",
        help="Forward --include-quantum to the downstream Task 2 benchmark run.",
    )
    parser.add_argument(
        "--benchmark-quantum-only",
        action="store_true",
        help="Forward --quantum-only to the downstream Task 2 benchmark run.",
    )
    parser.add_argument(
        "--benchmark-preflight-only",
        action="store_true",
        help="Forward --preflight-only to the downstream Task 2 benchmark run.",
    )
    parser.add_argument(
        "--benchmark-n-qubits",
        type=int,
        default=4,
        help="Qubit count forwarded to the downstream Task 2 benchmark run.",
    )
    parser.add_argument(
        "--benchmark-n-qlayers",
        type=int,
        default=1,
        help="Quantum depth forwarded to the downstream Task 2 benchmark run.",
    )
    parser.add_argument(
        "--benchmark-n-esteps",
        type=int,
        default=1,
        help="Entangling steps forwarded to the downstream Task 2 benchmark run.",
    )
    parser.add_argument(
        "--benchmark-data-reupload",
        action="store_true",
        help="Forward --data-reupload to the downstream Task 2 benchmark run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = list(args.datasets)
    prediction_root = Path(args.prediction_root).expanduser().resolve()
    run_root, resolved_run_tag = resolve_task1a_run_root(
        prediction_root=prediction_root,
        dataset_names=dataset_names,
        task1a_model=args.task1a_model,
        run_tag=args.run_tag,
    )
    prediction_paths = resolve_prediction_paths(run_root, dataset_names, args.task1a_model)

    output_suffix = args.suffix
    if output_suffix is None:
        model_tag = sanitize_name(args.task1a_model)
        if resolved_run_tag == "legacy":
            output_suffix = model_tag
        else:
            output_suffix = sanitize_name(f"{model_tag}_{resolved_run_tag}")
    else:
        output_suffix = sanitize_name(output_suffix)

    output_dir = Path(args.output_dir).expanduser().resolve()
    print(f"[run] Task 1A source -> {run_root}")
    print(f"[run] Task 1A model  -> {args.task1a_model}")
    print(f"[run] Task 2 suffix  -> {output_suffix}")

    built_paths: list[Path] = []
    for dataset_name in dataset_names:
        spec = DATASET_SPECS[dataset_name]
        built_paths.append(
            build_updated_dataset(
            dataset_name=dataset_name,
            task2_path=spec["task2_path"],
            prediction_path=prediction_paths[dataset_name],
            out_suffix=output_suffix,
            output_dir=output_dir,
        )
        )

    if args.run_benchmarks:
        run_task2_benchmarks(
            dataset_paths=built_paths,
            args=args,
            output_suffix=output_suffix,
        )


if __name__ == "__main__":
    main()
