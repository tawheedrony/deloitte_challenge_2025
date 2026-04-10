#!/usr/bin/env python3
"""
Evaluate TSLib long-term forecasting models on the Task 1A wildfire datasets.

This script adapts the panel-style Task 1A data (one 4-year series per ZIP) to
TSLib's long-term forecasting model interface:
  - input years:  2018, 2019, 2020
  - target year:  2021
  - target:       avg_fire_risk_score

It preflights each requested TSLib model on the 3-step annual shape, skips
models that require unavailable optional dependencies or a longer context, and
trains/evaluates the compatible ones on each selected dataset.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import os
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[1]
TSLIB_ROOT = ROOT / "libs" / "Time-Series-Library"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
TSLIB_ROOT_STR = str(TSLIB_ROOT)
if TSLIB_ROOT_STR not in sys.path:
    sys.path.insert(0, TSLIB_ROOT_STR)


TARGET_COLUMN = "avg_fire_risk_score"
ID_COLUMNS = {"ZIP", "Year"}
DEFAULT_DATASETS = {
    "minimal": ROOT / "data" / "preprocessed" / "task1a" / "wildfire_risk_minimal.csv",
    "extended": ROOT / "data" / "preprocessed" / "task1a" / "wildfire_risk_extended.csv",
}
OUTPUT_ROOT = ROOT / "output" / "task1a_tslib_benchmarks"

ZERO_SHOT_MODELS = {
    "Chronos",
    "Chronos2",
    "Moirai",
    "Sundial",
    "TiRex",
    "TimeMoE",
    "TimesFM",
}
TASK_SPECIFIC_MODELS = {
    "KANAD": "anomaly-detection-only upstream model",
    "MambaSingleLayer": "classification-only upstream model",
}
MODEL_SPECIFIC_OVERRIDES = {
    "Koopa": {"status": "skipped", "reason": "builds its spectrum mask through TSLib's built-in data loader and is not compatible with the Task 1A panel adapter"},
    "MultiPatchFormer": {"status": "skipped", "reason": "hardcodes patch kernels far larger than the 3-step Task 1A history length"},
    "SCINet": {"status": "skipped", "reason": "requires a longer input sequence than 3 annual history steps"},
    "TemporalFusionTransformer": {"status": "skipped", "reason": "hardcodes upstream dataset schemas and does not support custom feature layouts"},
}


@dataclass
class DatasetBundle:
    name: str
    path: Path
    feature_columns: list[str]
    dropped_nan_columns: list[str]
    years: list[int]
    zip_codes: list[int]
    full_dataset: "WildfirePanelDataset"
    train_dataset: Subset
    val_dataset: Subset
    test_dataset: Subset
    feature_scaler: StandardScaler
    target_mean: float
    target_scale: float


class WildfirePanelDataset(Dataset):
    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        x_marks: np.ndarray,
        y_marks: np.ndarray,
        zip_codes: list[int],
        years: list[int],
    ) -> None:
        self.xs = torch.tensor(xs, dtype=torch.float32)
        self.ys = torch.tensor(ys, dtype=torch.float32)
        self.x_marks = torch.tensor(x_marks, dtype=torch.float32)
        self.y_marks = torch.tensor(y_marks, dtype=torch.float32)
        self.zip_codes = list(zip_codes)
        self.years = list(years)

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, index: int):
        return (
            self.xs[index],
            self.ys[index],
            self.x_marks[index],
            self.y_marks[index],
            self.zip_codes[index],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TSLib long-term forecasting models on Task 1A datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS.keys()),
        help="Dataset aliases and/or CSV paths. Defaults to: minimal extended",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="TSLib model names to evaluate. Use 'all' to scan the upstream models directory.",
    )
    parser.add_argument("--target", default=TARGET_COLUMN, help="Prediction target column.")
    parser.add_argument("--epochs", type=int, default=25, help="Max training epochs per model.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--train-split", type=float, default=0.70, help="Train fraction.")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Execution device.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--d-model", type=int, default=32, help="Shared TSLib d_model.")
    parser.add_argument("--d-ff", type=int, default=64, help="Shared TSLib d_ff.")
    parser.add_argument("--e-layers", type=int, default=2, help="Shared TSLib encoder layers.")
    parser.add_argument("--d-layers", type=int, default=1, help="Shared TSLib decoder layers.")
    parser.add_argument("--n-heads", type=int, default=4, help="Shared TSLib attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Shared dropout.")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k periods for TimesNet-style models.")
    parser.add_argument("--num-kernels", type=int, default=4, help="Kernel count for TimesNet.")
    parser.add_argument("--pred-len", type=int, default=1, help="Forecast horizon. Task 1A uses 1.")
    parser.add_argument("--history-len", type=int, default=3, help="History length. Task 1A uses 3.")
    parser.add_argument("--n-qubits", type=int, default=4, help="Qubit count for Q* hybrid models.")
    parser.add_argument("--n-qlayers", type=int, default=1, help="Variational layers for Q* hybrid models.")
    parser.add_argument("--n-esteps", type=int, default=1, help="Entangling steps for Q* hybrid models.")
    parser.add_argument(
        "--quantum-backend",
        default="default.qubit",
        help="PennyLane backend for Q* hybrid models.",
    )
    parser.add_argument(
        "--data-reupload",
        action="store_true",
        help="Re-encode data between variational layers in Q* hybrid models.",
    )
    parser.add_argument(
        "--include-quantum",
        action="store_true",
        help="When --models all, include Q* hybrid wrappers alongside classical models.",
    )
    parser.add_argument(
        "--quantum-only",
        action="store_true",
        help="When --models all, only run the Q* hybrid wrappers.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_ROOT),
        help="Directory for metrics, logs, and per-model predictions.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Only test model compatibility; do not train.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def base_model_name(model_name: str) -> str:
    if model_name.startswith("Q") and len(model_name) > 1:
        return model_name[1:]
    return model_name


def discover_model_names(include_quantum: bool = False) -> list[str]:
    model_dir = TSLIB_ROOT / "models"
    names = sorted(
        path.stem
        for path in model_dir.glob("*.py")
        if path.name != "__init__.py"
    )
    if include_quantum:
        return names
    return [name for name in names if not name.startswith("Q")]


def resolve_dataset_specs(dataset_args: list[str]) -> list[tuple[str, Path]]:
    resolved = []
    for dataset_arg in dataset_args:
        if dataset_arg in DEFAULT_DATASETS:
            resolved.append((dataset_arg, DEFAULT_DATASETS[dataset_arg]))
            continue
        csv_path = Path(dataset_arg).expanduser().resolve()
        resolved.append((csv_path.stem, csv_path))
    return resolved


def build_time_marks(years: Iterable[int]) -> np.ndarray:
    dates = pd.to_datetime([f"{int(year)}-01-01" for year in years])
    return np.stack(
        [
            dates.month.to_numpy(),
            dates.day.to_numpy(),
            dates.dayofweek.to_numpy(),
            np.zeros(len(dates), dtype=np.int64),
        ],
        axis=1,
    )


def load_panel_rows(
    dataset_name: str,
    csv_path: Path,
    target: str,
    history_len: int,
    pred_len: int,
) -> tuple[pd.DataFrame, list[str], list[str], list[int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path).sort_values(["ZIP", "Year"]).reset_index(drop=True)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {csv_path}")

    years = sorted(int(year) for year in df["Year"].unique())
    expected_steps = history_len + pred_len
    if len(years) != expected_steps:
        raise ValueError(
            f"{dataset_name}: expected exactly {expected_steps} yearly steps, found {years}"
        )

    base_features = [column for column in df.columns if column not in ID_COLUMNS and column != target]
    dropped_nan_columns = sorted(column for column in base_features if df[column].isna().any())
    feature_columns = [column for column in base_features if column not in dropped_nan_columns]
    feature_columns.append(target)
    return df, feature_columns, dropped_nan_columns, years


def split_indices(num_samples: int, train_frac: float, val_frac: float) -> tuple[list[int], list[int], list[int]]:
    n_train = int(num_samples * train_frac)
    n_val = int(num_samples * val_frac)
    n_test = num_samples - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"Invalid split for {num_samples} samples: train={n_train}, val={n_val}, test={n_test}"
        )
    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_train + n_val))
    test_idx = list(range(n_train + n_val, num_samples))
    return train_idx, val_idx, test_idx


def make_dataset_bundle(
    dataset_name: str,
    csv_path: Path,
    args: argparse.Namespace,
) -> DatasetBundle:
    df, feature_columns, dropped_nan_columns, years = load_panel_rows(
        dataset_name=dataset_name,
        csv_path=csv_path,
        target=args.target,
        history_len=args.history_len,
        pred_len=args.pred_len,
    )

    history_years = years[: args.history_len]
    decoder_years = years[: args.history_len] + years[args.history_len :]
    x_marks = build_time_marks(history_years)
    y_marks = build_time_marks(decoder_years)

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    zip_codes: list[int] = []

    for zip_code, zip_df in df.groupby("ZIP", sort=True):
        zip_df = zip_df.set_index("Year")
        if not all(year in zip_df.index for year in years):
            continue
        series_block = zip_df.loc[years, feature_columns]
        if series_block.isna().any().any():
            continue
        series_values = series_block.to_numpy(dtype=np.float32)
        x_rows.append(series_values[: args.history_len])
        y_rows.append(series_values)
        zip_codes.append(int(zip_code))

    if not x_rows:
        raise ValueError(f"{dataset_name}: no complete ZIP panels remained after filtering")

    x_array = np.stack(x_rows, axis=0)
    y_array = np.stack(y_rows, axis=0)
    x_marks_array = np.repeat(x_marks[np.newaxis, :, :], len(zip_codes), axis=0)
    y_marks_array = np.repeat(y_marks[np.newaxis, :, :], len(zip_codes), axis=0)

    train_idx, val_idx, test_idx = split_indices(
        num_samples=len(zip_codes),
        train_frac=args.train_split,
        val_frac=args.val_split,
    )

    scaler = StandardScaler()
    train_flat = np.concatenate(
        [
            x_array[train_idx].reshape(-1, x_array.shape[-1]),
            y_array[train_idx][:, -args.pred_len :, :].reshape(-1, y_array.shape[-1]),
        ],
        axis=0,
    )
    scaler.fit(train_flat)
    x_scaled = scaler.transform(x_array.reshape(-1, x_array.shape[-1])).reshape(x_array.shape)
    y_scaled = scaler.transform(y_array.reshape(-1, y_array.shape[-1])).reshape(y_array.shape)

    dataset = WildfirePanelDataset(
        xs=x_scaled,
        ys=y_scaled,
        x_marks=x_marks_array,
        y_marks=y_marks_array,
        zip_codes=zip_codes,
        years=years,
    )

    target_index = len(feature_columns) - 1
    return DatasetBundle(
        name=dataset_name,
        path=csv_path,
        feature_columns=feature_columns,
        dropped_nan_columns=dropped_nan_columns,
        years=years,
        zip_codes=zip_codes,
        full_dataset=dataset,
        train_dataset=Subset(dataset, train_idx),
        val_dataset=Subset(dataset, val_idx),
        test_dataset=Subset(dataset, test_idx),
        feature_scaler=scaler,
        target_mean=float(scaler.mean_[target_index]),
        target_scale=float(scaler.scale_[target_index]),
    )


def make_loader(subset: Subset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.runtime_device.type == "cuda",
        drop_last=False,
    )


def base_model_config(args: argparse.Namespace, feature_dim: int) -> dict:
    return {
        "task_name": "long_term_forecast",
        "is_training": 1,
        "model_id": "task1a",
        "data": "custom",
        "root_path": "",
        "data_path": "",
        "features": "MS",
        "target": args.target,
        "freq": "h",
        "checkpoints": str(Path(args.output_dir) / "checkpoints"),
        "seq_len": args.history_len,
        "label_len": args.history_len,
        "pred_len": args.pred_len,
        "seasonal_patterns": "Yearly",
        "inverse": False,
        "mask_rate": 0.25,
        "anomaly_ratio": 0.25,
        "expand": 2,
        "d_conv": 4,
        "tv_dt": 0,
        "tv_B": 0,
        "tv_C": 0,
        "use_D": 0,
        "top_k": args.top_k,
        "num_kernels": args.num_kernels,
        "enc_in": feature_dim,
        "dec_in": feature_dim,
        "c_out": feature_dim,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "e_layers": args.e_layers,
        "d_layers": args.d_layers,
        "d_ff": args.d_ff,
        "moving_avg": 1,
        "factor": 1,
        "distil": True,
        "dropout": args.dropout,
        "embed": "fixed",
        "activation": "gelu",
        "channel_independence": 1,
        "decomp_method": "moving_avg",
        "use_norm": 1,
        "down_sampling_layers": 0,
        "down_sampling_window": 1,
        "down_sampling_method": None,
        "seg_len": 1,
        "num_workers": args.num_workers,
        "itr": 1,
        "train_epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "learning_rate": args.lr,
        "des": "task1a",
        "loss": "MSE",
        "lradj": "type1",
        "use_amp": False,
        "use_gpu": args.runtime_device.type == "cuda",
        "gpu": 0,
        "gpu_type": "cuda",
        "use_multi_gpu": False,
        "devices": "0",
        "device": args.runtime_device,
        "p_hidden_dims": [128, 128],
        "p_hidden_layers": 2,
        "use_dtw": False,
        "augmentation_ratio": 0,
        "seed": args.seed,
        "jitter": False,
        "scaling": False,
        "permutation": False,
        "randompermutation": False,
        "magwarp": False,
        "timewarp": False,
        "windowslice": False,
        "windowwarp": False,
        "rotation": False,
        "spawner": False,
        "dtwwarp": False,
        "shapedtwwarp": False,
        "wdba": False,
        "discdtw": False,
        "discsdtw": False,
        "extra_tag": "",
        "patch_len": 1,
        "patch_stride": 1,
        "pyraformer_window_size": [1, 1],
        "pyraformer_inner_size": 1,
        "node_dim": 10,
        "gcn_depth": 2,
        "gcn_dropout": 0.3,
        "propalpha": 0.3,
        "conv_channel": 32,
        "skip_channel": 32,
        "individual": False,
        "alpha": 0.1,
        "top_p": 0.5,
        "pos": 1,
        "n_qubits": args.n_qubits,
        "n_qlayers": args.n_qlayers,
        "n_esteps": args.n_esteps,
        "data_reupload": args.data_reupload,
        "quantum_backend": args.quantum_backend,
    }


def build_model_namespace(model_name: str, args: argparse.Namespace, feature_dim: int) -> SimpleNamespace:
    config = base_model_config(args, feature_dim)
    config["model"] = model_name
    resolved_name = base_model_name(model_name)
    if resolved_name == "ETSformer":
        config["d_layers"] = config["e_layers"]
    elif resolved_name in {"PAttn", "PatchTST"}:
        config["patch_len"] = 1
        config["patch_stride"] = 1
    elif resolved_name == "Pyraformer":
        config["pyraformer_window_size"] = [1, 1]
        config["pyraformer_inner_size"] = 1
    elif resolved_name == "TimeMixer":
        config["down_sampling_layers"] = 1
        config["down_sampling_window"] = 1
        config["down_sampling_method"] = "avg"
    return SimpleNamespace(**config)


def import_model_class(model_name: str):
    module = importlib.import_module(f"models.{model_name}")
    return getattr(module, "Model", None) or getattr(module, model_name)


def try_build_model(model_name: str, model_args: SimpleNamespace, device: torch.device) -> nn.Module:
    model_class = import_model_class(model_name)
    resolved_name = base_model_name(model_name)
    if resolved_name in {"PAttn", "PatchTST"}:
        model = model_class(
            model_args,
            patch_len=model_args.patch_len,
            stride=model_args.patch_stride,
        ).float()
    elif resolved_name == "Pyraformer":
        model = model_class(
            model_args,
            window_size=model_args.pyraformer_window_size,
            inner_size=model_args.pyraformer_inner_size,
        ).float()
    else:
        model = model_class(model_args).float()
    return model.to(device)


def prepare_batch(batch, device: torch.device, pred_len: int) -> tuple[torch.Tensor, ...]:
    batch_x, batch_y, batch_x_mark, batch_y_mark, _ = batch
    batch_x = batch_x.to(device=device, dtype=torch.float32)
    batch_y = batch_y.to(device=device, dtype=torch.float32)
    batch_x_mark = batch_x_mark.to(device=device, dtype=torch.float32)
    batch_y_mark = batch_y_mark.to(device=device, dtype=torch.float32)
    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :], dtype=torch.float32)
    dec_inp = torch.cat([batch_y[:, :-pred_len, :], dec_inp], dim=1).to(device)
    return batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp


def forward_model(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_x_mark: torch.Tensor,
    batch_y_mark: torch.Tensor,
    dec_inp: torch.Tensor,
    pred_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    preds = outputs[:, -pred_len:, -1:]
    trues = batch_y[:, -pred_len:, -1:]
    return preds, trues


def preflight_model(
    model_name: str,
    bundle: DatasetBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[bool, str]:
    resolved_name = base_model_name(model_name)
    if resolved_name == "Mamba" and device.type != "cuda":
        return False, "upstream Mamba kernel requires CUDA, but this run is on CPU"
    if resolved_name in ZERO_SHOT_MODELS:
        return False, "zero-shot foundation model; this script only runs supervised long-term forecasting models"
    if resolved_name in TASK_SPECIFIC_MODELS:
        return False, TASK_SPECIFIC_MODELS[resolved_name]
    if resolved_name in MODEL_SPECIFIC_OVERRIDES:
        override = MODEL_SPECIFIC_OVERRIDES[resolved_name]
        return False, override["reason"]

    model_args = build_model_namespace(model_name, args, len(bundle.feature_columns))
    sample_batch = next(iter(make_loader(bundle.train_dataset, args, shuffle=False)))
    try:
        model = try_build_model(model_name, model_args, device)
        model.eval()
        with torch.no_grad():
            batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = prepare_batch(
                sample_batch, device, args.pred_len
            )
            preds, _ = forward_model(model, batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, args.pred_len)
        if preds.shape[-2:] != (args.pred_len, 1):
            return False, f"unexpected prediction shape {tuple(preds.shape)}"
        return True, "ok"
    except Exception as exc:
        short = f"{type(exc).__name__}: {exc}"
        return False, short


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pred_len: int,
    criterion: nn.Module,
) -> float:
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = prepare_batch(batch, device, pred_len)
            preds, trues = forward_model(model, batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, pred_len)
            losses.append(float(criterion(preds, trues).item()))
    return float(np.mean(losses)) if losses else float("nan")


def fit_model(
    model_name: str,
    bundle: DatasetBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    model_args = build_model_namespace(model_name, args, len(bundle.feature_columns))
    model = try_build_model(model_name, model_args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    train_loader = make_loader(bundle.train_dataset, args, shuffle=True)
    val_loader = make_loader(bundle.val_dataset, args, shuffle=False)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = prepare_batch(batch, device, args.pred_len)
            optimizer.zero_grad()
            preds, trues = forward_model(model, batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, args.pred_len)
            loss = criterion(preds, trues)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = evaluate_loss(model, val_loader, device, args.pred_len, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    model.load_state_dict(best_state)
    return model, {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "epochs_ran": history[-1]["epoch"] if history else 0,
        "history": history,
    }


def inverse_target(values: np.ndarray, mean: float, scale: float) -> np.ndarray:
    return values * scale + mean


def evaluate_model(
    model: nn.Module,
    bundle: DatasetBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    test_loader = make_loader(bundle.test_dataset, args, shuffle=False)
    preds_scaled = []
    trues_scaled = []
    test_zips = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = prepare_batch(batch, device, args.pred_len)
            preds, trues = forward_model(model, batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, args.pred_len)
            preds_scaled.append(preds.detach().cpu().numpy())
            trues_scaled.append(trues.detach().cpu().numpy())
            test_zips.extend(int(zip_code) for zip_code in batch[4])

    preds_scaled_np = np.concatenate(preds_scaled, axis=0).reshape(-1)
    trues_scaled_np = np.concatenate(trues_scaled, axis=0).reshape(-1)
    preds = inverse_target(preds_scaled_np, bundle.target_mean, bundle.target_scale)
    trues = inverse_target(trues_scaled_np, bundle.target_mean, bundle.target_scale)

    rmse = float(np.sqrt(mean_squared_error(trues, preds)))
    metrics = {
        "mae": float(mean_absolute_error(trues, preds)),
        "mse": float(mean_squared_error(trues, preds)),
        "rmse": rmse,
        "r2": float(r2_score(trues, preds)),
    }
    prediction_df = pd.DataFrame(
        {
            "ZIP": test_zips,
            "target_year": bundle.years[-1],
            "actual": trues,
            "prediction": preds,
        }
    )
    return metrics, prediction_df


def requested_models(model_args: list[str], args: argparse.Namespace, device: torch.device) -> list[str]:
    if len(model_args) == 1 and model_args[0].lower() == "all":
        excluded = set(ZERO_SHOT_MODELS) | set(TASK_SPECIFIC_MODELS) | set(MODEL_SPECIFIC_OVERRIDES)
        model_names = discover_model_names(include_quantum=args.include_quantum or args.quantum_only)
        if args.quantum_only:
            model_names = [model_name for model_name in model_names if model_name.startswith("Q")]
        if device.type != "cuda":
            excluded.add("Mamba")
        return [
            model_name
            for model_name in model_names
            if base_model_name(model_name) not in excluded
        ]
    return model_args


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "predictions": output_dir / "predictions",
        "histories": output_dir / "histories",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def main() -> None:
    args = parse_args()
    output_paths = ensure_output_dirs(Path(args.output_dir).resolve())
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.runtime_device = device

    dataset_specs = resolve_dataset_specs(args.datasets)
    model_names = requested_models(args.models, args, device)

    results = []
    skips = []

    for dataset_name, dataset_path in dataset_specs:
        bundle = make_dataset_bundle(dataset_name, dataset_path, args)
        print(
            f"[dataset] {dataset_name}: {len(bundle.full_dataset)} ZIP series, "
            f"{len(bundle.feature_columns)} channels, dropped_nan_columns={bundle.dropped_nan_columns}"
        )

        for model_name in model_names:
            ok, reason = preflight_model(model_name, bundle, args, device)
            if not ok:
                record = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "status": "skipped",
                    "reason": reason,
                    "dataset_path": str(dataset_path),
                }
                skips.append(record)
                print(f"[skip] {dataset_name} :: {model_name} :: {reason}")
                continue

            if args.preflight_only:
                record = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "status": "preflight_ok",
                    "reason": "compatible",
                    "dataset_path": str(dataset_path),
                }
                results.append(record)
                print(f"[ok] {dataset_name} :: {model_name} :: preflight only")
                continue

            try:
                print(f"[train] {dataset_name} :: {model_name}")
                model, fit_info = fit_model(model_name, bundle, args, device)
                metrics, prediction_df = evaluate_model(model, bundle, args, device)

                prediction_path = output_paths["predictions"] / f"{dataset_name}__{model_name}.csv"
                prediction_df.to_csv(prediction_path, index=False)

                history_path = output_paths["histories"] / f"{dataset_name}__{model_name}.json"
                pd.DataFrame(fit_info["history"]).to_json(history_path, orient="records", indent=2)

                record = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "status": "ok",
                    "dataset_path": str(dataset_path),
                    "train_samples": len(bundle.train_dataset),
                    "val_samples": len(bundle.val_dataset),
                    "test_samples": len(bundle.test_dataset),
                    "feature_count": len(bundle.feature_columns),
                    "dropped_nan_columns": ",".join(bundle.dropped_nan_columns),
                    "best_epoch": fit_info["best_epoch"],
                    "epochs_ran": fit_info["epochs_ran"],
                    "best_val_loss": fit_info["best_val_loss"],
                    "prediction_path": str(prediction_path),
                    **metrics,
                }
                results.append(record)
                print(
                    f"[done] {dataset_name} :: {model_name} :: "
                    f"MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}"
                )
            except Exception as exc:
                error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                record = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "status": "failed",
                    "reason": error_text,
                    "dataset_path": str(dataset_path),
                }
                results.append(record)
                print(f"[fail] {dataset_name} :: {model_name} :: {error_text}")

    results_df = pd.DataFrame(results)
    skips_df = pd.DataFrame(skips)
    results_path = output_paths["root"] / "results.csv"
    skips_path = output_paths["root"] / "skipped.csv"
    results_df.to_csv(results_path, index=False)
    skips_df.to_csv(skips_path, index=False)

    print(f"[write] results  -> {results_path}")
    print(f"[write] skipped  -> {skips_path}")


if __name__ == "__main__":
    main()
