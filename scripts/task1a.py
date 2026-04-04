"""
scripts/task1a.py  —  Task 1A: Wildfire Risk Prediction (Train + Eval)
=======================================================================
Trains ALL model architectures on wildfire risk prediction, evaluates them,
and saves predictions to output/task1a/.

Quantum models are trained at 4, 8, and 12 qubits.
Classical baselines are trained once.
Optionally tests data re-uploading for cQLSTM and QSSM.

Usage:
    python scripts/task1a.py                         # all models, all qubit counts
    python scripts/task1a.py --qubits 4 8            # only 4 and 8 qubits
    python scripts/task1a.py --models cQLSTM LSTM    # specific models only
    python scripts/task1a.py --reupload              # test data re-uploading
    python scripts/task1a.py --epochs 20             # override max epochs

Output:
    output/task1a/
    ├── predictions/           per-model predicted risk scores (CSV)
    ├── task1a_results.csv     comparison table
    └── task1a_summary.txt     full report
"""

import argparse
import copy
import sys
import time
import warnings
from datetime import datetime
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as ls
import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                              mean_squared_error, r2_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from modules.model import LitModel
from modules.scheduler import scheduler_with_warmup

# ─────────────────────────── constants ───────────────────────────────────

QUANTUM_MODELS = ["QLSTM", "cQLSTM", "QSSM", "cQSSM", "QDeltaNet"]
CLASSICAL_MODELS = ["LSTM", "cLSTM", "SSM", "DeltaNet"]
ALL_MODELS = QUANTUM_MODELS + CLASSICAL_MODELS

# supports data_reupload flag
REUPLOAD_MODELS = ["cQLSTM", "QSSM"]

# base config for Task 1A (risk regression)
BASE_CFG = {
    "data": {
        "path": "data/wildfire_engineered.csv",
        "task": "regression",
        "time_steps": 3,
        "n_future": 1,
        "input_size": 15,
        "split": [0.70, 0.15, 0.15],
        "log_target": False,
        "features": [
            "avg_fire_risk_score", "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm",
            "temp_range", "drought_proxy", "log_prcp", "log_gis_acres",
            "fire_occurred", "high_risk_pct", "risk_concentration",
            "total_population", "poverty_rate", "log_median_income", "year_sin",
        ],
        "target": "avg_fire_risk_score",
    },
    "training": {
        "seed": 42,
        "batch_size": 32,
        "max_epochs": 40,
        "lr": 2.0e-3,
        "weight_decay": 1.0e-4,
        "warmup_epochs": 3,
        "warmup_start_factor": 0.01,
        "early_stop_patience": 10,
    },
    "logging": {
        "log_dir": "lightning_logs",
        "run_name": None,
    },
}


# ─────────────────────────── model configs ───────────────────────────────

def model_cfg(name, n_qubits=4, data_reupload=False):
    """Return model sub-config for a given architecture."""
    base = {"name": name, "hidden_size": 64}

    if name == "LSTM":
        return base
    if name == "cLSTM":
        return {**base, "decay_rate": 0.1}
    if name == "QLSTM":
        return {**base, "n_qubits": n_qubits, "n_qlayers": 1}
    if name == "cQLSTM":
        return {**base, "n_qubits": n_qubits, "n_qlayers": 1, "n_esteps": 1,
                "decay_rate": 0.1, "data_reupload": data_reupload}
    if name == "SSM":
        return {**base, "g_min": 0.05, "g_max": 0.95}
    if name == "QSSM":
        return {**base, "n_qubits": n_qubits, "n_qlayers": 1, "n_esteps": 1,
                "g_min": 0.05, "g_max": 0.95, "data_reupload": data_reupload}
    if name == "cQSSM":
        return {**base, "n_qubits": n_qubits, "n_qlayers": 2, "n_esteps": 1,
                "decay_rate": 0.1, "g_min": 0.05, "g_max": 0.95}
    if name == "DeltaNet":
        return {**base, "expand": 1, "neg_eigen": False, "chunk_size": 0}
    if name == "QDeltaNet":
        return {**base, "n_qubits": n_qubits, "n_qlayers": 1, "n_esteps": 1,
                "neg_eigen": False, "chunk_size": 0}
    raise ValueError(f"Unknown model: {name}")


# ─────────────────────────── dataset ─────────────────────────────────────

class WildfireDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def build_dataset(cfg):
    d = cfg["data"]
    df = pd.read_csv(ROOT / d["path"]).sort_values(["ZIP", "Year"])
    years = sorted(df["Year"].unique())
    in_years, tgt_year = years[:d["time_steps"]], years[-1]

    X_list, y_list, zip_list = [], [], []
    for zip_code, z_df in df.groupby("ZIP"):
        z_df = z_df.set_index("Year")
        if not all(yr in z_df.index for yr in in_years + [tgt_year]):
            continue
        xs = z_df.loc[in_years, d["features"]].values
        yv = z_df.loc[tgt_year, d["target"]]
        if np.isnan(xs).any() or np.isnan(yv):
            continue
        X_list.append(xs); y_list.append([yv]); zip_list.append(zip_code)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    N, T, F = X.shape

    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X.reshape(N * T, F)).reshape(N, T, F)

    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y)

    return WildfireDataset(X, y), scaler_X, scaler_y, zip_list


def make_loaders(dataset, cfg):
    split = cfg["data"]["split"]
    bs = cfg["training"]["batch_size"]
    n = len(dataset)
    n_tr = int(n * split[0])
    n_va = int(n * split[1])
    kw = dict(batch_size=bs, num_workers=4, pin_memory=torch.cuda.is_available(), drop_last=False)
    return (
        DataLoader(Subset(dataset, range(0, n_tr)), shuffle=True, **kw),
        DataLoader(Subset(dataset, range(n_tr, n_tr + n_va)), shuffle=False, **kw),
        DataLoader(Subset(dataset, range(n_tr + n_va, n)), shuffle=False, **kw),
    )


# ─────────────────────────── build model ─────────────────────────────────

def build_model(cfg):
    """Reuse train.py's model builder."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from train import build_model as _build
    return _build(cfg)


# ─────────────────────────── single experiment ───────────────────────────

def run_experiment(name, n_qubits, cfg, dataset, out_dir,
                   data_reupload=False, epoch_override=None):
    """Train one model, evaluate, return metrics dict."""
    cfg = copy.deepcopy(cfg)
    cfg["model"] = model_cfg(name, n_qubits, data_reupload)

    # run name
    q_tag = f"_{n_qubits}q" if name in QUANTUM_MODELS else ""
    r_tag = "_reupload" if data_reupload else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"task1a_{name}{q_tag}{r_tag}_{ts}"
    cfg["logging"]["run_name"] = run_name

    if epoch_override:
        cfg["training"]["max_epochs"] = epoch_override

    label = f"{name}{q_tag}{r_tag}"
    print(f"\n{'─'*60}")
    print(f"  Training: {label}")
    print(f"{'─'*60}")

    seed_everything(cfg["training"]["seed"], workers=True)
    torch.set_float32_matmul_precision("high")

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    t = cfg["training"]
    loss_fn = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = scheduler_with_warmup(
        scheduler=ls.CosineAnnealingLR(optimizer=optimizer, T_max=t["max_epochs"]),
        warmup_epochs=t["warmup_epochs"],
        start_factor=t["warmup_start_factor"],
    )

    lit = LitModel(model=model, output_size=cfg["data"]["n_future"],
                   criterion=loss_fn, optimizer=[optimizer], scheduler=[scheduler])

    # loaders
    train_loader, val_loader, test_loader = make_loaders(dataset, cfg)

    # logger
    log_dir = ROOT / cfg["logging"]["log_dir"]
    logger = TensorBoardLogger(save_dir=str(log_dir), name=run_name)
    Path(logger.log_dir).mkdir(parents=True, exist_ok=True)

    config_save = Path(logger.log_dir) / "run_config.yaml"
    with open(config_save, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    callbacks = [
        ModelCheckpoint(dirpath=logger.log_dir,
                        filename="best-epoch={epoch:02d}-valloss={val/loss:.4f}",
                        monitor="val/loss", mode="min", save_top_k=1),
        EarlyStopping(monitor="val/loss", patience=t["early_stop_patience"],
                      mode="min", verbose=False),
    ]

    trainer = Trainer(logger=logger, callbacks=callbacks, max_epochs=t["max_epochs"],
                      accelerator="auto", devices=1, log_every_n_steps=1,
                      enable_progress_bar=True, enable_model_summary=False)

    t0 = time.time()
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)
    train_time = time.time() - t0

    trainer.test(lit, dataloaders=test_loader, verbose=False)

    # ── collect metrics ──────────────────────────────────────────────
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(logger.log_dir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    metrics = {"model": label, "params": n_params, "train_time_s": round(train_time, 1)}
    for tag in ("val/loss", "val/r2", "test/loss", "test/r2", "test/rmse", "test/mae"):
        if tag in tags:
            vals = [e.value for e in ea.Scalars(tag)]
            metrics[tag] = vals[-1]
            if "loss" in tag:
                metrics[f"{tag}_best"] = min(vals)
            else:
                metrics[f"{tag}_best"] = max(vals)

    metrics["epochs"] = trainer.current_epoch

    # ── save predictions ─────────────────────────────────────────────
    lit.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for X, y in test_loader:
            all_pred.append(lit(X).numpy())
            all_true.append(y.numpy())

    pred_path = out_dir / "predictions" / f"{label}.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "predicted": np.vstack(all_pred).ravel(),
        "actual": np.vstack(all_true).ravel(),
    }).to_csv(pred_path, index=False)

    print(f"  R²={metrics.get('test/r2', 'N/A'):.4f}  "
          f"RMSE={metrics.get('test/rmse', 'N/A'):.4f}  "
          f"Time={train_time:.0f}s  Epochs={metrics['epochs']}")
    print(f"  Predictions -> {pred_path}")

    return metrics


# ─────────────────────────── main ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Task 1A: Train and evaluate all wildfire risk models.")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Models to train (default: all). Choices: {ALL_MODELS}")
    parser.add_argument("--qubits", nargs="+", type=int, default=[4, 8, 12],
                        help="Qubit counts for quantum models (default: 4 8 12)")
    parser.add_argument("--reupload", action="store_true",
                        help="Also test data re-uploading variants for cQLSTM/QSSM")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max_epochs")
    args = parser.parse_args()

    models = args.models or ALL_MODELS
    qubit_counts = args.qubits

    out_dir = ROOT / "output" / "task1a"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  TASK 1A — Wildfire Risk Prediction")
    print("  Models: " + ", ".join(models))
    print("  Qubits: " + ", ".join(map(str, qubit_counts)))
    if args.reupload:
        print("  Data re-uploading: enabled for cQLSTM, QSSM")
    print("=" * 60)

    # ── check data ───────────────────────────────────────────────────
    data_path = ROOT / BASE_CFG["data"]["path"]
    if not data_path.exists():
        print(f"Engineered data not found at {data_path}")
        print("Run:  python scripts/preprocess.py --engineered")
        sys.exit(1)

    # ── build dataset once ───────────────────────────────────────────
    cfg = copy.deepcopy(BASE_CFG)
    dataset, scaler_X, scaler_y, zip_list = build_dataset(cfg)
    print(f"\n  Dataset: {len(dataset):,} samples, "
          f"{cfg['data']['input_size']} features -> {cfg['data']['target']}")

    # ── run experiments ──────────────────────────────────────────────
    all_results = []

    for name in models:
        if name in QUANTUM_MODELS:
            for nq in qubit_counts:
                result = run_experiment(name, nq, cfg, dataset, out_dir,
                                        epoch_override=args.epochs)
                all_results.append(result)

                # optionally test data re-uploading
                if args.reupload and name in REUPLOAD_MODELS:
                    result = run_experiment(name, nq, cfg, dataset, out_dir,
                                            data_reupload=True,
                                            epoch_override=args.epochs)
                    all_results.append(result)
        else:
            result = run_experiment(name, 0, cfg, dataset, out_dir,
                                    epoch_override=args.epochs)
            all_results.append(result)

    # ── comparison table ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TASK 1A RESULTS  (ranked by test/r2)")
    print(f"{'='*70}")

    df = pd.DataFrame(all_results)
    display_cols = ["model", "params", "epochs", "test/r2", "test/rmse",
                    "test/mae", "val/loss_best", "train_time_s"]
    display_cols = [c for c in display_cols if c in df.columns]
    df_sorted = df[display_cols].sort_values("test/r2", ascending=False)
    print(df_sorted.to_string(index=False))

    # ── save results ─────────────────────────────────────────────────
    csv_path = out_dir / "task1a_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Results -> {csv_path}")

    # ── save summary report ──────────────────────────────────────────
    buf = StringIO()
    buf.write("=" * 70 + "\n")
    buf.write("  TASK 1A — WILDFIRE RISK PREDICTION SUMMARY\n")
    buf.write("  2026 Quantum Sustainability Challenge\n")
    buf.write("=" * 70 + "\n\n")
    buf.write(f"  Target: {cfg['data']['target']}\n")
    buf.write(f"  Dataset: {cfg['data']['path']}\n")
    buf.write(f"  Samples: {len(dataset):,}\n")
    buf.write(f"  Features: {cfg['data']['input_size']}\n")
    buf.write(f"  Split: {cfg['data']['split']}\n\n")

    buf.write("  RESULTS (ranked by Test R²)\n")
    buf.write("-" * 70 + "\n")
    buf.write(df_sorted.to_string(index=False))
    buf.write("\n\n")

    # best quantum vs best classical
    q_mask = df["model"].str.contains("|".join(QUANTUM_MODELS))
    c_mask = ~q_mask
    if q_mask.any() and c_mask.any():
        best_q = df.loc[q_mask].sort_values("test/r2", ascending=False).iloc[0]
        best_c = df.loc[c_mask].sort_values("test/r2", ascending=False).iloc[0]
        buf.write("  KEY COMPARISON\n")
        buf.write("-" * 70 + "\n")
        buf.write(f"  Best quantum  : {best_q['model']:<20} R²={best_q['test/r2']:.4f}  "
                  f"({best_q['params']:,} params)\n")
        buf.write(f"  Best classical: {best_c['model']:<20} R²={best_c['test/r2']:.4f}  "
                  f"({best_c['params']:,} params)\n")
        param_ratio = best_c["params"] / max(best_q["params"], 1)
        buf.write(f"  Param ratio   : {param_ratio:.1f}x (quantum uses fewer)\n")
        buf.write(f"  R² delta      : {best_q['test/r2'] - best_c['test/r2']:+.4f}\n")

    summary_path = out_dir / "task1a_summary.txt"
    summary_path.write_text(buf.getvalue())
    print(f"  Summary -> {summary_path}")

    print(f"\n  Predictions in: {out_dir / 'predictions'}")
    print(f"  Evaluate any run: python scripts/eval.py --run 'lightning_logs/task1a_*/version_0'")


if __name__ == "__main__":
    main()
