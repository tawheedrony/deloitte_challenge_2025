"""
train_wildfire.py  —  unified training entry point
===================================================
Trains any supported model on the Quantum Sustainability Challenge dataset.

Prerequisites:
    python preprocess_wildfire.py

Usage:
    # default (cQLSTM, settings from configs/wildfire.yaml)
    python train_wildfire.py

    # switch model
    python train_wildfire.py --model.name LSTM
    python train_wildfire.py --model.name QSSM

    # override any config key with dot notation
    python train_wildfire.py --model.name cLSTM --training.max_epochs 60
    python train_wildfire.py --logging.run_name my_exp --model.name cQLSTM

    # custom config file
    python train_wildfire.py --config configs/wildfire.yaml

Supported models
----------------
  LSTM      Classical LSTM
  cLSTM     Classical LSTM + decay + feature weighting
  QLSTM     4-gate Quantum LSTM (one VQC per gate)
  cQLSTM    Unified VQC + decay + feature weighting  (default)
  QSSM      Quantum State-Space Model (2-gate SSM recurrence)
  DeltaNet  Classical DeltaNet (delta rule memory update)
  QDeltaNet Quantum DeltaNet (VQC-generated Q/K/V + delta rule)

Outputs (per run)
-----------------
  lightning_logs/{run_name}/version_N/
    ├── best-{epoch}-{val/loss}.ckpt   best checkpoint
    ├── run_config.yaml                full resolved config (for eval)
    └── events.out.tfevents.*          TensorBoard logs

Evaluate a finished run:
    python eval_wildfire.py --run lightning_logs/{run_name}/version_N
"""

import argparse
import copy
import sys
import warnings
from datetime import datetime
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from modules.model import LitModel
from modules.scheduler import scheduler_with_warmup

# ─────────────────────────────── model registry ───────────────────────────

def build_model(cfg: dict) -> nn.Module:
    """Instantiate the model specified in cfg['model']['name']."""
    m = cfg["model"]
    name = m["name"]

    if name == "LSTM":
        from models.LSTM import LSTM
        return LSTM(input_size=cfg["data"]["input_size"],
                    hidden_size=m["hidden_size"])

    if name == "cLSTM":
        from models.cLSTM import cLSTM
        return cLSTM(input_size=cfg["data"]["input_size"],
                     hidden_size=m["hidden_size"],
                     decay_rate=m["decay_rate"])

    if name == "QLSTM":
        from models.QLSTM import QLSTM
        return QLSTM(input_size=cfg["data"]["input_size"],
                     hidden_size=m["hidden_size"],
                     n_qubits=m["n_qubits"],
                     n_qlayers=m["n_qlayers"])

    if name == "cQLSTM":
        from models.cQLSTM import cQLSTM
        return cQLSTM(input_size=cfg["data"]["input_size"],
                      hidden_size=m["hidden_size"],
                      decay_rate=m["decay_rate"],
                      n_qubits=m["n_qubits"],
                      n_qlayers=m["n_qlayers"],
                      n_esteps=m["n_esteps"],
                      data_reupload=m.get("data_reupload", False))

    if name == "SSM":
        from models.SSM import SSM
        return SSM(input_size=cfg["data"]["input_size"],
                   hidden_size=m["hidden_size"],
                   g_min=m["g_min"],
                   g_max=m["g_max"])

    if name == "QSSM":
        from models.QSSM import QuantumSSM
        return QuantumSSM(input_size=cfg["data"]["input_size"],
                          hidden_size=m["hidden_size"],
                          n_qubits=m["n_qubits"],
                          n_qlayers=m["n_qlayers"],
                          n_esteps=m["n_esteps"],
                          g_min=m["g_min"],
                          g_max=m["g_max"],
                          data_reupload=m.get("data_reupload", False))

    if name == "cQSSM":
        from models.cQSSM import cQSSM
        return cQSSM(input_size=cfg["data"]["input_size"],
                     hidden_size=m["hidden_size"],
                     n_qubits=m["n_qubits"],
                     n_qlayers=m["n_qlayers"],
                     n_esteps=m["n_esteps"],
                     decay_rate=m["decay_rate"],
                     g_min=m["g_min"],
                     g_max=m["g_max"])

    if name == "DeltaNet":
        from models.DeltaNet import DeltaNet
        return DeltaNet(input_size=cfg["data"]["input_size"],
                        hidden_size=m["hidden_size"],
                        expand=m.get("expand", 1),
                        neg_eigen=m.get("neg_eigen", False),
                        chunk_size=m.get("chunk_size", 0))

    if name == "QDeltaNet":
        from models.QDeltaNet import QDeltaNet
        return QDeltaNet(input_size=cfg["data"]["input_size"],
                         hidden_size=m["hidden_size"],
                         n_qubits=m["n_qubits"],
                         n_qlayers=m["n_qlayers"],
                         n_esteps=m["n_esteps"],
                         neg_eigen=m.get("neg_eigen", False),
                         chunk_size=m.get("chunk_size", 0))

    raise ValueError(
        f"Unknown model '{name}'. "
        "Choose from: LSTM, cLSTM, QLSTM, cQLSTM, SSM, QSSM, cQSSM, DeltaNet, QDeltaNet"
    )

# ─────────────────────────────── dataset ──────────────────────────────────

class WildfireDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_dataset(df: pd.DataFrame, cfg: dict):
    d = cfg["data"]
    features    = d["features"]
    target      = d["target"]
    time_steps  = d["time_steps"]
    task        = d.get("task", "regression")

    df = df.sort_values(["ZIP", "Year"])
    years       = sorted(df["Year"].unique())
    input_years = years[:time_steps]
    target_year = years[-1]

    X_list, y_list = [], []
    for _, z_df in df.groupby("ZIP"):
        z_df = z_df.set_index("Year")
        if not all(yr in z_df.index for yr in input_years + [target_year]):
            continue
        x_seq = z_df.loc[input_years, features].values
        y_val = z_df.loc[target_year, target]
        if np.isnan(x_seq).any() or np.isnan(y_val):
            continue
        X_list.append(x_seq)
        y_list.append([y_val])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    N, T, F  = X.shape
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X.reshape(N * T, F)).reshape(N, T, F)

    if task == "classification":
        # binary labels — no scaling or log transform
        scaler_y = None
    else:
        scaler_y = StandardScaler()
        if d.get("log_target", False):
            y = np.log1p(y)
        y = scaler_y.fit_transform(y)

    print(f"  Task    : {task}")
    print(f"  Samples : {N:,}  (train/val/test split: {d['split']})")
    print(f"  X shape : {X.shape}    y shape: {y.shape}")
    if task == "regression" and d.get("log_target", False):
        print(f"  log_target=True  (predicting log1p of target)")

    return WildfireDataset(X, y), scaler_X, scaler_y


def make_loaders(dataset: WildfireDataset, cfg: dict):
    split = cfg["data"]["split"]
    bs    = cfg["training"]["batch_size"]
    n     = len(dataset)
    n_tr  = int(n * split[0])
    n_va  = int(n * split[1])

    loader_cfg = dict(
        batch_size=bs, num_workers=4,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )
    return (
        DataLoader(Subset(dataset, range(0, n_tr)),           shuffle=True,  **loader_cfg),
        DataLoader(Subset(dataset, range(n_tr, n_tr + n_va)), shuffle=False, **loader_cfg),
        DataLoader(Subset(dataset, range(n_tr + n_va, n)),    shuffle=False, **loader_cfg),
    )

# ─────────────────────────────── config helpers ───────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_dataset_config(cfg: dict, dataset_path: str) -> dict:
    """
    Load a dataset config and replace the 'data' section in cfg.
    Allows decoupling dataset configs from model configs:
        python train.py --config configs/model/cQLSTM.yaml \\
                        --dataset configs/dataset/engineered.yaml
    """
    cfg = copy.deepcopy(cfg)
    ds = load_config(dataset_path)
    if "data" not in ds:
        raise ValueError(f"Dataset config '{dataset_path}' must contain a top-level 'data' key.")
    cfg["data"] = ds["data"]
    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Apply dot-notation CLI overrides to a nested config dict.
    e.g. ["model.name", "LSTM", "training.max_epochs", "60"]
    """
    cfg = copy.deepcopy(cfg)
    it  = iter(overrides)
    for key in it:
        val_str = next(it)
        # parse value: try int → float → bool → keep string
        for cast in (int, float, lambda x: {"true": True, "false": False}[x.lower()]):
            try:
                val = cast(val_str)
                break
            except (ValueError, KeyError):
                val = val_str

        parts  = key.lstrip("-").split(".")
        target = cfg
        for p in parts[:-1]:
            target = target[p]
        target[parts[-1]] = val
    return cfg

# ─────────────────────────────── main ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train a model on the wildfire dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Model configs (configs/model/):\n"
               "  LSTM.yaml  cLSTM.yaml  QLSTM.yaml  cQLSTM.yaml (default)\n"
               "  SSM.yaml   QSSM.yaml   cQSSM.yaml\n"
               "  DeltaNet.yaml  QDeltaNet.yaml\n\n"
               "Dataset configs (configs/dataset/):\n"
               "  preprocessed.yaml  (9 features, default)\n"
               "  engineered.yaml    (14 features, requires wildfire_engineered.csv)\n\n"
               "Examples:\n"
               "  python train.py --config configs/model/cQLSTM.yaml\n"
               "  python train.py --config configs/model/LSTM.yaml "
               "--dataset configs/dataset/engineered.yaml\n",
    )
    parser.add_argument("--config", default="configs/model/cQLSTM.yaml",
                        help="Path to model config YAML (default: configs/model/cQLSTM.yaml)")
    parser.add_argument("--dataset", default=None,
                        help="Optional dataset config YAML; overrides the 'data' section "
                             "in the model config (e.g. configs/dataset/engineered.yaml)")
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg = merge_dataset_config(cfg, args.dataset)
    cfg = apply_overrides(cfg, overrides)

    # ── run name ──────────────────────────────────────────────────────────
    if not cfg["logging"]["run_name"]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg["logging"]["run_name"] = f"{cfg['model']['name']}_{ts}"

    seed_everything(cfg["training"]["seed"], workers=True)
    torch.set_float32_matmul_precision("high")

    # ── header ────────────────────────────────────────────────────────────
    task = cfg["data"].get("task", "regression")
    task_label = {
        "regression":     "Wildfire Risk / Premium Regression",
        "classification": "Wildfire Occurrence Classification",
    }.get(task, task)

    print("=" * 60)
    print(f"  {cfg['model']['name']}  —  {task_label}")
    print(f"  Run: {cfg['logging']['run_name']}")
    print("=" * 60)

    # ── data ──────────────────────────────────────────────────────────────
    data_path = ROOT / cfg["data"]["path"]
    if not data_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {data_path}\n"
            "Run  python preprocess_wildfire.py  first."
        )
    df = pd.read_csv(data_path)
    dataset, scaler_X, scaler_y = build_dataset(df, cfg)
    train_loader, val_loader, test_loader = make_loaders(dataset, cfg)

    # ── model ──────────────────────────────────────────────────────────────
    model    = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model  : {cfg['model']['name']}  ({n_params:,} params)")

    # ── loss / optimizer / scheduler ──────────────────────────────────────
    t = cfg["training"]
    loss_fn = (nn.BCEWithLogitsLoss() if task == "classification"
               else nn.SmoothL1Loss())
    optimizer = optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = scheduler_with_warmup(
        scheduler=ls.CosineAnnealingLR(optimizer=optimizer, T_max=t["max_epochs"]),
        warmup_epochs=t["warmup_epochs"],
        start_factor=t["warmup_start_factor"],
    )
    lit = LitModel(
        model=model,
        output_size=cfg["data"]["n_future"],
        criterion=loss_fn,
        optimizer=[optimizer],
        scheduler=[scheduler],
        task=task,
    )

    # ── logger ────────────────────────────────────────────────────────────
    log_dir = ROOT / cfg["logging"]["log_dir"]
    logger  = TensorBoardLogger(
        save_dir=str(log_dir),
        name=cfg["logging"]["run_name"],
    )

    # ── save resolved config alongside logs ───────────────────────────────
    Path(logger.log_dir).mkdir(parents=True, exist_ok=True)
    config_save = Path(logger.log_dir) / "run_config.yaml"
    with open(config_save, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # ── callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath=logger.log_dir,
            filename="best-epoch={epoch:02d}-valloss={val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=t["early_stop_patience"],
            mode="min",
            verbose=False,
        ),
    ]

    # ── trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=t["max_epochs"],
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    print(f"\n  Training …  (logs → {logger.log_dir})")
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("\n  Test evaluation …")
    trainer.test(lit, dataloaders=test_loader, verbose=False)

    # ── per-epoch summary ─────────────────────────────────────────────────
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(logger.log_dir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    print(f"\n{'─'*50}")
    print(f"  Final metrics  (epoch {trainer.current_epoch})")
    print(f"{'─'*50}")
    task_metrics = (("loss", "accuracy", "f1")
                    if task == "classification"
                    else ("loss", "rmse", "mae", "r2"))
    for split in ("val", "test"):
        for metric in task_metrics:
            tag = f"{split}/{metric}"
            if tag in tags:
                v = ea.Scalars(tag)[-1].value
                print(f"  {tag:<18} {v:.4f}")

    print(f"\n  Config  →  {config_save}")
    print(f"  TBoard  →  tensorboard --logdir {log_dir}")
    print(f"  Eval    →  python scripts/eval.py --run {logger.log_dir}")


if __name__ == "__main__":
    main()
