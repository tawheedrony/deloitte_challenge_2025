"""
eval_wildfire.py  —  evaluate a finished training run
======================================================
Loads a checkpoint from a lightning_logs run directory, rebuilds the test
split with the same seed, and reports metrics in both normalized scale and
the original dollar scale.

Usage:
    python eval_wildfire.py --run lightning_logs/{run_name}/version_N
    python eval_wildfire.py --run lightning_logs/cQLSTM_20250323_120000/version_0

    # compare multiple runs side by side
    python eval_wildfire.py \\
        --run lightning_logs/LSTM_*/version_0 \\
        --run lightning_logs/cQLSTM_*/version_0

Output:
    • Per-run metric table (scaled + dollar)
    • Learning-curve summary (best/final val loss per run)
    • If multiple runs: ranked comparison table
"""

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                              mean_squared_error, r2_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader, Dataset, Subset

from modules.model import LitModel
from modules.scheduler import scheduler_with_warmup

# ─────────────────────────────── dataset (mirrors train) ──────────────────

class WildfireDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def rebuild_test_loader(cfg: dict):
    """Rebuild the test DataLoader and scaler_y using the run config."""
    d        = cfg["data"]
    features = d["features"]
    target   = d["target"]
    t_steps  = d["time_steps"]
    split    = d["split"]
    task     = d.get("task", "regression")

    data_path = ROOT / d["path"]
    df = pd.read_csv(data_path).sort_values(["ZIP", "Year"])

    years       = sorted(df["Year"].unique())
    input_years = years[:t_steps]
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
        scaler_y = None
    else:
        scaler_y = StandardScaler()
        if d.get("log_target", False):
            y = np.log1p(y)
        y = scaler_y.fit_transform(y)

    n_tr = int(N * split[0])
    n_va = int(N * split[1])
    test_set = Subset(WildfireDataset(X, y), range(n_tr + n_va, N))

    loader = DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available(),
    )
    return loader, scaler_y, N - n_tr - n_va

# ─────────────────────────────── model loader ─────────────────────────────

def load_model_from_run(run_dir: Path) -> tuple[LitModel, dict]:
    """Load the best checkpoint from a run directory. Returns (lit, cfg)."""
    config_path = run_dir / "run_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"run_config.yaml not found in {run_dir}\n"
            "This run was not created by the unified train_wildfire.py."
        )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # find best checkpoint (may be in a subdir if run with old naming)
    ckpts = sorted(run_dir.glob("best-*.ckpt")) or sorted(run_dir.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    ckpt_path = ckpts[-1]

    # rebuild model
    m = cfg["model"]
    name = m["name"]
    if name == "LSTM":
        from models.LSTM import LSTM
        model = LSTM(input_size=cfg["data"]["input_size"], hidden_size=m["hidden_size"])
    elif name == "cLSTM":
        from models.cLSTM import cLSTM
        model = cLSTM(input_size=cfg["data"]["input_size"], hidden_size=m["hidden_size"],
                      decay_rate=m["decay_rate"])
    elif name == "QLSTM":
        from models.QLSTM import QLSTM
        model = QLSTM(input_size=cfg["data"]["input_size"], hidden_size=m["hidden_size"],
                      n_qubits=m["n_qubits"], n_qlayers=m["n_qlayers"])
    elif name == "cQLSTM":
        from models.cQLSTM import cQLSTM
        model = cQLSTM(input_size=cfg["data"]["input_size"], hidden_size=m["hidden_size"],
                       decay_rate=m["decay_rate"], n_qubits=m["n_qubits"],
                       n_qlayers=m["n_qlayers"], n_esteps=m["n_esteps"])
    elif name == "SSM":
        from models.SSM import SSM
        model = SSM(input_size=cfg["data"]["input_size"], hidden_size=m["hidden_size"],
                    g_min=m["g_min"], g_max=m["g_max"])
    elif name == "QSSM":
        from models.QSSM import QuantumSSM
        model = QuantumSSM(input_size=cfg["data"]["input_size"], hidden_size=m["hidden_size"],
                           n_qubits=m["n_qubits"], n_qlayers=m["n_qlayers"],
                           n_esteps=m["n_esteps"], g_min=m["g_min"], g_max=m["g_max"])
    elif name == "cQSSM":
        from models.cQSSM import cQSSM
        model = cQSSM(input_size=cfg["data"]["input_size"], hidden_size=m["hidden_size"],
                      n_qubits=m["n_qubits"], n_qlayers=m["n_qlayers"],
                      n_esteps=m["n_esteps"], decay_rate=m["decay_rate"],
                      g_min=m["g_min"], g_max=m["g_max"])
    elif name == "DeltaNet":
        from models.DeltaNet import DeltaNet
        model = DeltaNet(input_size=cfg["data"]["input_size"], hidden_size=m["hidden_size"],
                         expand=m.get("expand", 1), neg_eigen=m.get("neg_eigen", False),
                         chunk_size=m.get("chunk_size", 0))
    elif name == "QDeltaNet":
        from models.QDeltaNet import QDeltaNet
        # auto-detect legacy shared-VQC architecture from checkpoint keys
        _sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)["state_dict"]
        _shared_vqc = "model.entry.weight" in _sd
        model = QDeltaNet(input_size=cfg["data"]["input_size"], hidden_size=m["hidden_size"],
                          n_qubits=m["n_qubits"], n_qlayers=m["n_qlayers"],
                          n_esteps=m["n_esteps"], neg_eigen=m.get("neg_eigen", False),
                          chunk_size=m.get("chunk_size", 0), shared_vqc=_shared_vqc)
    else:
        raise ValueError(f"Unknown model: {name}")

    task = cfg["data"].get("task", "regression")
    t = cfg["training"]
    optimizer = optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    loss_fn = (nn.BCEWithLogitsLoss() if task == "classification"
               else nn.SmoothL1Loss())
    lit = LitModel(
        model=model,
        output_size=cfg["data"]["n_future"],
        criterion=loss_fn,
        optimizer=[optimizer],
        scheduler=None,
        task=task,
    )
    lit.load(str(ckpt_path), verbose=False)
    lit.eval()

    return lit, cfg, ckpt_path

# ─────────────────────────────── metrics ──────────────────────────────────

def compute_metrics(lit: LitModel, test_loader, scaler_y,
                    log_target: bool = False, task: str = "regression") -> dict:
    all_pred, all_true = [], []
    with torch.no_grad():
        for X, y in test_loader:
            all_pred.append(lit(X).numpy())
            all_true.append(y.numpy())

    pred_s = np.vstack(all_pred)
    true_s = np.vstack(all_true)

    if task == "classification":
        # convert logits → probabilities → binary predictions
        prob = 1 / (1 + np.exp(-pred_s.ravel()))
        pred_bin = (prob >= 0.5).astype(int)
        true_bin = true_s.ravel().astype(int)
        acc  = float(accuracy_score(true_bin, pred_bin))
        f1   = float(f1_score(true_bin, pred_bin, zero_division=0))
        # AUROC: requires both classes present in test set
        try:
            auc = float(roc_auc_score(true_bin, prob))
        except ValueError:
            auc = float("nan")
        n_pos = int(true_bin.sum())
        n_neg = int((1 - true_bin).sum())
        return {
            "accuracy":    acc,
            "f1":          f1,
            "auroc":       auc,
            "n_positive":  n_pos,
            "n_negative":  n_neg,
            "prevalence":  n_pos / (n_pos + n_neg),
        }

    # ── regression path ──────────────────────────────────────────────────
    r2_s   = r2_score(true_s, pred_s)
    rmse_s = float(np.sqrt(mean_squared_error(true_s, pred_s)))
    mae_s  = float(mean_absolute_error(true_s, pred_s))

    # original-scale metrics
    pred = scaler_y.inverse_transform(pred_s)
    true = scaler_y.inverse_transform(true_s)
    if log_target:
        pred = np.expm1(pred)
        true = np.expm1(true)
    mask = true.ravel() > 10_000

    r2    = r2_score(true, pred)
    rmse  = float(np.sqrt(mean_squared_error(true, pred)))
    mae   = float(mean_absolute_error(true, pred))
    mdape = float(np.median(np.abs((true[mask] - pred[mask]) / true[mask])) * 100)

    return {
        "scaled/r2":      r2_s,
        "scaled/rmse":    rmse_s,
        "scaled/mae":     mae_s,
        "dollar/r2":      r2,
        "dollar/rmse":    rmse,
        "dollar/mae":     mae,
        "dollar/mdape":   mdape,
        "pred_mean":      float(pred.mean()),
        "pred_std":       float(pred.std()),
        "true_mean":      float(true.mean()),
        "true_std":       float(true.std()),
    }


def learning_curve_summary(run_dir: Path) -> dict:
    """Extract best/final val metrics from TensorBoard logs."""
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    out  = {}
    for metric in ("val/loss", "val/rmse", "val/mae", "val/r2",
                   "train/loss", "train/r2"):
        if metric not in tags:
            continue
        vals = [e.value for e in ea.Scalars(metric)]
        minimize = "loss" in metric or "rmse" in metric or "mae" in metric
        out[metric] = {
            "ep1":   vals[0],
            "final": vals[-1],
            "best":  min(vals) if minimize else max(vals),
            "epochs": len(vals),
        }
    return out

# ─────────────────────────────── report ───────────────────────────────────

def print_run_report(run_dir: Path, metrics: dict, lc: dict, cfg: dict, ckpt: Path):
    m    = cfg["model"]
    task = cfg["data"].get("task", "regression")
    n_params = sum(p.numel() for p in _dummy_model(cfg, ckpt).parameters())

    w = 55
    print(f"\n{'═'*w}")
    print(f"  Run   : {run_dir}")
    print(f"  Model : {m['name']}  ({n_params:,} params)  task={task}")
    print(f"  Ckpt  : {ckpt.name}")
    print(f"{'─'*w}")

    print(f"  Learning curves")
    for tag, d in lc.items():
        print(f"    {tag:<22}  ep1={d['ep1']:.4f}  final={d['final']:.4f}"
              f"  best={d['best']:.4f}  ({d['epochs']} epochs)")

    print(f"{'─'*w}")
    if task == "classification":
        print(f"  Test set — classification metrics")
        print(f"    Accuracy   {metrics['accuracy']:>8.4f}")
        print(f"    F1 Score   {metrics['f1']:>8.4f}")
        print(f"    AUROC      {metrics['auroc']:>8.4f}")
        print(f"    Positive   {metrics['n_positive']:>8d}  "
              f"({metrics['prevalence']:.1%} prevalence)")
        print(f"    Negative   {metrics['n_negative']:>8d}")
    else:
        print(f"  Test set — scaled space")
        print(f"    R²   {metrics['scaled/r2']:>10.4f}")
        print(f"    RMSE {metrics['scaled/rmse']:>10.4f}")
        print(f"    MAE  {metrics['scaled/mae']:>10.4f}")
        print(f"{'─'*w}")
        print(f"  Test set — original scale")
        print(f"    R²        {metrics['dollar/r2']:>10.4f}")
        print(f"    RMSE      {metrics['dollar/rmse']:>14,.4f}")
        print(f"    MAE       {metrics['dollar/mae']:>14,.4f}")
        print(f"    MdAPE     {metrics['dollar/mdape']:>9.1f}%")
        print(f"    Pred mean {metrics['pred_mean']:>14,.4f}  ±{metrics['pred_std']:,.4f}")
        print(f"    True mean {metrics['true_mean']:>14,.4f}  ±{metrics['true_std']:,.4f}")
    print(f"{'═'*w}")


def _dummy_model(cfg: dict, ckpt: Path = None):
    """Instantiate model without quantum state for param count only."""
    m = cfg["model"]
    name = m["name"]
    if name == "LSTM":
        from models.LSTM import LSTM
        return LSTM(cfg["data"]["input_size"], m["hidden_size"])
    if name == "cLSTM":
        from models.cLSTM import cLSTM
        return cLSTM(cfg["data"]["input_size"], m["hidden_size"])
    if name == "QLSTM":
        from models.QLSTM import QLSTM
        return QLSTM(cfg["data"]["input_size"], m["hidden_size"], n_qubits=m["n_qubits"])
    if name == "cQLSTM":
        from models.cQLSTM import cQLSTM
        return cQLSTM(cfg["data"]["input_size"], m["hidden_size"], n_qubits=m["n_qubits"])
    if name == "SSM":
        from models.SSM import SSM
        return SSM(cfg["data"]["input_size"], m["hidden_size"])
    if name == "QSSM":
        from models.QSSM import QuantumSSM
        return QuantumSSM(cfg["data"]["input_size"], m["hidden_size"], n_qubits=m["n_qubits"])
    if name == "cQSSM":
        from models.cQSSM import cQSSM
        return cQSSM(cfg["data"]["input_size"], m["hidden_size"], n_qubits=m["n_qubits"])
    if name == "DeltaNet":
        from models.DeltaNet import DeltaNet
        return DeltaNet(cfg["data"]["input_size"], m["hidden_size"],
                        expand=m.get("expand", 1), neg_eigen=m.get("neg_eigen", False))
    if name == "QDeltaNet":
        from models.QDeltaNet import QDeltaNet
        _shared_vqc = False
        if ckpt is not None:
            _sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)["state_dict"]
            _shared_vqc = "model.entry.weight" in _sd
        return QDeltaNet(cfg["data"]["input_size"], m["hidden_size"],
                         n_qubits=m["n_qubits"], shared_vqc=_shared_vqc)
    raise ValueError(name)

# ─────────────────────────────── main ─────────────────────────────────────

def resolve_run_dirs(raw_runs: list[str]) -> list[Path]:
    """Expand globs and normalise to version dirs."""
    dirs = []
    for pattern in raw_runs:
        expanded = glob.glob(pattern)
        if not expanded:
            expanded = [pattern]
        for p in expanded:
            p = Path(p)
            if (p / "run_config.yaml").exists():
                dirs.append(p)
            else:
                # user may have pointed at the run name dir, not version dir
                version_dirs = sorted(p.glob("version_*"))
                if version_dirs:
                    dirs.append(version_dirs[-1])
                else:
                    print(f"[warn] No run_config.yaml found under {p}, skipping.")
    return dirs


def main():
    parser = argparse.ArgumentParser(description="Evaluate finished wildfire training runs.")
    parser.add_argument("--run", action="append", required=True, metavar="DIR",
                        help="Path to lightning_logs run version dir. "
                             "Repeat for multiple runs. Globs accepted.")
    args = parser.parse_args()

    run_dirs = resolve_run_dirs(args.run)
    if not run_dirs:
        print("No valid run directories found.")
        sys.exit(1)

    all_rows = []
    for run_dir in run_dirs:
        print(f"\nLoading  {run_dir} …")
        try:
            lit, cfg, ckpt = load_model_from_run(run_dir)
        except FileNotFoundError as e:
            print(f"  [error] {e}")
            continue

        test_loader, scaler_y, n_test = rebuild_test_loader(cfg)
        print(f"  Test samples: {n_test}")

        log_target = cfg["data"].get("log_target", False)
        task       = cfg["data"].get("task", "regression")
        metrics = compute_metrics(lit, test_loader, scaler_y,
                                  log_target=log_target, task=task)
        lc      = learning_curve_summary(run_dir)
        print_run_report(run_dir, metrics, lc, cfg, ckpt)

        # extract timestamp suffix from parent dir  e.g. "QDeltaNet_20260323_151836"
        import re as _re
        _ts_match = _re.search(r"_(\d{8}_\d{6})$", run_dir.parent.name)
        _ts = _ts_match.group(1) if _ts_match else None

        if task == "classification":
            row = {
                "Model":         cfg["model"]["name"],
                "_ts":           _ts,
                "Task":          "classification",
                "Run":           run_dir.name,
                "Epochs":        lc.get("val/loss", {}).get("epochs", "?"),
                "Best val/loss": lc.get("val/loss", {}).get("best", float("nan")),
                "Accuracy":      round(metrics["accuracy"], 4),
                "F1":            round(metrics["f1"], 4),
                "AUROC":         round(metrics["auroc"], 4),
            }
        else:
            row = {
                "Model":         cfg["model"]["name"],
                "_ts":           _ts,
                "Task":          "regression",
                "Run":           run_dir.name,
                "Epochs":        lc.get("val/loss", {}).get("epochs", "?"),
                "Best val/loss": lc.get("val/loss", {}).get("best", float("nan")),
                "Test R²":       round(metrics["dollar/r2"], 4),
                "RMSE":          round(metrics["dollar/rmse"], 4),
                "MAE":           round(metrics["dollar/mae"], 4),
                "MdAPE (%)":     round(metrics["dollar/mdape"], 1),
            }
        all_rows.append(row)

    if len(all_rows) > 1:
        # append time suffix to model names that appear more than once
        from collections import Counter as _Counter
        name_counts = _Counter(r["Model"] for r in all_rows)
        for row in all_rows:
            if name_counts[row["Model"]] > 1 and row["_ts"]:
                row["Model"] = f"{row['Model']} ({row['_ts'].split('_')[1]})"
        for row in all_rows:
            del row["_ts"]

        df = pd.DataFrame(all_rows).set_index("Model")
        sort_col = "Accuracy" if "Accuracy" in df.columns else "Test R²"
        print(f"\n{'═'*70}")
        print(f"  COMPARISON SUMMARY  (ranked by {sort_col})")
        print(f"{'═'*70}")
        print(df.sort_values(sort_col, ascending=False).to_string())
        print(f"{'═'*70}")


if __name__ == "__main__":
    main()
