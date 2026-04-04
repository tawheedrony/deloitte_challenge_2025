"""
scripts/task1a_visualize.py  —  Task 1A: Wildfire Risk Prediction Visualization
================================================================================
Plots predicted vs actual wildfire risk scores and fire occurrence predictions
for Task 1A runs.

Usage:
    # all Task 1A runs
    python scripts/task1a_visualize.py --run "lightning_logs/*task1a*/version_0"

    # specific models
    python scripts/task1a_visualize.py \\
        --run lightning_logs/cQLSTM*task1a*/version_0 \\
        --run lightning_logs/LSTM*task1a*/version_0

    # save to file
    python scripts/task1a_visualize.py --run "lightning_logs/*task1a*/version_0" \\
                                       --out output/task1a_risk_prediction.png
"""

import argparse
import glob
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              mean_absolute_error, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from modules.model import LitModel
from train import build_model

# ─────────────────────────── dataset helpers ─────────────────────────────

class WildfireDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def rebuild_test(cfg):
    d = cfg["data"]
    task = d.get("task", "regression")
    df = pd.read_csv(ROOT / d["path"]).sort_values(["ZIP", "Year"])
    years = sorted(df["Year"].unique())
    in_years, tgt_year = years[:d["time_steps"]], years[-1]

    X_list, y_list = [], []
    for _, z in df.groupby("ZIP"):
        z = z.set_index("Year")
        if not all(yr in z.index for yr in in_years + [tgt_year]):
            continue
        xs = z.loc[in_years, d["features"]].values
        yv = z.loc[tgt_year, d["target"]]
        if np.isnan(xs).any() or np.isnan(yv):
            continue
        X_list.append(xs); y_list.append([yv])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    N, T, F = X.shape
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X.reshape(N * T, F)).reshape(N, T, F)

    if task == "classification":
        scaler_y = None
    else:
        scaler_y = StandardScaler()
        y = scaler_y.fit_transform(y)

    split = d["split"]
    n_tr = int(N * split[0]); n_va = int(N * split[1])
    test_idx = range(n_tr + n_va, N)
    loader = DataLoader(Subset(WildfireDataset(X, y), test_idx),
                        batch_size=256, shuffle=False, num_workers=2)
    return loader, scaler_y, task


def load_run(run_dir):
    cfg_path = run_dir / "run_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    ckpts = sorted(run_dir.glob("best-*.ckpt")) or sorted(run_dir.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint in {run_dir}")

    model = build_model(cfg)
    task = cfg["data"].get("task", "regression")
    t = cfg["training"]
    opt = optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.SmoothL1Loss()
    lit = LitModel(model=model, output_size=cfg["data"]["n_future"],
                   criterion=loss_fn, optimizer=[opt], scheduler=None, task=task)
    lit.load(str(ckpts[-1]), verbose=False)
    lit.eval()
    return lit, cfg


def predict(lit, loader, scaler_y, task):
    preds, trues = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.append(lit(X).numpy())
            trues.append(y.numpy())
    pred_s = np.vstack(preds)
    true_s = np.vstack(trues)

    if task == "classification":
        prob = 1 / (1 + np.exp(-pred_s.ravel()))
        return prob, true_s.ravel()

    if scaler_y is not None:
        pred = scaler_y.inverse_transform(pred_s).ravel()
        true = scaler_y.inverse_transform(true_s).ravel()
    else:
        pred, true = pred_s.ravel(), true_s.ravel()
    return pred, true


# ─────────────────────────── plotting ────────────────────────────────────

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4",
          "#795548", "#607D8B"]


def plot_regression(runs_data, out_path):
    """Plot for regression task (risk score prediction)."""
    n = len(runs_data)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Task 1A — Wildfire Risk Score Prediction (Test Set)",
                 fontsize=14, fontweight="bold")

    ax_scatter, ax_rmse, ax_r2 = axes

    # scatter: predicted vs actual
    all_true = runs_data[0][2]
    lo, hi = all_true.min(), all_true.max()
    ax_scatter.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="Perfect")

    for i, (name, pred, true, m) in enumerate(runs_data):
        c = COLORS[i % len(COLORS)]
        ax_scatter.scatter(true, pred, alpha=0.35, s=18, color=c,
                           label=f"{name}  (R²={m['r2']:.3f})")

    ax_scatter.set_xlabel("Actual Risk Score", fontsize=11)
    ax_scatter.set_ylabel("Predicted Risk Score", fontsize=11)
    ax_scatter.set_title("Predicted vs Actual", fontsize=12)
    ax_scatter.legend(fontsize=9, loc="upper left")

    # bar: RMSE
    names = [r[0] for r in runs_data]
    rmses = [r[3]["rmse"] for r in runs_data]
    bars = ax_rmse.bar(names, rmses, color=COLORS[:n], edgecolor="white", width=0.5)
    ax_rmse.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax_rmse.set_ylabel("RMSE", fontsize=11)
    ax_rmse.set_title("Test RMSE  (lower is better)", fontsize=12)
    ax_rmse.tick_params(axis="x", rotation=20)

    # bar: R²
    r2s = [r[3]["r2"] for r in runs_data]
    bars = ax_r2.bar(names, r2s, color=COLORS[:n], edgecolor="white", width=0.5)
    ax_r2.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax_r2.set_ylim(0, 1)
    ax_r2.set_ylabel("R²", fontsize=11)
    ax_r2.set_title("Test R²  (higher is better)", fontsize=12)
    ax_r2.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"Saved  ->  {out_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_classification(runs_data, out_path):
    """Plot for classification task (fire occurrence)."""
    n = len(runs_data)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Task 1A — Wildfire Occurrence Classification (Test Set)",
                 fontsize=14, fontweight="bold")

    ax_prob, ax_f1, ax_acc = axes

    # probability distribution
    for i, (name, prob, true, m) in enumerate(runs_data):
        c = COLORS[i % len(COLORS)]
        mask_pos = true == 1
        mask_neg = true == 0
        ax_prob.hist(prob[mask_pos], bins=30, alpha=0.4, color=c,
                     label=f"{name} (fire)", density=True)
        ax_prob.hist(prob[mask_neg], bins=30, alpha=0.2, color=c,
                     linestyle="--", density=True)
    ax_prob.set_xlabel("Predicted Probability", fontsize=11)
    ax_prob.set_ylabel("Density", fontsize=11)
    ax_prob.set_title("Probability Distribution", fontsize=12)
    ax_prob.legend(fontsize=8)
    ax_prob.axvline(0.5, color="red", linestyle="--", alpha=0.5)

    # bar: F1
    names = [r[0] for r in runs_data]
    f1s = [r[3]["f1"] for r in runs_data]
    bars = ax_f1.bar(names, f1s, color=COLORS[:n], edgecolor="white", width=0.5)
    ax_f1.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax_f1.set_ylim(0, 1)
    ax_f1.set_ylabel("F1 Score", fontsize=11)
    ax_f1.set_title("Test F1  (higher is better)", fontsize=12)
    ax_f1.tick_params(axis="x", rotation=20)

    # bar: Accuracy
    accs = [r[3]["accuracy"] for r in runs_data]
    bars = ax_acc.bar(names, accs, color=COLORS[:n], edgecolor="white", width=0.5)
    ax_acc.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax_acc.set_ylim(0, 1)
    ax_acc.set_ylabel("Accuracy", fontsize=11)
    ax_acc.set_title("Test Accuracy  (higher is better)", fontsize=12)
    ax_acc.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"Saved  ->  {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────── main ────────────────────────────────────────

def resolve_dirs(raw):
    dirs = []
    for pattern in raw:
        for p in glob.glob(pattern) or [pattern]:
            p = Path(p)
            if (p / "run_config.yaml").exists():
                dirs.append(p)
            else:
                for v in sorted(p.glob("version_*")):
                    if (v / "run_config.yaml").exists():
                        dirs.append(v)
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description="Task 1A: Visualize wildfire risk predictions.")
    parser.add_argument("--run", action="append", required=True, metavar="DIR")
    parser.add_argument("--out", default=None, metavar="FILE",
                        help="Save plot to file (PNG/PDF)")
    args = parser.parse_args()

    run_dirs = resolve_dirs(args.run)
    if not run_dirs:
        print("No valid run directories found."); return

    # separate regression and classification runs
    reg_data, cls_data = [], []

    for run_dir in run_dirs:
        print(f"Loading  {run_dir} ...")
        try:
            lit, cfg = load_run(run_dir)
        except (FileNotFoundError, Exception) as e:
            print(f"  [skip] {e}"); continue

        loader, scaler_y, task = rebuild_test(cfg)
        pred, true = predict(lit, loader, scaler_y, task)
        name = cfg["model"]["name"]

        if task == "classification":
            pred_bin = (pred >= 0.5).astype(int)
            true_bin = true.astype(int)
            metrics = {
                "accuracy": float(accuracy_score(true_bin, pred_bin)),
                "f1": float(f1_score(true_bin, pred_bin, zero_division=0)),
            }
            cls_data.append((name, pred, true, metrics))
            print(f"  {name:<12} [cls] Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}")
        else:
            metrics = {
                "r2": r2_score(true, pred),
                "rmse": float(np.sqrt(mean_squared_error(true, pred))),
                "mae": float(mean_absolute_error(true, pred)),
            }
            reg_data.append((name, pred, true, metrics))
            print(f"  {name:<12} [reg] R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}")

    if reg_data:
        out = args.out
        if out and cls_data:
            # use separate filenames for each task type
            base = Path(out)
            out = str(base.with_stem(base.stem + "_regression"))
        plot_regression(reg_data, out)

    if cls_data:
        out = args.out
        if out and reg_data:
            base = Path(out)
            out = str(base.with_stem(base.stem + "_classification"))
        plot_classification(cls_data, out or args.out)

    if not reg_data and not cls_data:
        print("Nothing to plot.")


if __name__ == "__main__":
    main()
