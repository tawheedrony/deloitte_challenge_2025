"""
scripts/task2_visualize.py  —  Task 2: Insurance Premium Prediction Visualization
==================================================================================
Plots predicted vs actual insurance premiums for Task 2 runs.
Supports both baseline and Task 1A-chained runs.

Usage:
    # all Task 2 runs
    python scripts/task2_visualize.py --run "lightning_logs/*/version_0"

    # compare baseline vs chained
    python scripts/task2_visualize.py \\
        --run "lightning_logs/task2_*baseline*/version_0" \\
        --run "lightning_logs/task2_*chained*/version_0"

    # save to file
    python scripts/task2_visualize.py --run "lightning_logs/*/version_0" \\
                                      --out output/task2_premium_prediction.png
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    if task != "regression":
        return None, None  # skip non-regression runs

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
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X.reshape(N * T, F)).reshape(N, T, F)

    if d.get("log_target", False):
        y = np.log1p(y)
    y = scaler_y.fit_transform(y)

    split = d["split"]
    n_tr = int(N * split[0]); n_va = int(N * split[1])
    loader = DataLoader(Subset(WildfireDataset(X, y), range(n_tr + n_va, N)),
                        batch_size=256, shuffle=False, num_workers=2)
    return loader, scaler_y


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


def predict(lit, loader, scaler_y, log_target=False):
    preds, trues = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.append(lit(X).numpy())
            trues.append(y.numpy())
    pred = scaler_y.inverse_transform(np.vstack(preds))
    true = scaler_y.inverse_transform(np.vstack(trues))
    if log_target:
        pred = np.expm1(pred)
        true = np.expm1(true)
    return pred.ravel(), true.ravel()


# ─────────────────────────── plotting ────────────────────────────────────

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4",
          "#795548", "#607D8B"]


def plot_all(runs_data, out_path):
    n = len(runs_data)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Task 2 — Insurance Premium Prediction (Test Set)",
                 fontsize=14, fontweight="bold")

    ax_scatter, ax_rmse, ax_r2 = axes

    # scatter
    all_true = runs_data[0][2]
    lo, hi = all_true.min() / 1e6, all_true.max() / 1e6
    ax_scatter.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="Perfect")

    for i, (name, pred, true, m) in enumerate(runs_data):
        c = COLORS[i % len(COLORS)]
        ax_scatter.scatter(true / 1e6, pred / 1e6, alpha=0.35, s=18, color=c,
                           label=f"{name}  (R²={m['r2']:.3f})")

    ax_scatter.set_xlabel("Actual Premium ($M)", fontsize=11)
    ax_scatter.set_ylabel("Predicted Premium ($M)", fontsize=11)
    ax_scatter.set_title("Predicted vs Actual", fontsize=12)
    ax_scatter.legend(fontsize=9, loc="upper left")

    # RMSE bars
    names = [r[0] for r in runs_data]
    rmses = [r[3]["rmse"] / 1e6 for r in runs_data]
    bars = ax_rmse.bar(names, rmses, color=COLORS[:n], edgecolor="white", width=0.5)
    ax_rmse.bar_label(bars, fmt="$%.2fM", padding=3, fontsize=9)
    ax_rmse.set_ylabel("RMSE ($M)", fontsize=11)
    ax_rmse.set_title("Test RMSE  (lower is better)", fontsize=12)
    ax_rmse.tick_params(axis="x", rotation=20)

    # R² bars
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
        description="Task 2: Visualize insurance premium predictions.")
    parser.add_argument("--run", action="append", required=True, metavar="DIR")
    parser.add_argument("--out", default=None, metavar="FILE",
                        help="Save plot to file (PNG/PDF)")
    args = parser.parse_args()

    run_dirs = resolve_dirs(args.run)
    if not run_dirs:
        print("No valid run directories found."); return

    runs_data = []
    for run_dir in run_dirs:
        print(f"Loading  {run_dir} ...")
        try:
            lit, cfg = load_run(run_dir)
        except (FileNotFoundError, Exception) as e:
            print(f"  [skip] {e}"); continue

        # only process regression (premium) runs
        task = cfg["data"].get("task", "regression")
        target = cfg["data"].get("target", "")
        if task != "regression" or target not in ("earned_premium", ""):
            print(f"  [skip] not a premium regression run (task={task}, target={target})")
            continue

        loader, scaler_y = rebuild_test(cfg)
        if loader is None:
            continue

        log_target = cfg["data"].get("log_target", False)
        pred, true = predict(lit, loader, scaler_y, log_target)

        mask = true > 10_000
        metrics = {
            "r2": r2_score(true, pred),
            "rmse": float(np.sqrt(mean_squared_error(true, pred))),
            "mae": float(mean_absolute_error(true, pred)),
            "mdape": float(np.median(np.abs((true[mask] - pred[mask]) / true[mask])) * 100),
        }

        # label: include "chained" tag if present in run name
        name = cfg["model"]["name"]
        run_name = run_dir.parent.name
        if "chained" in run_name:
            name = f"{name} (chained)"
        elif "baseline" in run_name:
            name = f"{name} (baseline)"

        print(f"  {name:<20}  R²={metrics['r2']:.4f}  RMSE=${metrics['rmse']:,.0f}")
        runs_data.append((name, pred, true, metrics))

    if not runs_data:
        print("Nothing to plot."); return

    plot_all(runs_data, args.out)


if __name__ == "__main__":
    main()
