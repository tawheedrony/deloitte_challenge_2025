"""
scripts/task1a_visualize.py  —  Task 1A: Wildfire Risk Prediction Visualization
================================================================================
Plots predicted vs actual wildfire risk scores as time-series-style line plots,
sorted by actual value, so you can see how well the model tracks the true signal.

Supports loading from:
  - NN runs (lightning_logs)
  - CSV files (output/task1a/predictions/)

Usage:
    # from saved CSVs (one model per plot)
    python scripts/task1a_visualize.py \\
        --from-csv output/task1a/predictions/cQLSTM_4q.csv \\
        --out output/task1a_cQLSTM_4q.png

    # multiple models overlaid
    python scripts/task1a_visualize.py \\
        --from-csv output/task1a/predictions/cQLSTM_4q.csv \\
        --from-csv output/task1a/predictions/LSTM.csv \\
        --out output/task1a_comparison.png

    # from NN runs
    python scripts/task1a_visualize.py \\
        --run "lightning_logs/task1a_cQLSTM_4q_*/version_0" \\
        --out output/task1a_cQLSTM.png

    # mix CSV + NN runs
    python scripts/task1a_visualize.py \\
        --from-csv output/task1a/predictions/cQLSTM_4q.csv \\
        --run "lightning_logs/task1a_LSTM_*/version_0" \\
        --out output/task1a_mixed.png
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
from sklearn.metrics import (accuracy_score, f1_score,
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


def predict_nn(lit, loader, scaler_y, task):
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


def load_from_csv(csv_paths):
    """Load predictions from CSV files (output by task1a.py)."""
    runs = []
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"  [skip] {csv_path} not found"); continue

        df = pd.read_csv(csv_path)
        if "predicted" not in df.columns or "actual" not in df.columns:
            print(f"  [skip] {csv_path} missing predicted/actual columns"); continue

        # these are in scaled space — we don't have the scaler, so plot as-is
        pred = df["predicted"].values
        true = df["actual"].values
        name = csv_path.stem

        metrics = {
            "r2": r2_score(true, pred),
            "rmse": float(np.sqrt(mean_squared_error(true, pred))),
            "mae": float(mean_absolute_error(true, pred)),
        }
        if "ZIP" in df.columns:
            metrics["zips"] = df["ZIP"].values

        print(f"  {name:<25} R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}")
        runs.append((name, pred, true, metrics))

    return runs


# ─────────────────────────── plotting ────────────────────────────────────

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4",
          "#795548", "#607D8B", "#E91E63", "#009688"]


def plot_timeseries(runs_data, out_path, title="Task 1A — Wildfire Risk Score"):
    """
    Time-series style plot: actual (black) + predicted (colored) sorted by actual value.
    One subplot per model.
    """
    n = len(runs_data)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    fig.suptitle(f"{title} — Predicted vs Actual (Test Set)",
                 fontsize=14, fontweight="bold", y=1.02)

    for idx, (name, pred, true, m) in enumerate(runs_data):
        ax = axes[idx // cols][idx % cols]

        # sort by actual value for clean visualization
        order = np.argsort(true)
        true_sorted = true[order]
        pred_sorted = pred[order]
        x = np.arange(len(true_sorted))

        ax.plot(x, true_sorted, color="black", linewidth=1.2, alpha=0.8, label="Actual")
        ax.plot(x, pred_sorted, color=COLORS[idx % len(COLORS)], linewidth=1.0,
                alpha=0.7, label="Predicted")

        # fill error region
        ax.fill_between(x, true_sorted, pred_sorted, alpha=0.15,
                         color=COLORS[idx % len(COLORS)])

        ax.set_title(f"{name}  (R²={m['r2']:.3f}, RMSE={m['rmse']:.3f})", fontsize=11)
        ax.set_xlabel("ZIP codes (sorted by actual)", fontsize=10)
        ax.set_ylabel("Risk Score", fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)

    # hide empty subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"Saved  ->  {out_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_by_zip(runs_data, out_path, data_path=None, target_col="avg_fire_risk_score",
                n_zips=4, selected_zips=None, title="Task 1A — Per-ZIP Time Series"):
    """
    For selected ZIP codes, plot historical target values (2018-2020) + predicted
    vs actual 2021. One subplot per ZIP, each model as a different colored line.
    """
    # load historical data
    if data_path is None:
        data_path = ROOT / "data" / "wildfire_engineered.csv"
    if not Path(data_path).exists():
        data_path = ROOT / "data" / "wildfire_preprocessed.csv"
    raw_df = pd.read_csv(data_path).sort_values(["ZIP", "Year"])
    years = sorted(raw_df["Year"].unique())
    tgt_year = years[-1]
    hist_years = years[:-1]

    # get ZIPs from the first model that has them
    first_zips = None
    for name, pred, true, m in runs_data:
        if "zips" in m:
            first_zips = m["zips"]
            break

    if first_zips is None:
        print("  [skip] No ZIP info in prediction CSVs. Re-run task1a.py to include ZIP column.")
        return

    if selected_zips is None:
        # pick n_zips spread across the risk range
        zip_true = {z: t for z, t in zip(first_zips, runs_data[0][2])}
        sorted_zips = sorted(zip_true.keys(), key=lambda z: zip_true[z])
        n = len(sorted_zips)
        indices = [int(i * (n - 1) / (n_zips - 1)) for i in range(n_zips)]
        selected_zips = [sorted_zips[i] for i in indices]

    n_zips = len(selected_zips)

    fig, axes = plt.subplots(1, n_zips, figsize=(6 * n_zips, 5), squeeze=False)
    fig.suptitle(f"{title} — Selected ZIP Codes", fontsize=14, fontweight="bold", y=1.02)

    for col_idx, zip_code in enumerate(selected_zips):
        ax = axes[0][col_idx]

        # historical values from raw data
        z_df = raw_df[raw_df["ZIP"] == zip_code].set_index("Year").sort_index()
        if target_col in z_df.columns:
            hist_vals = [z_df.loc[yr, target_col] if yr in z_df.index else np.nan
                         for yr in hist_years]
            ax.plot(hist_years, hist_vals, "ko-", linewidth=1.5, markersize=6,
                    label="Historical", zorder=5)

        # predicted vs actual for 2021 from each model
        for m_idx, (name, pred, true, metrics) in enumerate(runs_data):
            if "zips" not in metrics:
                continue
            zips = metrics["zips"]
            if zip_code in zips:
                z_i = list(zips).index(zip_code)
                color = COLORS[m_idx % len(COLORS)]
                ax.plot(tgt_year, true[z_i], "s", color="black", markersize=10, zorder=6)
                ax.plot(tgt_year, pred[z_i], "^", color=color, markersize=10,
                        label=f"{name} pred", zorder=7)

        ax.set_title(f"ZIP {zip_code}", fontsize=11)
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel(target_col.replace("_", " ").title(), fontsize=10)
        ax.set_xticks(years)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--run", action="append", default=None, metavar="DIR",
                        help="NN run version dir (repeat or glob)")
    parser.add_argument("--from-csv", action="append", default=None, metavar="CSV",
                        help="Prediction CSV from task1a.py. Repeat for multiple. Globs accepted.")
    parser.add_argument("--out", default=None, metavar="FILE",
                        help="Save plot to file (PNG/PDF)")
    parser.add_argument("--by-zip", action="store_true",
                        help="Plot per-ZIP time series (historical + predicted 2021)")
    parser.add_argument("--zips", nargs="+", type=int, default=None,
                        help="Specific ZIP codes for --by-zip (default: auto-select 4)")
    args = parser.parse_args()

    if not args.run and not args.from_csv:
        parser.error("Provide at least one of --run or --from-csv")

    runs_data = []

    # ── load from CSV ────────────────────────────────────────────────
    if args.from_csv:
        csv_paths = []
        for pattern in args.from_csv:
            expanded = glob.glob(pattern)
            csv_paths.extend(expanded if expanded else [pattern])
        runs_data.extend(load_from_csv(csv_paths))

    # ── load from NN runs ────────────────────────────────────────────
    if args.run:
        run_dirs = resolve_dirs(args.run)
        for run_dir in run_dirs:
            print(f"Loading  {run_dir} ...")
            try:
                lit, cfg = load_run(run_dir)
            except (FileNotFoundError, Exception) as e:
                print(f"  [skip] {e}"); continue

            loader, scaler_y, task = rebuild_test(cfg)

            if task == "classification":
                print(f"  [skip] classification run — use separate plot")
                continue

            pred, true = predict_nn(lit, loader, scaler_y, task)
            name = cfg["model"]["name"]

            # add qubit info from run name
            run_name = run_dir.parent.name
            for tag in ["_4q", "_8q", "_12q"]:
                if tag in run_name:
                    name = f"{name}{tag}"
                    break
            if "reupload" in run_name:
                name = f"{name}_reupload"

            metrics = {
                "r2": r2_score(true, pred),
                "rmse": float(np.sqrt(mean_squared_error(true, pred))),
                "mae": float(mean_absolute_error(true, pred)),
            }
            print(f"  {name:<25} R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}")
            runs_data.append((name, pred, true, metrics))

    if not runs_data:
        print("Nothing to plot."); return

    if args.by_zip:
        plot_by_zip(runs_data, args.out, selected_zips=args.zips)
    else:
        plot_timeseries(runs_data, args.out)


if __name__ == "__main__":
    main()
