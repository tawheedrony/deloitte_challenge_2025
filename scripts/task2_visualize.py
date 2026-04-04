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


def plot_timeseries(runs_data, out_path, title="Task 2 — Insurance Premium Prediction"):
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

        ax.set_title(f"{name}  (R²={m['r2']:.3f}, RMSE=${m['rmse']:,.0f})", fontsize=11)
        ax.set_xlabel("ZIP codes (sorted by actual)", fontsize=10)
        ax.set_ylabel("Premium ($)", fontsize=10)
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


def plot_by_zip(runs_data, out_path, data_path=None, target_col="earned_premium",
                n_zips=4, selected_zips=None, title="Task 2 — Per-ZIP Premium"):
    """
    For selected ZIP codes, plot historical premium values (2018-2020) + predicted
    vs actual 2021. One subplot per ZIP, each model as a different colored marker.
    """
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
        print("  [skip] No ZIP info in prediction CSVs. Re-run task2.py to include ZIP column.")
        return

    if selected_zips is None:
        # pick n_zips spread across the premium range
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

        # actual 2021 value (black square)
        actual_plotted = False
        for m_idx, (name, pred, true, metrics) in enumerate(runs_data):
            if "zips" not in metrics:
                continue
            zips = metrics["zips"]
            if zip_code in zips:
                z_i = list(zips).index(zip_code)
                if not actual_plotted:
                    ax.plot(tgt_year, true[z_i], "ks", markersize=10,
                            label="Actual 2021", zorder=6)
                    # connect historical to actual
                    if target_col in z_df.columns and len(hist_vals) > 0:
                        ax.plot([hist_years[-1], tgt_year],
                                [hist_vals[-1], true[z_i]], "k--", linewidth=1, alpha=0.5)
                    actual_plotted = True
                color = COLORS[m_idx % len(COLORS)]
                ax.plot(tgt_year, pred[z_i], "^", color=color, markersize=10,
                        label=f"{name}", zorder=7)

        ax.set_title(f"ZIP {zip_code}", fontsize=11)
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel("Premium ($)", fontsize=10)
        ax.set_xticks(years)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="plain", axis="y")

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"Saved  ->  {out_path}")
    else:
        plt.show()
    plt.close(fig)


def load_from_csv(csv_paths):
    """Load predictions from CSV files (output by task2.py)."""
    runs_data = []
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"  [skip] {csv_path} not found"); continue

        df = pd.read_csv(csv_path)
        if "predicted" not in df.columns or "actual" not in df.columns:
            print(f"  [skip] {csv_path} missing predicted/actual columns"); continue

        pred = df["predicted"].values
        true = df["actual"].values
        name = csv_path.stem

        mask = true > 10_000
        metrics = {
            "r2": r2_score(true, pred),
            "rmse": float(np.sqrt(mean_squared_error(true, pred))),
            "mae": float(mean_absolute_error(true, pred)),
            "mdape": float(np.median(np.abs((true[mask] - pred[mask]) / true[mask])) * 100) if mask.any() else float("nan"),
        }
        if "ZIP" in df.columns:
            metrics["zips"] = df["ZIP"].values

        print(f"  {name:<20}  R²={metrics['r2']:.4f}  RMSE=${metrics['rmse']:,.0f}")
        runs_data.append((name, pred, true, metrics))

    return runs_data


def main():
    parser = argparse.ArgumentParser(
        description="Task 2: Visualize insurance premium predictions.")
    parser.add_argument("--run", action="append", default=None, metavar="DIR",
                        help="NN run version dir (repeat or glob)")
    parser.add_argument("--from-csv", action="append", default=None, metavar="CSV",
                        help="Prediction CSV from task2.py (e.g. output/task2/predictions/XGBoost.csv). "
                             "Repeat for multiple models. Globs accepted.")
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

    # ── load from CSV predictions ────────────────────────────────────
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

    if args.by_zip:
        plot_by_zip(runs_data, args.out, selected_zips=args.zips)
    else:
        plot_timeseries(runs_data, args.out)


if __name__ == "__main__":
    main()
