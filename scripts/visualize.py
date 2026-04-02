"""
scripts/visualize.py
====================
Plot predicted vs actual insurance premiums for one or more finished runs.

Usage:
    conda activate qlstm

    # all benchmark runs
    python scripts/visualize.py --run "lightning_logs/*/version_0"

    # specific models
    python scripts/visualize.py \\
        --run lightning_logs/LSTM_*/version_0 \\
        --run lightning_logs/cQLSTM_*/version_0

    # save to file instead of displaying
    python scripts/visualize.py --run "lightning_logs/*/version_0" --out results.png

Outputs:
    • Scatter: predicted vs actual  (one series per model)
    • Bar chart: Test R², RMSE, MAE side-by-side
"""

import argparse
import glob
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
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

# ────────────────────────────── dataset helpers ───────────────────────────────

class WildfireDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def rebuild_test(cfg: dict):
    d = cfg["data"]
    df = pd.read_csv(ROOT / d["path"]).sort_values(["ZIP", "Year"])
    years = sorted(df["Year"].unique())
    in_years = years[:d["time_steps"]]
    tgt_year = years[-1]

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
    y = scaler_y.fit_transform(y)

    split = d["split"]
    n_tr = int(N * split[0]); n_va = int(N * split[1])
    test_idx = range(n_tr + n_va, N)
    loader = DataLoader(
        Subset(WildfireDataset(X, y), test_idx),
        batch_size=256, shuffle=False, num_workers=2,
    )
    return loader, scaler_y


def load_run(run_dir: Path):
    cfg_path = run_dir / "run_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"run_config.yaml missing in {run_dir}")
    cfg = yaml.safe_load(cfg_path.read_text())

    ckpts = sorted(run_dir.glob("best-*.ckpt")) or sorted(run_dir.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint in {run_dir}")

    m = cfg["model"]
    name = m["name"]
    if name == "LSTM":
        from models.LSTM import LSTM
        model = LSTM(cfg["data"]["input_size"], m["hidden_size"])
    elif name == "cLSTM":
        from models.cLSTM import cLSTM
        model = cLSTM(cfg["data"]["input_size"], m["hidden_size"], m["decay_rate"])
    elif name == "QLSTM":
        from models.QLSTM import QLSTM
        model = QLSTM(cfg["data"]["input_size"], m["hidden_size"], m["n_qubits"], m["n_qlayers"])
    elif name == "cQLSTM":
        from models.cQLSTM import cQLSTM
        model = cQLSTM(cfg["data"]["input_size"], m["hidden_size"],
                       m["decay_rate"], m["n_qubits"], m["n_qlayers"], m["n_esteps"])
    elif name == "SSM":
        from models.SSM import SSM
        model = SSM(cfg["data"]["input_size"], m["hidden_size"], m["g_min"], m["g_max"])
    elif name == "QSSM":
        from models.QSSM import QuantumSSM
        model = QuantumSSM(cfg["data"]["input_size"], m["hidden_size"],
                           m["n_qubits"], m["n_qlayers"], m["n_esteps"], m["g_min"], m["g_max"])
    else:
        raise ValueError(f"Unknown model: {name}")

    t = cfg["training"]
    opt = optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    lit = LitModel(model=model, output_size=cfg["data"]["n_future"],
                   criterion=nn.SmoothL1Loss(), optimizer=[opt], scheduler=None)
    lit.load(str(ckpts[-1]), verbose=False)
    lit.eval()
    return lit, cfg


def predict(lit, loader, scaler_y):
    preds, trues = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.append(lit(X).numpy())
            trues.append(y.numpy())
    pred = scaler_y.inverse_transform(np.vstack(preds))
    true = scaler_y.inverse_transform(np.vstack(trues))
    return pred.ravel(), true.ravel()


# ────────────────────────────── plotting ──────────────────────────────────────

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


def plot_scatter(ax, true, pred, label, color, r2):
    ax.scatter(true / 1e6, pred / 1e6, alpha=0.35, s=18, color=color,
               label=f"{label}  (R²={r2:.3f})")


def plot_all(runs_data: list, out_path):
    """
    runs_data: list of (model_name, true, pred, metrics_dict)
    """
    n_models = len(runs_data)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Wildfire Insurance Premium — Test Set Comparison", fontsize=14, fontweight="bold")

    ax_scatter, ax_rmse, ax_r2 = axes

    # ── scatter: predicted vs actual ─────────────────────────────────────
    all_true = runs_data[0][1]
    lo = all_true.min() / 1e6; hi = all_true.max() / 1e6
    ax_scatter.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="Perfect")

    for i, (name, true, pred, m) in enumerate(runs_data):
        plot_scatter(ax_scatter, true, pred, name, COLORS[i % len(COLORS)], m["r2"])

    ax_scatter.set_xlabel("Actual Premium ($M)", fontsize=11)
    ax_scatter.set_ylabel("Predicted Premium ($M)", fontsize=11)
    ax_scatter.set_title("Predicted vs Actual", fontsize=12)
    ax_scatter.legend(fontsize=9, loc="upper left")

    # ── bar: RMSE ─────────────────────────────────────────────────────────
    names = [r[0] for r in runs_data]
    rmses = [r[3]["rmse"] / 1e6 for r in runs_data]
    bars  = ax_rmse.bar(names, rmses, color=COLORS[:n_models], edgecolor="white", width=0.5)
    ax_rmse.bar_label(bars, fmt="$%.2fM", padding=3, fontsize=9)
    ax_rmse.set_ylabel("RMSE ($M)", fontsize=11)
    ax_rmse.set_title("Test RMSE  (lower is better)", fontsize=12)
    ax_rmse.tick_params(axis="x", rotation=20)

    # ── bar: R² ───────────────────────────────────────────────────────────
    r2s  = [r[3]["r2"] for r in runs_data]
    bars = ax_r2.bar(names, r2s, color=COLORS[:n_models], edgecolor="white", width=0.5)
    ax_r2.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax_r2.set_ylim(0, 1)
    ax_r2.set_ylabel("R²", fontsize=11)
    ax_r2.set_title("Test R²  (higher is better)", fontsize=12)
    ax_r2.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"Saved  →  {out_path}")
    else:
        plt.show()

    plt.close(fig)


# ────────────────────────────── main ─────────────────────────────────────────

def resolve_dirs(raw: list[str]) -> list[Path]:
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
    parser = argparse.ArgumentParser(description="Visualize wildfire run predictions.")
    parser.add_argument("--run", action="append", required=True, metavar="DIR",
                        help="Run version dir (repeat or use glob). "
                             "E.g. --run 'lightning_logs/*/version_0'")
    parser.add_argument("--out", default=None, metavar="FILE",
                        help="Save plot to file (PNG/PDF). Default: display interactively.")
    args = parser.parse_args()

    run_dirs = resolve_dirs(args.run)
    if not run_dirs:
        print("No valid run directories found."); return

    runs_data = []
    seen_cfgs = {}

    for run_dir in run_dirs:
        print(f"Loading  {run_dir} …")
        try:
            lit, cfg = load_run(run_dir)
        except FileNotFoundError as e:
            print(f"  [skip] {e}"); continue

        cfg_key = id(cfg)
        if cfg_key not in seen_cfgs:
            loader, scaler_y = rebuild_test(cfg)
            seen_cfgs[cfg_key] = (loader, scaler_y)
        else:
            loader, scaler_y = seen_cfgs[cfg_key]

        # rebuild fresh for each run so scaler is independent per cfg
        loader, scaler_y = rebuild_test(cfg)
        pred, true = predict(lit, loader, scaler_y)

        mask = true > 10_000
        metrics = {
            "r2":    r2_score(true, pred),
            "rmse":  float(np.sqrt(mean_squared_error(true, pred))),
            "mae":   float(mean_absolute_error(true, pred)),
            "mdape": float(np.median(np.abs((true[mask] - pred[mask]) / true[mask])) * 100),
        }

        name = cfg["model"]["name"]
        print(f"  {name:<8}  R²={metrics['r2']:.4f}  RMSE=${metrics['rmse']:,.0f}  "
              f"MAE=${metrics['mae']:,.0f}  MdAPE={metrics['mdape']:.1f}%")

        runs_data.append((name, true, pred, metrics))

    if not runs_data:
        print("Nothing to plot."); return

    plot_all(runs_data, args.out)


if __name__ == "__main__":
    main()
