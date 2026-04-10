#!/usr/bin/env python3
"""
Create paper-ready predicted-vs-actual regression plots for Task 1A and Task 2.

Produces two separate figures (one per task), each as a standalone PDF + PNG.

The script accepts either:
  1. direct prediction CSVs with columns: ZIP, target_year, actual, prediction
  2. results CSVs produced by scripts/task1a.py, from which it can resolve the
     best prediction file automatically.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_TASK1A_RESULTS = ROOT / "output" / "task1a_dq_nqb4_nl1" / "results.csv"
DEFAULT_TASK2_RESULTS = ROOT / "output" / "task2_tslib_benchmarks" / "results.csv"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "figures"


@dataclass
class PlotSpec:
    label: str
    dataframe: pd.DataFrame
    title: str
    subtitle: str
    model_name: str
    dataset_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paper-ready predicted-vs-actual plots for Task 1A and Task 2."
    )
    parser.add_argument("--task1a-prediction", default=None, help="Direct Task 1A prediction CSV.")
    parser.add_argument("--task2-prediction", default=None, help="Direct Task 2 prediction CSV.")
    parser.add_argument(
        "--task1a-results",
        default=str(DEFAULT_TASK1A_RESULTS),
        help="Task 1A results.csv used to auto-resolve the prediction CSV.",
    )
    parser.add_argument(
        "--task2-results",
        default=str(DEFAULT_TASK2_RESULTS),
        help="Task 2 results.csv used to auto-resolve the prediction CSV.",
    )
    parser.add_argument("--task1a-model", default=None, help="Optional Task 1A model filter.")
    parser.add_argument("--task2-model", default=None, help="Optional Task 2 model filter.")
    parser.add_argument("--task1a-dataset", default=None, help="Optional Task 1A dataset filter.")
    parser.add_argument("--task2-dataset", default=None, help="Optional Task 2 dataset filter.")
    parser.add_argument(
        "--selection-metric",
        choices=["r2", "rmse", "mse", "mae"],
        default="r2",
        help="Metric used to select the best row from results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for figures.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG output DPI.")
    parser.add_argument("--bins", type=int, default=30, help="Hexbin grid size.")
    return parser.parse_args()


def load_prediction_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"actual", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def better_metric(metric: str, lhs: float, rhs: float) -> bool:
    if metric == "r2":
        return lhs > rhs
    return lhs < rhs


def select_prediction_from_results(
    results_path: Path,
    metric: str,
    model_filter: str | None,
    dataset_filter: str | None,
) -> tuple[Path, pd.Series]:
    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_path}")

    df = pd.read_csv(results_path)
    df = df[df["status"] == "ok"].copy()
    if model_filter:
        df = df[df["model"] == model_filter]
    if dataset_filter:
        df = df[df["dataset"] == dataset_filter]
    if df.empty:
        raise ValueError(
            f"No matching successful rows in {results_path} for model={model_filter!r}, dataset={dataset_filter!r}"
        )

    ascending = metric != "r2"
    row = df.sort_values(metric, ascending=ascending).iloc[0]
    prediction_path = resolve_prediction_path(results_path, row)
    return prediction_path, row


def resolve_prediction_path(results_path: Path, row: pd.Series) -> Path:
    raw_path = Path(str(row["prediction_path"]))
    if raw_path.exists():
        return raw_path

    basename = raw_path.name
    direct_candidates = [
        results_path.parent / "predictions" / basename,
        results_path.parent / row.get("run_tag", "") / "predictions" / basename,
        results_path.parent / "runs" / str(row.get("run_tag", "")) / "predictions" / basename,
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    glob_candidates = sorted(ROOT.glob(f"output/**/predictions/{basename}"))
    if len(glob_candidates) == 1:
        return glob_candidates[0]
    if len(glob_candidates) > 1:
        run_tag = str(row.get("run_tag", ""))
        for candidate in glob_candidates:
            if candidate.parent.parent.name == run_tag:
                return candidate
        return glob_candidates[0]

    raise FileNotFoundError(
        f"Prediction CSV referenced by results row does not exist and could not be recovered: {raw_path}"
    )


def resolve_spec(
    label: str,
    prediction_arg: str | None,
    results_arg: str,
    metric: str,
    model_filter: str | None,
    dataset_filter: str | None,
) -> PlotSpec:
    if prediction_arg:
        prediction_path = Path(prediction_arg).expanduser().resolve()
        df = load_prediction_csv(prediction_path)
        return PlotSpec(
            label=label,
            dataframe=df,
            title=label,
            subtitle=prediction_path.stem,
            model_name=prediction_path.stem,
            dataset_name="",
        )

    prediction_path, row = select_prediction_from_results(
        results_path=Path(results_arg).expanduser().resolve(),
        metric=metric,
        model_filter=model_filter,
        dataset_filter=dataset_filter,
    )
    df = load_prediction_csv(prediction_path)
    model_name = str(row["model"])
    dataset_name = str(row["dataset"])
    subtitle = f"{model_name}  |  {dataset_name} dataset"

    # Add quantum info if available
    nq = row.get("n_qubits", "")
    nl = row.get("n_qlayers", "")
    if pd.notna(nq) and str(nq).strip():
        reup = row.get("data_reupload", "")
        reup_str = ", re-upload" if str(reup) == "True" else ""
        subtitle += f"  |  {int(float(nq))}q, {int(float(nl))}L{reup_str}"

    return PlotSpec(
        label=label,
        dataframe=df,
        title=label,
        subtitle=subtitle,
        model_name=model_name,
        dataset_name=dataset_name,
    )


def compute_metrics(df: pd.DataFrame) -> dict:
    actual = df["actual"].to_numpy()
    pred = df["prediction"].to_numpy()
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual, pred = actual[mask], pred[mask]
    return {
        "n": len(actual),
        "mae": mean_absolute_error(actual, pred),
        "mse": mean_squared_error(actual, pred),
        "rmse": mean_squared_error(actual, pred) ** 0.5,
        "r2": r2_score(actual, pred),
    }


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

TASK1A_PALETTE = {
    "cmap": "YlGnBu",
    "line_color": "#b5342e",       # regression line
    "diag_color": "#2c2c2c",       # y=x diagonal
    "box_face": "#f0f7fb",         # metrics box fill
    "box_edge": "#7fafc9",         # metrics box border
    "accent": "#1b6d91",           # title accent
    "scatter_edge": "#3a7ca5",
}

TASK2_PALETTE = {
    "cmap": "YlOrRd",
    "line_color": "#1a5276",
    "diag_color": "#2c2c2c",
    "box_face": "#fdf5ee",
    "box_edge": "#d4915e",
    "accent": "#b35c1e",
    "scatter_edge": "#c4692a",
}


def draw_figure(spec: PlotSpec, bins: int, palette: dict, dpi: int) -> plt.Figure:
    """Draw a single standalone predicted-vs-actual figure."""
    df = spec.dataframe
    actual = df["actual"].to_numpy()
    pred = df["prediction"].to_numpy()

    mask = np.isfinite(actual) & np.isfinite(pred)
    actual, pred = actual[mask], pred[mask]
    if len(actual) == 0:
        raise ValueError(f"{spec.label}: no finite actual/prediction pairs to plot")

    metrics = compute_metrics(df)

    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    # --- Axis range ---
    lower = float(min(actual.min(), pred.min()))
    upper = float(max(actual.max(), pred.max()))
    span = upper - lower if upper > lower else 1.0
    margin = 0.06 * span
    lower -= margin
    upper += margin

    # --- Hexbin density plot ---
    hb = ax.hexbin(
        actual,
        pred,
        gridsize=bins,
        cmap=palette["cmap"],
        mincnt=1,
        linewidths=0.2,
        edgecolors="white",
        alpha=0.92,
    )

    # --- Perfect prediction diagonal (y = x) ---
    ax.plot(
        [lower, upper],
        [lower, upper],
        color=palette["diag_color"],
        linestyle="--",
        linewidth=1.3,
        alpha=0.7,
        label="Perfect prediction",
        zorder=3,
    )

    # --- Linear regression fit ---
    coeffs = np.polyfit(actual, pred, deg=1)
    xs = np.linspace(lower, upper, 300)
    fit_label = f"Fit: y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}"
    ax.plot(
        xs,
        coeffs[0] * xs + coeffs[1],
        color=palette["line_color"],
        linewidth=2.2,
        alpha=0.9,
        label=fit_label,
        zorder=4,
    )

    # --- Axis formatting ---
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Actual", fontsize=12, fontweight="medium", labelpad=8)
    ax.set_ylabel("Predicted", fontsize=12, fontweight="medium", labelpad=8)

    # --- Title and subtitle ---
    ax.set_title(
        spec.title,
        fontsize=15,
        fontweight="bold",
        color=palette["accent"],
        pad=22,
    )
    # ax.text(
    #     0.5,
    #     1.015,
    #     spec.subtitle,
    #     transform=ax.transAxes,
    #     ha="center",
    #     va="bottom",
    #     fontsize=9.5,
    #     color="#666666",
    #     style="italic",
    # )

    # --- Metrics annotation box ---
    metrics_lines = [
        f"$n$ = {metrics['n']:,}",
        f"MAE = {metrics['mae']:.4f}",
        f"RMSE = {metrics['rmse']:.4f}",
        f"$R^2$ = {metrics['r2']:.4f}",
    ]
    metrics_text = "\n".join(metrics_lines)
    ax.text(
        0.04,
        0.96,
        metrics_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor=palette["box_face"],
            edgecolor=palette["box_edge"],
            linewidth=1.2,
            alpha=0.95,
        ),
        zorder=5,
    )

    # --- Grid ---
    ax.grid(True, color="#d4d4d4", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # --- Tick formatting ---
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(axis="both", labelsize=10, direction="out", length=4)

    # --- Spine styling ---
    for spine in ax.spines.values():
        spine.set_color("#aaaaaa")
        spine.set_linewidth(0.8)

    # --- Colorbar ---
    cbar = fig.colorbar(hb, ax=ax, pad=0.025, fraction=0.046, shrink=0.88)
    cbar.ax.set_ylabel("ZIP codes per bin", fontsize=10, labelpad=8)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor("#aaaaaa")

    # --- Legend ---
    legend = ax.legend(
        loc="lower right",
        fontsize=9,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        edgecolor="#cccccc",
    )
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#aaaaaa",
            "axes.linewidth": 0.8,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.size": 10.5,
            "font.family": "sans-serif",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, name: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    print(f"  [png] {png_path}")
    print(f"  [pdf] {pdf_path}")


def main() -> None:
    args = parse_args()
    configure_style()
    output_dir = Path(args.output_dir).expanduser().resolve()

    # --- Task 1A ---
    print("Task 1A: Wildfire Risk Forecasting")
    try:
        task1a_spec = resolve_spec(
            label="Task 1A: Wildfire Risk Score",
            prediction_arg=args.task1a_prediction,
            results_arg=args.task1a_results,
            metric=args.selection_metric,
            model_filter=args.task1a_model,
            dataset_filter=args.task1a_dataset,
        )
        m = compute_metrics(task1a_spec.dataframe)
        print(f"  Model:   {task1a_spec.model_name}")
        print(f"  Dataset: {task1a_spec.dataset_name}")
        print(f"  Samples: {m['n']:,}")
        print(f"  MAE:     {m['mae']:.4f}")
        print(f"  RMSE:    {m['rmse']:.4f}")
        print(f"  R2:      {m['r2']:.4f}")

        fig1 = draw_figure(task1a_spec, bins=args.bins, palette=TASK1A_PALETTE, dpi=args.dpi)
        save_figure(fig1, output_dir, "predicted_vs_actual_task1a", args.dpi)
        plt.close(fig1)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  [skip] {exc}")

    print()

    # --- Task 2 ---
    print("Task 2: Insurance Premium Forecasting")
    try:
        task2_spec = resolve_spec(
            label="Task 2: Insurance Earned Premium",
            prediction_arg=args.task2_prediction,
            results_arg=args.task2_results,
            metric=args.selection_metric,
            model_filter=args.task2_model,
            dataset_filter=args.task2_dataset,
        )
        m = compute_metrics(task2_spec.dataframe)
        print(f"  Model:   {task2_spec.model_name}")
        print(f"  Dataset: {task2_spec.dataset_name}")
        print(f"  Samples: {m['n']:,}")
        print(f"  MAE:     {m['mae']:.4f}")
        print(f"  RMSE:    {m['rmse']:.4f}")
        print(f"  R2:      {m['r2']:.4f}")

        fig2 = draw_figure(task2_spec, bins=args.bins, palette=TASK2_PALETTE, dpi=args.dpi)
        save_figure(fig2, output_dir, "predicted_vs_actual_task2", args.dpi)
        plt.close(fig2)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  [skip] {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
