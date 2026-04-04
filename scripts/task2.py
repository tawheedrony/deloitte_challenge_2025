"""
scripts/task2.py  —  Task 2: Insurance Premium Prediction
==========================================================
Trains and evaluates classical time series / ML models for premium prediction:
  - ARIMA       (per-ZIP univariate, statsmodels)
  - Prophet     (per-ZIP univariate, prophet)
  - XGBoost     (tabular, xgboost)
  - LightGBM    (tabular, lightgbm)
  - RandomForest (tabular, sklearn)
  - GradientBoosting (tabular, sklearn)

Also optionally runs the existing neural network models (LSTM, cQLSTM, etc.)
through train.py for direct comparison.

Usage:
    python scripts/task2.py                          # all classical ML models
    python scripts/task2.py --models xgboost lgbm    # specific models
    python scripts/task2.py --include-nn              # also train NN models
    python scripts/task2.py --dataset engineered      # use engineered features

Output:
    output/task2/
    ├── predictions/           per-model predicted premiums (CSV)
    ├── task2_results.csv      comparison table
    └── task2_summary.txt      full report
"""

import argparse
import sys
import time
import warnings
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ─────────────────────────── data loading ────────────────────────────────

FEATURES_BASE = [
    "avg_fire_risk_score", "earned_exposure", "cat_fire_losses",
    "noncat_fire_losses", "total_population", "median_income",
    "premium_rolling_mean", "year_sin", "year_cos",
]

FEATURES_ENGINEERED = [
    "loss_ratio", "claim_frequency", "cat_share", "log_earned_premium",
    "avg_fire_risk_score", "high_risk_pct", "risk_concentration",
    "drought_proxy", "temp_range", "heat_index", "log_prcp",
    "log_gis_acres", "fire_occurred",
    "poverty_rate", "edu_rate", "log_median_income", "vacancy_rate",
    "housing_cost_burden", "vulnerability_index", "year_sin",
]

TARGET = "earned_premium"


def load_data(dataset_type="preprocessed"):
    if dataset_type == "engineered":
        path = ROOT / "data" / "wildfire_engineered.csv"
        features = FEATURES_ENGINEERED
    else:
        path = ROOT / "data" / "wildfire_preprocessed.csv"
        features = FEATURES_BASE

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run:  python scripts/preprocess.py"
            + (" --engineered" if dataset_type == "engineered" else ""))

    df = pd.read_csv(path).sort_values(["ZIP", "Year"])
    return df, features


def build_tabular_dataset(df, features, time_steps=3):
    """
    Flatten multi-year sequences into a single feature vector per ZIP.
    Input: 3 years of features → flat vector.
    Target: earned_premium in the final year.
    """
    years = sorted(df["Year"].unique())
    in_years = years[:time_steps]
    tgt_year = years[-1]

    X_list, y_list, zip_list = [], [], []
    for zip_code, z_df in df.groupby("ZIP"):
        z_df = z_df.set_index("Year")
        if not all(yr in z_df.index for yr in in_years + [tgt_year]):
            continue
        # flatten: [year1_feat1, year1_feat2, ..., year3_featN]
        x_flat = z_df.loc[in_years, features].values.ravel()
        yv = z_df.loc[tgt_year, TARGET]
        if np.isnan(x_flat).any() or np.isnan(yv):
            continue
        X_list.append(x_flat)
        y_list.append(yv)
        zip_list.append(zip_code)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # train/val/test split (70/15/15)
    n = len(X)
    n_tr = int(n * 0.70)
    n_va = int(n * 0.15)

    X_train, y_train = X[:n_tr], y[:n_tr]
    X_val, y_val = X[n_tr:n_tr+n_va], y[n_tr:n_tr+n_va]
    X_test, y_test = X[n_tr+n_va:], y[n_tr+n_va:]

    # feature names for interpretability
    feat_names = []
    for yr in in_years:
        for f in features:
            feat_names.append(f"{f}_yr{yr}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "zip_test": zip_list[n_tr+n_va:],
        "feat_names": feat_names,
        "n_total": n,
    }


def build_timeseries_data(df, time_steps=3):
    """
    Build per-ZIP premium time series for ARIMA/Prophet.
    Returns list of (zip_code, premium_series, target_value).
    """
    years = sorted(df["Year"].unique())
    in_years = years[:time_steps]
    tgt_year = years[-1]

    series_list = []
    for zip_code, z_df in df.groupby("ZIP"):
        z_df = z_df.set_index("Year").sort_index()
        if not all(yr in z_df.index for yr in in_years + [tgt_year]):
            continue
        premiums = z_df.loc[in_years, TARGET].values
        target = z_df.loc[tgt_year, TARGET]
        if np.isnan(premiums).any() or np.isnan(target):
            continue
        series_list.append((zip_code, premiums, target))

    return series_list, in_years, tgt_year


# ─────────────────────────── model runners ───────────────────────────────

def run_xgboost(data):
    """XGBoost gradient boosted trees."""
    try:
        import xgboost as xgb
    except ImportError:
        print("  [skip] xgboost not installed. pip install xgboost")
        return None

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        early_stopping_rounds=20, random_state=42,
    )
    t0 = time.time()
    model.fit(data["X_train"], data["y_train"],
              eval_set=[(data["X_val"], data["y_val"])], verbose=False)
    train_time = time.time() - t0

    pred = model.predict(data["X_test"])
    return pred, train_time, model


def run_lightgbm(data):
    """LightGBM gradient boosted trees."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("  [skip] lightgbm not installed. pip install lightgbm")
        return None

    model = lgb.LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1,
    )
    t0 = time.time()
    model.fit(data["X_train"], data["y_train"],
              eval_set=[(data["X_val"], data["y_val"])],
              callbacks=[lgb.early_stopping(20, verbose=False)])
    train_time = time.time() - t0

    pred = model.predict(data["X_test"])
    return pred, train_time, model


def run_random_forest(data):
    """Random Forest regressor."""
    model = RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=5,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    t0 = time.time()
    model.fit(data["X_train"], data["y_train"])
    train_time = time.time() - t0

    pred = model.predict(data["X_test"])
    return pred, train_time, model


def run_gradient_boosting(data):
    """Sklearn GradientBoosting regressor."""
    model = GradientBoostingRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42,
    )
    t0 = time.time()
    model.fit(data["X_train"], data["y_train"])
    train_time = time.time() - t0

    pred = model.predict(data["X_test"])
    return pred, train_time, model


def run_arima(series_data, split_ratio=0.85):
    """
    Per-ZIP ARIMA(1,1,0) on premium series.
    With only 3 historical points, ARIMA is severely limited.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        print("  [skip] statsmodels not installed. pip install statsmodels")
        return None

    preds, trues, zips = [], [], []
    n_total = len(series_data)
    n_test_start = int(n_total * split_ratio)

    t0 = time.time()
    for i, (zip_code, premiums, target) in enumerate(series_data):
        if i < n_test_start:
            continue
        try:
            model = ARIMA(premiums, order=(1, 1, 0))
            fitted = model.fit()
            forecast = fitted.forecast(steps=1)[0]
            preds.append(forecast)
            trues.append(target)
            zips.append(zip_code)
        except Exception:
            # ARIMA can fail on very short or constant series
            preds.append(premiums[-1])  # naive fallback
            trues.append(target)
            zips.append(zip_code)

    train_time = time.time() - t0
    return np.array(preds), np.array(trues), train_time, zips


def run_prophet(series_data, in_years, tgt_year, split_ratio=0.85):
    """
    Per-ZIP Prophet on premium series.
    With only 3 historical points, Prophet is at its minimum viable input.
    """
    try:
        from prophet import Prophet
    except ImportError:
        print("  [skip] prophet not installed. pip install prophet")
        return None

    import logging
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    preds, trues, zips = [], [], []
    n_total = len(series_data)
    n_test_start = int(n_total * split_ratio)

    t0 = time.time()
    for i, (zip_code, premiums, target) in enumerate(series_data):
        if i < n_test_start:
            continue
        try:
            df_prophet = pd.DataFrame({
                "ds": pd.to_datetime([f"{yr}-01-01" for yr in in_years]),
                "y": premiums,
            })
            m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                        daily_seasonality=False)
            m.fit(df_prophet)
            future = pd.DataFrame({"ds": [pd.Timestamp(f"{tgt_year}-01-01")]})
            forecast = m.predict(future)["yhat"].values[0]
            preds.append(forecast)
            trues.append(target)
            zips.append(zip_code)
        except Exception:
            preds.append(premiums[-1])
            trues.append(target)
            zips.append(zip_code)

    train_time = time.time() - t0
    return np.array(preds), np.array(trues), train_time, zips


# ─────────────────────────── metrics ─────────────────────────────────────

def compute_metrics(y_true, y_pred, model_name):
    mask = y_true > 10_000  # avoid divide-by-zero for tiny premiums
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mdape = float(np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")

    return {
        "model": model_name,
        "R²": round(r2, 4),
        "RMSE ($)": round(rmse),
        "MAE ($)": round(mae),
        "MdAPE (%)": round(mdape, 1),
    }


# ─────────────────────────── main ────────────────────────────────────────

MODEL_RUNNERS = {
    "xgboost": ("XGBoost", run_xgboost),
    "lgbm": ("LightGBM", run_lightgbm),
    "rf": ("RandomForest", run_random_forest),
    "gbr": ("GradientBoosting", run_gradient_boosting),
}

TS_RUNNERS = {
    "arima": "ARIMA",
    "prophet": "Prophet",
}


def main():
    parser = argparse.ArgumentParser(
        description="Task 2: Train classical ML models for premium prediction.")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Models to run. Tabular: {list(MODEL_RUNNERS.keys())}. "
                             f"Time series: {list(TS_RUNNERS.keys())}. Default: all.")
    parser.add_argument("--dataset", default="preprocessed",
                        choices=["preprocessed", "engineered"],
                        help="Dataset to use (default: preprocessed)")
    parser.add_argument("--include-nn", action="store_true",
                        help="Also train NN models (LSTM, cQLSTM) via train.py")
    args = parser.parse_args()

    out_dir = ROOT / "output" / "task2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)

    selected = args.models or list(MODEL_RUNNERS.keys()) + list(TS_RUNNERS.keys())

    print("=" * 60)
    print("  TASK 2 — Insurance Premium Prediction")
    print(f"  Dataset: {args.dataset}")
    print(f"  Models: {', '.join(selected)}")
    print("=" * 60)

    # ── load data ────────────────────────────────────────────────────
    df, features = load_data(args.dataset)
    print(f"\n  Loaded {len(df):,} rows, {df['ZIP'].nunique()} ZIPs, "
          f"years {sorted(df['Year'].unique())}")

    all_results = []

    # ── tabular models ───────────────────────────────────────────────
    tabular_models = [m for m in selected if m in MODEL_RUNNERS]
    if tabular_models:
        print(f"\n  Building tabular dataset ({len(features)} features x 3 years = "
              f"{len(features)*3} flat features)...")
        data = build_tabular_dataset(df, features)
        print(f"  Train: {len(data['X_train']):,}  Val: {len(data['X_val']):,}  "
              f"Test: {len(data['X_test']):,}")

        for key in tabular_models:
            name, runner = MODEL_RUNNERS[key]
            print(f"\n{'─'*50}")
            print(f"  Training: {name}")
            print(f"{'─'*50}")

            result = runner(data)
            if result is None:
                continue
            pred, train_time, model = result

            metrics = compute_metrics(data["y_test"], pred, name)
            metrics["train_time_s"] = round(train_time, 1)
            all_results.append(metrics)

            # save predictions
            pred_df = pd.DataFrame({
                "predicted": pred,
                "actual": data["y_test"],
            })
            if "zip_test" in data and len(data["zip_test"]) == len(pred_df):
                pred_df.insert(0, "ZIP", data["zip_test"])
            pred_df.to_csv(out_dir / "predictions" / f"{name}.csv", index=False)

            print(f"  R²={metrics['R²']:.4f}  RMSE=${metrics['RMSE ($)']:,}  "
                  f"MAE=${metrics['MAE ($)']:,}  Time={train_time:.1f}s")

            # feature importance (top 10)
            if hasattr(model, "feature_importances_"):
                imp = pd.Series(model.feature_importances_, index=data["feat_names"])
                top10 = imp.nlargest(10)
                print(f"  Top features: {', '.join(top10.index[:5])}")

    # ── time series models ───────────────────────────────────────────
    ts_models = [m for m in selected if m in TS_RUNNERS]
    if ts_models:
        print(f"\n  Building per-ZIP time series...")
        series_data, in_years, tgt_year = build_timeseries_data(df)
        print(f"  {len(series_data):,} ZIP series ({len(in_years)} points each)")

        if "arima" in ts_models:
            print(f"\n{'─'*50}")
            print(f"  Training: ARIMA (per-ZIP)")
            print(f"{'─'*50}")

            result = run_arima(series_data)
            if result is not None:
                pred, true, train_time, zips = result
                metrics = compute_metrics(true, pred, "ARIMA")
                metrics["train_time_s"] = round(train_time, 1)
                all_results.append(metrics)

                arima_df = pd.DataFrame({"predicted": pred, "actual": true})
                if zips and len(zips) == len(arima_df):
                    arima_df.insert(0, "ZIP", zips)
                arima_df.to_csv(out_dir / "predictions" / "ARIMA.csv", index=False)
                print(f"  R²={metrics['R²']:.4f}  RMSE=${metrics['RMSE ($)']:,}  "
                      f"Time={train_time:.1f}s")

        if "prophet" in ts_models:
            print(f"\n{'─'*50}")
            print(f"  Training: Prophet (per-ZIP)")
            print(f"{'─'*50}")

            result = run_prophet(series_data, in_years, tgt_year)
            if result is not None:
                pred, true, train_time, zips = result
                metrics = compute_metrics(true, pred, "Prophet")
                metrics["train_time_s"] = round(train_time, 1)
                all_results.append(metrics)

                prophet_df = pd.DataFrame({"predicted": pred, "actual": true})
                if zips and len(zips) == len(prophet_df):
                    prophet_df.insert(0, "ZIP", zips)
                prophet_df.to_csv(out_dir / "predictions" / "Prophet.csv", index=False)
                print(f"  R²={metrics['R²']:.4f}  RMSE=${metrics['RMSE ($)']:,}  "
                      f"Time={train_time:.1f}s")

    # ── NN models (optional) ─────────────────────────────────────────
    if args.include_nn:
        print(f"\n  Training NN models via train.py...")
        import subprocess
        for model_name in ["LSTM", "cQLSTM"]:
            cfg_file = f"configs/model/{model_name}.yaml"
            print(f"\n{'─'*50}")
            print(f"  Training: {model_name} (neural network)")
            print(f"{'─'*50}")
            cmd = [sys.executable, "scripts/train.py", "--config", cfg_file]
            subprocess.run(cmd, cwd=str(ROOT))

    # ── comparison table ─────────────────────────────────────────────
    if all_results:
        print(f"\n{'='*70}")
        print(f"  TASK 2 RESULTS  (ranked by R²)")
        print(f"{'='*70}")

        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values("R²", ascending=False)
        print(results_df.to_string(index=False))

        # save
        csv_path = out_dir / "task2_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n  Results -> {csv_path}")

        # summary
        buf = StringIO()
        buf.write("=" * 70 + "\n")
        buf.write("  TASK 2 — INSURANCE PREMIUM PREDICTION SUMMARY\n")
        buf.write("  2026 Quantum Sustainability Challenge\n")
        buf.write("=" * 70 + "\n\n")
        buf.write(f"  Target: {TARGET}\n")
        buf.write(f"  Dataset: {args.dataset}\n\n")
        buf.write("  RESULTS (ranked by R²)\n")
        buf.write("-" * 70 + "\n")
        buf.write(results_df.to_string(index=False))
        buf.write("\n\n")
        buf.write("  Note: ARIMA and Prophet operate on 3-point per-ZIP series,\n")
        buf.write("  which is the minimum viable input. Tabular models (XGBoost,\n")
        buf.write("  LightGBM, RF) use the full feature set flattened across years.\n")

        summary_path = out_dir / "task2_summary.txt"
        summary_path.write_text(buf.getvalue())
        print(f"  Summary -> {summary_path}")
        print(f"  Predictions in: {out_dir / 'predictions'}")

    print(f"\n{'='*60}")
    print(f"  Task 2 complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
