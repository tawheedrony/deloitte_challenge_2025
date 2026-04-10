"""
Preprocess the raw wildfire/insurance join into the retained task-specific datasets.

Outputs:
  Task 1A -> data/preprocessed/task1a/
  Task 2  -> data/preprocessed/task2/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "abfa2rbci2UF6CTj_cal_insurance_fire_census_weather.csv"
TASK1A_DIR = DATA_DIR / "preprocessed" / "task1a"
TASK2_DIR = DATA_DIR / "preprocessed" / "task2"

INSURANCE_SUM_COLS = [
    "Earned Premium",
    "Earned Exposure",
    "CAT Cov A Fire -  Incurred Losses",
    "Non-CAT Cov A Fire -  Incurred Losses",
    "CAT Cov A Smoke -  Incurred Losses",
    "Non-CAT Cov A Smoke -  Incurred Losses",
    "CAT Cov C Fire -  Incurred Losses",
    "Non-CAT Cov C Fire -  Incurred Losses",
    "CAT Cov C Smoke -  Incurred Losses",
    "Non-CAT Cov C Smoke -  Incurred Losses",
    "CAT Cov A Fire -  Number of Claims",
    "Non-CAT Cov A Fire -  Number of Claims",
    "CAT Cov C Fire -  Number of Claims",
    "Non-CAT Cov C Fire -  Number of Claims",
    "Number of High Fire Risk Exposure",
    "Number of Low Fire Risk Exposure",
    "Number of Moderate Fire Risk Exposure",
    "Number of Negligible Fire Risk Exposure",
    "Number of Very High Fire Risk Exposure",
]
INSURANCE_MEAN_COLS = ["Avg Fire Risk Score", "Avg PPC"]
WEATHER_MEAN_COLS = ["avg_tmax_c", "avg_tmin_c", "tot_prcp_mm"]
FIRE_AGG_COLS = ["GIS_ACRES"]
CENSUS_FIRST_COLS = [
    "total_population",
    "median_income",
    "total_housing_units",
    "housing_vacancy_number",
    "educational_attainment_bachelor_or_higher",
    "poverty_status",
    "owner_occupied_housing_units",
    "renter_occupied_housing_units",
    "housing_occupancy_number",
    "housing_value",
    "median_monthly_housing_costs",
]

RENAME = {
    "Earned Premium": "earned_premium",
    "Earned Exposure": "earned_exposure",
    "CAT Cov A Fire -  Incurred Losses": "cat_cov_a_fire",
    "Non-CAT Cov A Fire -  Incurred Losses": "noncat_cov_a_fire",
    "CAT Cov A Smoke -  Incurred Losses": "cat_cov_a_smoke",
    "Non-CAT Cov A Smoke -  Incurred Losses": "noncat_cov_a_smoke",
    "CAT Cov C Fire -  Incurred Losses": "cat_cov_c_fire",
    "Non-CAT Cov C Fire -  Incurred Losses": "noncat_cov_c_fire",
    "CAT Cov C Smoke -  Incurred Losses": "cat_cov_c_smoke",
    "Non-CAT Cov C Smoke -  Incurred Losses": "noncat_cov_c_smoke",
    "CAT Cov A Fire -  Number of Claims": "cat_fire_claims",
    "Non-CAT Cov A Fire -  Number of Claims": "noncat_fire_claims",
    "CAT Cov C Fire -  Number of Claims": "cat_c_fire_claims",
    "Non-CAT Cov C Fire -  Number of Claims": "noncat_c_fire_claims",
    "Avg Fire Risk Score": "avg_fire_risk_score",
    "Avg PPC": "avg_ppc",
    "Number of High Fire Risk Exposure": "n_high_risk",
    "Number of Low Fire Risk Exposure": "n_low_risk",
    "Number of Moderate Fire Risk Exposure": "n_moderate_risk",
    "Number of Negligible Fire Risk Exposure": "n_negligible_risk",
    "Number of Very High Fire Risk Exposure": "n_very_high_risk",
    "GIS_ACRES": "total_gis_acres",
}

TASK1A_BASE_COLS = [
    "ZIP", "Year",
    "avg_fire_risk_score",
    "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm",
    "log_gis_acres", "fire_occurred",
    "total_population", "median_income",
    "year_sin",
]

TASK1A_ENGINEERED_COLS = [
    "ZIP", "Year",
    "avg_fire_risk_score",
    "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm",
    "temp_range", "drought_proxy", "log_prcp",
    "log_gis_acres", "fire_occurred",
    "avg_ppc", "high_risk_pct", "low_risk_pct", "risk_concentration",
    "total_population", "poverty_rate", "log_median_income",
    "edu_rate", "vacancy_rate", "vulnerability_index",
    "year_sin", "year_cos",
]

TASK2_BASE_COLS = [
    "ZIP", "Year",
    "earned_premium", "earned_exposure",
    "cat_fire_losses", "noncat_fire_losses",
    "avg_fire_risk_score", "avg_ppc",
    "total_population", "median_income",
    "premium_rolling_mean", "year_sin", "year_cos",
]

TASK2_ENGINEERED_COLS = [
    "ZIP", "Year",
    "earned_premium",
    "total_cat_losses", "total_noncat_losses", "total_losses", "total_claims",
    "loss_ratio", "loss_per_exposure", "claim_frequency", "avg_loss_per_claim",
    "cat_share", "log_earned_premium", "log_total_losses",
    "avg_fire_risk_score", "avg_ppc",
    "high_risk_pct", "low_risk_pct", "risk_concentration",
    "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm",
    "temp_range", "heat_index", "drought_proxy", "dry_hot", "log_prcp",
    "log_gis_acres", "fire_occurred",
    "total_population", "median_income",
    "poverty_rate", "edu_rate", "vacancy_rate",
    "owner_rate", "renter_rate",
    "log_median_income", "log_housing_value",
    "income_to_housing", "housing_cost_burden",
    "vulnerability_index", "premium_yoy_growth",
    "year_sin", "year_cos", "premium_rolling_mean",
]


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    sum_cols = [c for c in INSURANCE_SUM_COLS + FIRE_AGG_COLS + ["_fire_flag"] if c in df.columns]
    mean_cols = [c for c in INSURANCE_MEAN_COLS + WEATHER_MEAN_COLS if c in df.columns]
    first_cols = [c for c in CENSUS_FIRST_COLS if c in df.columns]

    agg_dict = {}
    for col in sum_cols:
        agg_dict[col] = "sum"
    for col in mean_cols:
        agg_dict[col] = "mean"
    for col in first_cols:
        agg_dict.setdefault(col, "first")

    grouped = df.groupby(["ZIP", "Year"], as_index=False).agg(agg_dict)
    grouped.rename(columns={k: v for k, v in RENAME.items() if k in grouped.columns}, inplace=True)
    grouped.rename(columns={"_fire_flag": "fire_count"}, inplace=True)
    grouped["fire_occurred"] = (grouped["fire_count"] > 0).astype(int)
    return grouped


def _filter_complete_zips(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("ZIP")["Year"].nunique()
    keep = counts[counts == 4].index
    return df[df["ZIP"].isin(keep)].copy()


def _impute_zip(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = df.groupby("ZIP")[col].transform(lambda x: x.ffill().bfill())
        df[col] = df[col].fillna(df[col].median())
    return df


def _add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df["year_norm"] = df["Year"] - df["Year"].min()
    df["year_sin"] = np.sin(2 * np.pi * df["year_norm"] / 4)
    df["year_cos"] = np.cos(2 * np.pi * df["year_norm"] / 4)
    df = df.sort_values(["ZIP", "Year"])
    df["premium_rolling_mean"] = (
        df.groupby("ZIP")["earned_premium"]
        .transform(lambda x: x.expanding().mean().shift(1))
        .fillna(0)
    )
    return df


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(0.0, index=df.index)


def _build_full_aggregation(raw_path: Path) -> pd.DataFrame:
    print("=" * 60)
    print("  Building Aggregated Feature Table")
    print("=" * 60)

    df = pd.read_csv(raw_path, low_memory=False)
    print(f"Step 1 | {len(df):,} raw rows, {df['ZIP'].nunique():,} ZIPs")

    df["_fire_flag"] = (~df["GIS_ACRES"].isna()).astype(float)
    grouped = _aggregate(df)
    print(f"Step 2 | {len(grouped):,} aggregated ZIP-Year rows")

    grouped = _filter_complete_zips(grouped)
    print(f"Step 3 | {grouped['ZIP'].nunique():,} complete ZIP panels")

    grouped = _impute_zip(
        grouped,
        [
            "total_population", "median_income", "total_housing_units",
            "housing_vacancy_number", "poverty_status",
            "educational_attainment_bachelor_or_higher",
            "owner_occupied_housing_units", "renter_occupied_housing_units",
            "housing_occupancy_number", "housing_value",
            "median_monthly_housing_costs",
            "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm", "total_gis_acres",
        ],
    )
    grouped.dropna(subset=["earned_premium", "earned_exposure", "avg_fire_risk_score"], inplace=True)
    print(f"Step 4 | {len(grouped):,} rows after required-column filtering")

    grouped = _add_temporal(grouped)

    eps = 1e-9
    grouped["cat_fire_losses"] = _col(grouped, "cat_cov_a_fire")
    grouped["noncat_fire_losses"] = _col(grouped, "noncat_cov_a_fire")

    grouped["total_cat_losses"] = (
        _col(grouped, "cat_cov_a_fire") + _col(grouped, "cat_cov_c_fire")
        + _col(grouped, "cat_cov_a_smoke") + _col(grouped, "cat_cov_c_smoke")
    )
    grouped["total_noncat_losses"] = (
        _col(grouped, "noncat_cov_a_fire") + _col(grouped, "noncat_cov_c_fire")
        + _col(grouped, "noncat_cov_a_smoke") + _col(grouped, "noncat_cov_c_smoke")
    )
    grouped["total_losses"] = grouped["total_cat_losses"] + grouped["total_noncat_losses"]
    grouped["total_claims"] = (
        _col(grouped, "cat_fire_claims") + _col(grouped, "cat_c_fire_claims")
        + _col(grouped, "noncat_fire_claims") + _col(grouped, "noncat_c_fire_claims")
    )

    grouped["loss_ratio"] = grouped["total_losses"] / (grouped["earned_premium"] + eps)
    grouped["loss_per_exposure"] = grouped["total_losses"] / (grouped["earned_exposure"] + eps)
    grouped["claim_frequency"] = grouped["total_claims"] / (grouped["earned_exposure"] + eps)
    grouped["avg_loss_per_claim"] = grouped["total_losses"] / (grouped["total_claims"] + eps)
    grouped["cat_share"] = grouped["total_cat_losses"] / (grouped["total_losses"] + eps)
    grouped["log_earned_premium"] = np.log1p(grouped["earned_premium"].clip(lower=0))
    grouped["log_total_losses"] = np.log1p(grouped["total_losses"].clip(lower=0))

    risk_cols = [c for c in ["n_high_risk", "n_low_risk", "n_moderate_risk", "n_negligible_risk", "n_very_high_risk"] if c in grouped.columns]
    if risk_cols:
        total_risk = grouped[risk_cols].sum(axis=1).replace(0, np.nan)
        pct_very_high = _col(grouped, "n_very_high_risk") / (total_risk + eps)
        pct_high = _col(grouped, "n_high_risk") / (total_risk + eps)
        pct_low = _col(grouped, "n_low_risk") / (total_risk + eps)
        pct_negligible = _col(grouped, "n_negligible_risk") / (total_risk + eps)
        grouped["high_risk_pct"] = (pct_very_high + pct_high).fillna(0)
        grouped["low_risk_pct"] = (pct_low + pct_negligible).fillna(0)
        grouped["risk_concentration"] = (pct_very_high - pct_negligible).fillna(0).clip(-1, 1)
    else:
        grouped["high_risk_pct"] = 0.0
        grouped["low_risk_pct"] = 0.0
        grouped["risk_concentration"] = 0.0

    grouped["temp_range"] = grouped["avg_tmax_c"] - grouped["avg_tmin_c"]
    grouped["heat_index"] = grouped["avg_tmax_c"] * (1 - grouped["tot_prcp_mm"].clip(upper=50) / 51)
    grouped["drought_proxy"] = grouped["temp_range"] / (grouped["tot_prcp_mm"] + 1)
    grouped["dry_hot"] = ((grouped["avg_tmax_c"] > 30) & (grouped["tot_prcp_mm"] < 5)).astype(int)
    grouped["log_prcp"] = np.log1p(grouped["tot_prcp_mm"].clip(lower=0))
    grouped["log_gis_acres"] = np.log1p(grouped["total_gis_acres"].clip(lower=0))

    grouped["poverty_rate"] = _col(grouped, "poverty_status") / (grouped["total_population"] + eps)
    grouped["edu_rate"] = _col(grouped, "educational_attainment_bachelor_or_higher") / (grouped["total_population"] + eps)
    grouped["vacancy_rate"] = _col(grouped, "housing_vacancy_number") / (_col(grouped, "total_housing_units") + eps)
    grouped["owner_rate"] = _col(grouped, "owner_occupied_housing_units") / (_col(grouped, "housing_occupancy_number") + eps)
    grouped["renter_rate"] = _col(grouped, "renter_occupied_housing_units") / (_col(grouped, "housing_occupancy_number") + eps)
    grouped["log_median_income"] = np.log1p(grouped["median_income"].clip(lower=0))
    grouped["log_housing_value"] = np.log1p(_col(grouped, "housing_value").clip(lower=0))
    grouped["income_to_housing"] = grouped["median_income"] / (_col(grouped, "housing_value") + eps)
    grouped["housing_cost_burden"] = (_col(grouped, "median_monthly_housing_costs") * 12) / (grouped["median_income"] + eps)
    grouped["premium_yoy_growth"] = grouped.groupby("ZIP")["earned_premium"].pct_change()

    vuln_cols = ["avg_fire_risk_score", "poverty_rate", "drought_proxy", "vacancy_rate", "housing_cost_burden"]
    mask = grouped[vuln_cols].notna().all(axis=1)
    if mask.sum() > 100:
        grouped.loc[mask, "vulnerability_index"] = StandardScaler().fit_transform(grouped.loc[mask, vuln_cols]).mean(axis=1)

    print("Step 5 | derived engineered features")
    return grouped.sort_values(["ZIP", "Year"]).reset_index(drop=True)


def _write(df: pd.DataFrame, cols: list[str], out_path: Path) -> pd.DataFrame:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = df[[c for c in cols if c in df.columns]].reset_index(drop=True)
    result.to_csv(out_path, index=False)
    print(f"[write] {out_path} ({len(result):,} rows, {len(result.columns) - 2} features)")
    return result


def preprocess_task1a_minimal(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    grouped = _build_full_aggregation(raw_path)
    return _write(grouped, TASK1A_BASE_COLS, TASK1A_DIR / "wildfire_risk_minimal.csv")


def preprocess_task1a_extended(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    grouped = _build_full_aggregation(raw_path)
    return _write(grouped, TASK1A_ENGINEERED_COLS, TASK1A_DIR / "wildfire_risk_extended.csv")


def preprocess_task2_minimal(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    grouped = _build_full_aggregation(raw_path)
    return _write(grouped, TASK2_BASE_COLS, TASK2_DIR / "insurance_minimal.csv")


def preprocess_task2_extended(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    grouped = _build_full_aggregation(raw_path)
    return _write(grouped, TASK2_ENGINEERED_COLS, TASK2_DIR / "insurance_extended.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess the raw wildfire/insurance join into task-specific datasets."
    )
    parser.add_argument("--task", choices=["task1a", "task2", "all"], default="all")
    parser.add_argument("--no-extended", action="store_true", help="Skip extended feature variants.")
    args = parser.parse_args()

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found: {RAW_PATH}")

    run_task1a = args.task in ("task1a", "all")
    run_task2 = args.task in ("task2", "all")
    run_ext = not args.no_extended

    if run_task1a:
        preprocess_task1a_minimal()
        if run_ext:
            preprocess_task1a_extended()

    if run_task2:
        preprocess_task2_minimal()
        if run_ext:
            preprocess_task2_extended()

    print("\nDone.")
