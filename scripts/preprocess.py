"""
preprocess.py
=============
Preprocessing pipeline for the Quantum Sustainability Challenge dataset.

Modes
-----
  default     →  data/wildfire_preprocessed.csv   (9 features)
  --engineered →  data/wildfire_engineered.csv    (14 EDA-derived features)

Usage
-----
  python scripts/preprocess.py                  # base dataset
  python scripts/preprocess.py --engineered     # extended engineered dataset

RAW DATA STRUCTURE
──────────────────
The raw CSV is a three-way JOIN of:
  1. Insurance table   – one row per (ZIP, Year, policy category)
  2. Wildfire table    – one row per wildfire event
  3. Census + Weather  – one row per (ZIP, year_month)

One ZIP-Year pair can appear 5-84 times in the raw file.

BASE PIPELINE (wildfire_preprocessed.csv)
──────────────────────────────────────────
  Step 1  Aggregate to one row per (ZIP, Year)
  Step 2  Keep ZIPs present in ALL four years (2018-2021) → 1,906 ZIPs
  Step 3  Impute nulls in census columns
  Step 4  Cyclical year encoding (sin/cos on 4-year cycle)
  Step 5  Rolling-mean premium feature (lagged, per ZIP)

ENGINEERED PIPELINE (wildfire_engineered.csv)
──────────────────────────────────────────────
  Runs the same Steps 1-5, then additionally derives:
    • Insurance ratios  : loss_ratio, claim_frequency
    • Risk composition  : high_risk_pct, risk_concentration
    • Weather features  : temp_range, drought_proxy, log_prcp
    • Socioeconomic     : poverty_rate, log_median_income,
                          vacancy_rate, housing_cost_burden
  Weather / socioeconomic nulls are imputed by ZIP-wise ffill/bfill
  then global median fallback.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────── paths ────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent   # project root
DATA_DIR  = ROOT / "data"
RAW_PATH  = DATA_DIR / "abfa2rbci2UF6CTj_cal_insurance_fire_census_weather.csv"
BASE_PATH = DATA_DIR / "wildfire_preprocessed.csv"
ENG_PATH  = DATA_DIR / "wildfire_engineered.csv"

# ─────────────────── aggregation specs ────────────────────────────────────
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
WEATHER_MEAN_COLS   = ["avg_tmax_c", "avg_tmin_c", "tot_prcp_mm"]
FIRE_AGG_COLS       = ["GIS_ACRES"]   # sum  → total burned acres per ZIP-Year
CENSUS_FIRST_COLS   = [
    "total_population", "median_income",
    "total_housing_units", "housing_vacancy_number",
    "educational_attainment_bachelor_or_higher", "poverty_status",
    "owner_occupied_housing_units", "renter_occupied_housing_units",
    "housing_occupancy_number", "housing_value",
    "median_monthly_housing_costs",
]

RENAME = {
    "Earned Premium":                          "earned_premium",
    "Earned Exposure":                         "earned_exposure",
    "CAT Cov A Fire -  Incurred Losses":       "cat_cov_a_fire",
    "Non-CAT Cov A Fire -  Incurred Losses":   "noncat_cov_a_fire",
    "CAT Cov A Smoke -  Incurred Losses":      "cat_cov_a_smoke",
    "Non-CAT Cov A Smoke -  Incurred Losses":  "noncat_cov_a_smoke",
    "CAT Cov C Fire -  Incurred Losses":       "cat_cov_c_fire",
    "Non-CAT Cov C Fire -  Incurred Losses":   "noncat_cov_c_fire",
    "CAT Cov C Smoke -  Incurred Losses":      "cat_cov_c_smoke",
    "Non-CAT Cov C Smoke -  Incurred Losses":  "noncat_cov_c_smoke",
    "CAT Cov A Fire -  Number of Claims":      "cat_fire_claims",
    "Non-CAT Cov A Fire -  Number of Claims":  "noncat_fire_claims",
    "CAT Cov C Fire -  Number of Claims":      "cat_c_fire_claims",
    "Non-CAT Cov C Fire -  Number of Claims":  "noncat_c_fire_claims",
    "Avg Fire Risk Score":                     "avg_fire_risk_score",
    "Avg PPC":                                 "avg_ppc",
    "Number of High Fire Risk Exposure":       "n_high_risk",
    "Number of Low Fire Risk Exposure":        "n_low_risk",
    "Number of Moderate Fire Risk Exposure":   "n_moderate_risk",
    "Number of Negligible Fire Risk Exposure": "n_negligible_risk",
    "Number of Very High Fire Risk Exposure":  "n_very_high_risk",
    "GIS_ACRES":                               "total_gis_acres",
}

# base pipeline uses only these two rename entries
RENAME_BASE = {
    "Earned Premium":                        "earned_premium",
    "Earned Exposure":                       "earned_exposure",
    "CAT Cov A Fire -  Incurred Losses":     "cat_fire_losses",
    "Non-CAT Cov A Fire -  Incurred Losses": "noncat_fire_losses",
    "Avg Fire Risk Score":                   "avg_fire_risk_score",
    "Avg PPC":                               "avg_ppc",
}

REQUIRED_BASE = [
    "earned_premium", "earned_exposure",
    "cat_fire_losses", "noncat_fire_losses",
    "avg_fire_risk_score", "total_population", "median_income",
]


# ──────────────────────── shared aggregation steps ────────────────────────

def _aggregate(df: pd.DataFrame, rename_map: dict,
               extra_sum: list, extra_mean: list,
               extra_first: list) -> pd.DataFrame:
    """Aggregate raw rows to one row per (ZIP, Year)."""
    all_sum   = [c for c in INSURANCE_SUM_COLS + extra_sum   if c in df.columns]
    all_mean  = [c for c in INSURANCE_MEAN_COLS + extra_mean if c in df.columns]
    all_first = [c for c in CENSUS_FIRST_COLS   + extra_first if c in df.columns]

    agg_dict = {}
    for c in all_sum:   agg_dict[c] = "sum"
    for c in all_mean:  agg_dict[c] = "mean"
    for c in all_first: agg_dict.setdefault(c, "first")

    grouped = df.groupby(["ZIP", "Year"], as_index=False).agg(agg_dict)
    grouped.rename(columns={k: v for k, v in rename_map.items() if k in grouped.columns},
                   inplace=True)
    return grouped


def _filter_complete_zips(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only ZIPs that have all four years (2018-2021)."""
    year_counts   = df.groupby("ZIP")["Year"].nunique()
    complete_zips = year_counts[year_counts == 4].index
    df = df[df["ZIP"].isin(complete_zips)].copy()
    print(f"         {len(complete_zips):,} ZIPs kept  →  {len(df):,} rows")
    return df


def _impute_zip(df: pd.DataFrame, cols: list, label: str) -> pd.DataFrame:
    """ZIP-wise ffill/bfill then global median fallback."""
    for col in cols:
        if col not in df.columns:
            continue
        n_before = df[col].isna().sum()
        df[col] = df.groupby("ZIP")[col].transform(lambda x: x.ffill().bfill())
        df[col] = df[col].fillna(df[col].median())
        n_after = df[col].isna().sum()
        if n_before:
            print(f"         {label} '{col}': {n_before} nulls → {n_after}")
    return df


def _add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical year encoding and lagged rolling-mean premium."""
    df["year_norm"] = df["Year"] - df["Year"].min()
    df["year_sin"]  = np.sin(2 * np.pi * df["year_norm"] / 4)
    df["year_cos"]  = np.cos(2 * np.pi * df["year_norm"] / 4)
    df = df.sort_values(["ZIP", "Year"])
    df["premium_rolling_mean"] = (
        df.groupby("ZIP")["earned_premium"]
        .transform(lambda x: x.expanding().mean().shift(1))
        .fillna(0)
    )
    return df


# ──────────────────────────── base pipeline ───────────────────────────────

def preprocess_base(raw_path: Path = RAW_PATH,
                    save_path: Path = BASE_PATH) -> pd.DataFrame:
    print("Step 1 | Loading raw data …")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"         {len(df):,} rows, {df['ZIP'].nunique():,} unique ZIPs, "
          f"years {sorted(df['Year'].unique())}")

    print("Step 2 | Aggregating per (ZIP, Year) …")
    grouped = _aggregate(df, rename_map=RENAME_BASE,
                         extra_sum=[], extra_mean=[], extra_first=[])
    print(f"         {len(grouped):,} rows after aggregation")

    print("Step 3 | Filtering ZIPs with all 4 years …")
    grouped = _filter_complete_zips(grouped)

    print("Step 4 | Imputing nulls …")
    grouped = _impute_zip(grouped, ["total_population", "median_income"], "census")
    grouped.dropna(subset=REQUIRED_BASE, inplace=True)

    print("Step 5 | Cyclical year encoding + rolling premium …")
    grouped = _add_temporal(grouped)

    grouped.reset_index(drop=True, inplace=True)
    out_cols = ["ZIP", "Year", "earned_premium", "earned_exposure",
                "cat_fire_losses", "noncat_fire_losses",
                "CAT Cov A Smoke -  Incurred Losses" if "CAT Cov A Smoke -  Incurred Losses" in grouped.columns else None,
                "avg_fire_risk_score", "avg_ppc",
                "total_population", "median_income",
                "year_norm", "year_sin", "year_cos", "premium_rolling_mean"]
    out_cols = [c for c in out_cols if c and c in grouped.columns]
    grouped[out_cols].to_csv(save_path, index=False)
    print(f"\nSaved {len(grouped):,} rows  →  {save_path}")

    print("\n── Feature Summary ─────────────────────────────────────────────")
    display_cols = [c for c in REQUIRED_BASE + ["year_sin", "year_cos", "premium_rolling_mean"]
                    if c in grouped.columns]
    print(grouped[display_cols].describe().round(1).to_string())
    return grouped


# ──────────────────────────── engineered pipeline ─────────────────────────

def preprocess_engineered(raw_path: Path = RAW_PATH,
                           save_path: Path = ENG_PATH) -> pd.DataFrame:
    """
    Full EDA-faithful engineered pipeline.
    Mirrors all feature derivations from scripts/eda.py, applied to
    properly aggregated (one row per ZIP-Year) data.
    """
    from sklearn.preprocessing import StandardScaler

    print("Step 1 | Loading raw data …")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"         {len(df):,} rows, {df['ZIP'].nunique():,} unique ZIPs")

    # ── Step 2: Aggregate ─────────────────────────────────────────────────
    print("Step 2 | Aggregating per (ZIP, Year) …")
    # fire_occurred: 1 if any fire row exists for that ZIP-Year
    df["_fire_flag"] = (~df["GIS_ACRES"].isna()).astype(float)
    grouped = _aggregate(df, rename_map=RENAME,
                         extra_sum=FIRE_AGG_COLS + ["_fire_flag"],
                         extra_mean=WEATHER_MEAN_COLS,
                         extra_first=[])
    grouped.rename(columns={"_fire_flag": "fire_count"}, inplace=True)
    grouped["fire_occurred"] = (grouped["fire_count"] > 0).astype(int)
    print(f"         {len(grouped):,} rows after aggregation")

    # ── Step 3: Filter complete ZIPs ──────────────────────────────────────
    print("Step 3 | Filtering ZIPs with all 4 years …")
    grouped = _filter_complete_zips(grouped)

    # ── Step 4: Impute nulls ──────────────────────────────────────────────
    print("Step 4 | Imputing nulls …")
    census_impute = ["total_population", "median_income", "total_housing_units",
                     "housing_vacancy_number", "poverty_status",
                     "educational_attainment_bachelor_or_higher",
                     "owner_occupied_housing_units", "renter_occupied_housing_units",
                     "housing_occupancy_number", "housing_value",
                     "median_monthly_housing_costs"]
    grouped = _impute_zip(grouped, census_impute, "census")
    grouped = _impute_zip(grouped, WEATHER_MEAN_COLS, "weather")
    grouped = _impute_zip(grouped, ["total_gis_acres"], "fire")
    grouped.dropna(subset=["earned_premium", "earned_exposure",
                            "avg_fire_risk_score"], inplace=True)

    # ── Step 5: Temporal features ─────────────────────────────────────────
    print("Step 5 | Cyclical year encoding + rolling premium …")
    grouped = _add_temporal(grouped)

    # ── Step 6: Derive all EDA features ───────────────────────────────────
    print("Step 6 | Deriving EDA features …")
    eps = 1e-9

    # Insurance loss & claim aggregates (matching eda.py exactly)
    def _col(name):
        return grouped[name] if name in grouped.columns else pd.Series(0, index=grouped.index)

    grouped["total_cat_losses"]    = (_col("cat_cov_a_fire") + _col("cat_cov_c_fire")
                                      + _col("cat_cov_a_smoke") + _col("cat_cov_c_smoke"))
    grouped["total_noncat_losses"] = (_col("noncat_cov_a_fire") + _col("noncat_cov_c_fire")
                                      + _col("noncat_cov_a_smoke") + _col("noncat_cov_c_smoke"))
    grouped["total_losses"]        = grouped["total_cat_losses"] + grouped["total_noncat_losses"]
    grouped["total_claims"]        = (_col("cat_fire_claims") + _col("cat_c_fire_claims")
                                      + _col("noncat_fire_claims") + _col("noncat_c_fire_claims"))

    grouped["loss_ratio"]         = grouped["total_losses"]      / (grouped["earned_premium"]  + eps)
    grouped["loss_per_exposure"]  = grouped["total_losses"]      / (grouped["earned_exposure"] + eps)
    grouped["claim_frequency"]    = grouped["total_claims"]      / (grouped["earned_exposure"] + eps)
    grouped["avg_loss_per_claim"] = grouped["total_losses"]      / (grouped["total_claims"]    + eps)
    grouped["cat_share"]          = grouped["total_cat_losses"]  / (grouped["total_losses"]    + eps)
    grouped["log_earned_premium"] = np.log1p(grouped["earned_premium"].clip(lower=0))
    grouped["log_total_losses"]   = np.log1p(grouped["total_losses"].clip(lower=0))

    # Risk tier composition
    risk_raw = ["n_high_risk", "n_low_risk", "n_moderate_risk",
                "n_negligible_risk", "n_very_high_risk"]
    present_risk = [c for c in risk_raw if c in grouped.columns]
    if present_risk:
        total_risk     = grouped[present_risk].sum(axis=1).replace(0, np.nan)
        pct_very_high  = _col("n_very_high_risk")  / (total_risk + eps)
        pct_high       = _col("n_high_risk")        / (total_risk + eps)
        pct_low        = _col("n_low_risk")         / (total_risk + eps)
        pct_negligible = _col("n_negligible_risk")  / (total_risk + eps)
        grouped["high_risk_pct"]      = (pct_very_high + pct_high).fillna(0)
        grouped["low_risk_pct"]       = (pct_low + pct_negligible).fillna(0)
        grouped["risk_concentration"] = (pct_very_high - pct_negligible).fillna(0).clip(-1, 1)
    else:
        grouped["high_risk_pct"]      = 0.0
        grouped["low_risk_pct"]       = 0.0
        grouped["risk_concentration"] = 0.0

    # Weather-derived (matching eda.py)
    grouped["temp_range"]    = grouped["avg_tmax_c"] - grouped["avg_tmin_c"]
    grouped["heat_index"]    = grouped["avg_tmax_c"] * (
                                1 - grouped["tot_prcp_mm"].clip(upper=50) / 51)
    grouped["drought_proxy"] = grouped["temp_range"] / (grouped["tot_prcp_mm"] + 1)
    grouped["dry_hot"]       = ((grouped["avg_tmax_c"] > 30)
                                & (grouped["tot_prcp_mm"] < 5)).astype(int)
    grouped["log_prcp"]      = np.log1p(grouped["tot_prcp_mm"].clip(lower=0))

    # Fire history
    grouped["log_gis_acres"] = np.log1p(grouped["total_gis_acres"].clip(lower=0))

    # Socioeconomic (matching eda.py)
    grouped["poverty_rate"]        = _col("poverty_status") / (grouped["total_population"] + eps)
    grouped["edu_rate"]            = (_col("educational_attainment_bachelor_or_higher")
                                       / (grouped["total_population"] + eps))
    grouped["vacancy_rate"]        = (_col("housing_vacancy_number")
                                       / (grouped["total_housing_units"] + eps))
    grouped["owner_rate"]          = (_col("owner_occupied_housing_units")
                                       / (grouped["housing_occupancy_number"] + eps))
    grouped["renter_rate"]         = (_col("renter_occupied_housing_units")
                                       / (grouped["housing_occupancy_number"] + eps))
    grouped["log_median_income"]   = np.log1p(grouped["median_income"].clip(lower=0))
    grouped["log_housing_value"]   = np.log1p(_col("housing_value").clip(lower=0))
    grouped["income_to_housing"]   = grouped["median_income"] / (_col("housing_value") + eps)
    grouped["housing_cost_burden"] = (_col("median_monthly_housing_costs") * 12
                                       / (grouped["median_income"] + eps))

    # Year-over-year premium growth (per ZIP, matching eda.py)
    grouped = grouped.sort_values(["ZIP", "Year"])
    grouped["premium_yoy_growth"] = (grouped.groupby("ZIP")["earned_premium"]
                                             .pct_change())

    # Composite vulnerability index (matching eda.py)
    vuln_cols = ["avg_fire_risk_score", "poverty_rate", "drought_proxy",
                 "vacancy_rate", "housing_cost_burden"]
    mask_vuln = grouped[vuln_cols].notna().all(axis=1)
    if mask_vuln.sum() > 100:
        grouped.loc[mask_vuln, "vulnerability_index"] = (
            StandardScaler()
            .fit_transform(grouped.loc[mask_vuln, vuln_cols])
            .mean(axis=1)
        )

    # Clip extremes
    grouped["loss_ratio"]          = grouped["loss_ratio"].clip(0, 50)
    grouped["claim_frequency"]     = grouped["claim_frequency"].clip(0, 10)
    grouped["avg_loss_per_claim"]  = grouped["avg_loss_per_claim"].clip(0)
    grouped["drought_proxy"]       = grouped["drought_proxy"].clip(0, 100)
    grouped["housing_cost_burden"] = grouped["housing_cost_burden"].clip(0, 5)
    grouped["poverty_rate"]        = grouped["poverty_rate"].clip(0, 1)
    grouped["edu_rate"]            = grouped["edu_rate"].clip(0, 1)
    grouped["vacancy_rate"]        = grouped["vacancy_rate"].clip(0, 1)
    grouped["owner_rate"]          = grouped["owner_rate"].clip(0, 1)
    grouped["renter_rate"]         = grouped["renter_rate"].clip(0, 1)
    grouped["income_to_housing"]   = grouped["income_to_housing"].clip(0)

    # ── Select output columns ─────────────────────────────────────────────
    out_cols = [
        "ZIP", "Year",
        # target
        "earned_premium",
        # insurance
        "total_cat_losses", "total_noncat_losses", "total_losses", "total_claims",
        "loss_ratio", "loss_per_exposure", "claim_frequency", "avg_loss_per_claim",
        "cat_share", "log_earned_premium", "log_total_losses",
        # risk composition
        "avg_fire_risk_score", "avg_ppc",
        "high_risk_pct", "low_risk_pct", "risk_concentration",
        # weather
        "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm",
        "temp_range", "heat_index", "drought_proxy", "dry_hot", "log_prcp",
        # fire history
        "log_gis_acres", "fire_occurred",
        # socioeconomic
        "total_population", "median_income",
        "poverty_rate", "edu_rate", "vacancy_rate",
        "owner_rate", "renter_rate",
        "log_median_income", "log_housing_value",
        "income_to_housing", "housing_cost_burden",
        # composite / temporal
        "vulnerability_index", "premium_yoy_growth",
        "year_sin", "year_cos", "premium_rolling_mean",
    ]
    out_cols = [c for c in out_cols if c in grouped.columns]
    result = grouped[out_cols].reset_index(drop=True)
    result.to_csv(save_path, index=False)

    n_feats = len(out_cols) - 2   # exclude ZIP and Year
    print(f"\nSaved {len(result):,} rows, {n_feats} columns  →  {save_path}")
    print("\n── Feature Summary ─────────────────────────────────────────────")
    feat_display = [c for c in out_cols if c not in ("ZIP", "Year")]
    print(result[feat_display].describe().round(3).to_string())
    return result


# ──────────────────────────────── CLI ─────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess wildfire dataset.")
    parser.add_argument("--engineered", action="store_true",
                        help="Generate the extended engineered feature dataset "
                             "(data/wildfire_engineered.csv) instead of the base one.")
    args = parser.parse_args()

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found: {RAW_PATH}")

    if args.engineered:
        print("=" * 60)
        print("  Engineered pipeline  →  wildfire_engineered.csv")
        print("=" * 60)
        preprocess_engineered()
    else:
        print("=" * 60)
        print("  Base pipeline  →  wildfire_preprocessed.csv")
        print("=" * 60)
        preprocess_base()
