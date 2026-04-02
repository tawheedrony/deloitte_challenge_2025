"""
Extensive EDA & Statistical Analysis for California Wildfire Insurance Dataset.
Extracts engineered features and saves all figures to output/viz/.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pointbiserialr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import seaborn as sns

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("/home/tawheed/exp/project_qml/QLSTM/data")
OUT_PATH   = Path("/home/tawheed/exp/project_qml/QLSTM/output/viz")
OUT_PATH.mkdir(parents=True, exist_ok=True)

MAIN_CSV   = DATA_PATH / "abfa2rbci2UF6CTj_cal_insurance_fire_census_weather.csv"
PROC_CSV   = DATA_PATH / "wildfire_preprocessed.csv"
FEAT_OUT   = DATA_PATH / "wildfire_features_engineered.csv"

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
SAVE_DPI   = 150

def savefig(name: str):
    plt.tight_layout()
    plt.savefig(OUT_PATH / f"{name}.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {name}.png")

# ─── Load ──────────────────────────────────────────────────────────────────────
print("Loading data …")
raw = pd.read_csv(MAIN_CSV, low_memory=False)
pre = pd.read_csv(PROC_CSV)

# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Engineering features …")
df = raw.copy()

# --- Insurance loss ratios ---------------------------------------------------
eps = 1e-9
df["total_cat_losses"]    = (df["CAT Cov A Fire -  Incurred Losses"]
                             + df["CAT Cov C Fire -  Incurred Losses"]
                             + df["CAT Cov A Smoke -  Incurred Losses"]
                             + df["CAT Cov C Smoke -  Incurred Losses"])
df["total_noncat_losses"] = (df["Non-CAT Cov A Fire -  Incurred Losses"]
                             + df["Non-CAT Cov C Fire -  Incurred Losses"]
                             + df["Non-CAT Cov A Smoke -  Incurred Losses"]
                             + df["Non-CAT Cov C Smoke -  Incurred Losses"])
df["total_losses"]        = df["total_cat_losses"] + df["total_noncat_losses"]
df["total_claims"]        = (df["CAT Cov A Fire -  Number of Claims"]
                             + df["CAT Cov C Fire -  Number of Claims"]
                             + df["Non-CAT Cov A Fire -  Number of Claims"]
                             + df["Non-CAT Cov C Fire -  Number of Claims"])
df["loss_ratio"]          = df["total_losses"] / (df["Earned Premium"] + eps)
df["loss_per_exposure"]   = df["total_losses"] / (df["Earned Exposure"] + eps)
df["claim_frequency"]     = df["total_claims"]  / (df["Earned Exposure"] + eps)
df["avg_loss_per_claim"]  = df["total_losses"]  / (df["total_claims"] + eps)
df["cat_share"]           = df["total_cat_losses"] / (df["total_losses"] + eps)
df["log_earned_premium"]  = np.log1p(df["Earned Premium"].clip(lower=0))
df["log_total_losses"]    = np.log1p(df["total_losses"].clip(lower=0))

# --- Fire risk composition --------------------------------------------------
risk_cols = ["Number of High Fire Risk Exposure",
             "Number of Low Fire Risk Exposure",
             "Number of Moderate Fire Risk Exposure",
             "Number of Negligible Fire Risk Exposure",
             "Number of Very High Fire Risk Exposure"]
df["total_risk_count"] = df[risk_cols].sum(axis=1)
for c in risk_cols:
    short = c.replace("Number of ", "").replace(" Fire Risk Exposure", "").lower().replace(" ", "_")
    df[f"pct_{short}"] = df[c] / (df["total_risk_count"] + eps)
df["high_risk_pct"]     = df["pct_high"] + df["pct_very_high"]
df["low_risk_pct"]      = df["pct_low"] + df["pct_negligible"]
df["risk_concentration"] = (df["pct_very_high"] - df["pct_negligible"]).clip(-1, 1)

# --- Weather-derived features -----------------------------------------------
df["temp_range"]     = df["avg_tmax_c"] - df["avg_tmin_c"]
df["heat_index"]     = df["avg_tmax_c"] * (1 - df["tot_prcp_mm"].clip(upper=50) / 51)
df["drought_proxy"]  = (df["avg_tmax_c"] - df["avg_tmin_c"]) / (df["tot_prcp_mm"] + 1)
df["dry_hot"]        = ((df["avg_tmax_c"] > 30) & (df["tot_prcp_mm"] < 5)).astype(int)
df["log_prcp"]       = np.log1p(df["tot_prcp_mm"])

# --- Fire history features --------------------------------------------------
df["log_gis_acres"] = np.log1p(df["GIS_ACRES"])
df["fire_occurred"] = (~df["GIS_ACRES"].isna()).astype(int)

if "ALARM_DATE" in df.columns:
    df["alarm_dt"] = pd.to_datetime(df["ALARM_DATE"], errors="coerce")
    df["fire_month"]  = df["alarm_dt"].dt.month
    df["fire_season"] = pd.cut(df["fire_month"],
                               bins=[0,3,6,9,12],
                               labels=["Winter","Spring","Summer","Fall"])

# --- Census / Socioeconomic -------------------------------------------------
df["pop_density_proxy"] = df["total_population"] / (df["total_housing_units"] + eps)
df["poverty_rate"]      = df["poverty_status"] / (df["total_population"] + eps)
df["edu_rate"]          = df["educational_attainment_bachelor_or_higher"] / (df["total_population"] + eps)
df["vacancy_rate"]      = df["housing_vacancy_number"] / (df["total_housing_units"] + eps)
df["owner_rate"]        = df["owner_occupied_housing_units"] / (df["housing_occupancy_number"] + eps)
df["renter_rate"]       = df["renter_occupied_housing_units"] / (df["housing_occupancy_number"] + eps)
df["log_median_income"] = np.log1p(df["median_income"])
df["log_housing_value"] = np.log1p(df["housing_value"])
df["income_to_housing"] = df["median_income"] / (df["housing_value"] + eps)
df["housing_cost_burden"]= df["median_monthly_housing_costs"] * 12 / (df["median_income"] + eps)

# --- Composite vulnerability index ------------------------------------------
scaler = StandardScaler()
vuln_cols = ["Avg Fire Risk Score", "poverty_rate", "drought_proxy",
             "vacancy_rate", "housing_cost_burden"]
mask_vuln = df[vuln_cols].notna().all(axis=1)
if mask_vuln.sum() > 100:
    df.loc[mask_vuln, "vulnerability_index"] = (
        scaler.fit_transform(df.loc[mask_vuln, vuln_cols]).mean(axis=1)
    )

# --- Year-over-year premium growth (per ZIP) --------------------------------
df_sorted = df.sort_values(["ZIP", "Year"])
df_sorted["premium_yoy_growth"] = df_sorted.groupby("ZIP")["Earned Premium"].pct_change()
df["premium_yoy_growth"] = df_sorted["premium_yoy_growth"].values

print(f"  Feature columns added: {df.shape[1] - raw.shape[1]}")

# Save engineered feature CSV
feat_cols = ["Year", "ZIP",
             "total_cat_losses", "total_noncat_losses", "total_losses", "total_claims",
             "loss_ratio", "loss_per_exposure", "claim_frequency", "avg_loss_per_claim",
             "cat_share", "log_earned_premium", "log_total_losses",
             "high_risk_pct", "low_risk_pct", "risk_concentration",
             "temp_range", "heat_index", "drought_proxy", "dry_hot", "log_prcp",
             "log_gis_acres", "fire_occurred",
             "poverty_rate", "edu_rate", "vacancy_rate", "owner_rate", "renter_rate",
             "log_median_income", "log_housing_value", "income_to_housing",
             "housing_cost_burden", "premium_yoy_growth",
             "Avg Fire Risk Score", "Avg PPC",
             "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm",
             "total_population", "median_income", "housing_value"]
df[feat_cols].to_csv(FEAT_OUT, index=False)
print(f"  Engineered features saved → {FEAT_OUT.name}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. MISSING-VALUE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Missing value analysis …")
miss = (df.isnull().mean() * 100).sort_values(ascending=False)
miss = miss[miss > 0].head(35)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(miss.index[::-1], miss.values[::-1], color=sns.color_palette("Reds_r", len(miss)))
ax.set_xlabel("Missing (%)")
ax.set_title("Missing Value Rate by Column (top 35)")
for bar, val in zip(bars, miss.values[::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=8)
savefig("01_missing_values")

# ══════════════════════════════════════════════════════════════════════════════
# 3. DISTRIBUTION OF KEY FINANCIAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Financial feature distributions …")
fin_feats = {
    "Earned Premium (log1p)": "log_earned_premium",
    "Total Losses (log1p)":   "log_total_losses",
    "Loss Ratio":             "loss_ratio",
    "CAT Share of Losses":    "cat_share",
    "Claim Frequency":        "claim_frequency",
    "Avg Loss per Claim":     "avg_loss_per_claim",
}
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (title, col) in zip(axes.flat, fin_feats.items()):
    data = df[col].dropna()
    # clip extreme outliers for visualisation
    lo, hi = data.quantile(0.01), data.quantile(0.99)
    data = data.clip(lo, hi)
    sns.histplot(data, kde=True, ax=ax, bins=50, color="#4878CF")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("")
    sk = stats.skew(data)
    ku = stats.kurtosis(data)
    ax.text(0.97, 0.95, f"skew={sk:.2f}\nkurt={ku:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
savefig("02_financial_distributions")

# ══════════════════════════════════════════════════════════════════════════════
# 4. FIRE RISK SCORE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Fire risk score analysis …")
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# 4a. Distribution
sns.histplot(df["Avg Fire Risk Score"].dropna().clip(-0.2, 4), bins=60,
             kde=True, ax=axes[0], color="#E74C3C")
axes[0].set_title("Fire Risk Score Distribution")
axes[0].set_xlabel("Avg Fire Risk Score")

# 4b. By year
yr_risk = df.groupby("Year")["Avg Fire Risk Score"].agg(["mean","std","median"]).reset_index()
axes[1].errorbar(yr_risk["Year"], yr_risk["mean"], yerr=yr_risk["std"],
                 fmt="o-", capsize=4, color="#E74C3C", linewidth=2)
axes[1].plot(yr_risk["Year"], yr_risk["median"], "s--", color="#8B0000",
             alpha=0.7, label="median")
axes[1].set_title("Fire Risk Score by Year")
axes[1].set_xlabel("Year"); axes[1].legend()

# 4c. Risk pct composition stacked bar
risk_pct_cols = ["pct_very_high","pct_high","pct_moderate","pct_low","pct_negligible"]
risk_labels    = ["Very High","High","Moderate","Low","Negligible"]
colors_risk    = ["#8B0000","#E74C3C","#F39C12","#27AE60","#2980B9"]
risk_agg = df.groupby("Year")[risk_pct_cols].mean()
risk_agg.plot(kind="bar", stacked=True, ax=axes[2], color=colors_risk,
              legend=True)
axes[2].set_title("Avg Risk Tier Composition by Year")
axes[2].set_xlabel("Year"); axes[2].set_ylabel("Mean Share")
axes[2].legend(risk_labels, loc="upper right", fontsize=8)
savefig("03_fire_risk_analysis")

# ══════════════════════════════════════════════════════════════════════════════
# 5. WEATHER FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Weather feature analysis …")
weather_df = df[["avg_tmax_c","avg_tmin_c","tot_prcp_mm","temp_range",
                  "drought_proxy","heat_index","Year"]].dropna()

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
pairs = [("avg_tmax_c","Max Temp (°C)"),("avg_tmin_c","Min Temp (°C)"),
         ("tot_prcp_mm","Precipitation (mm)"),("temp_range","Temp Range (°C)"),
         ("drought_proxy","Drought Proxy"),("heat_index","Heat Index")]
for ax, (col, label) in zip(axes.flat, pairs):
    data = weather_df[col].clip(weather_df[col].quantile(0.01),
                                weather_df[col].quantile(0.99))
    sns.histplot(data, kde=True, ax=ax, bins=50, color="#1A8FE3")
    ax.set_title(label); ax.set_xlabel("")
savefig("05_weather_distributions")

# Weather × fire risk
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
weather_risk = df[["avg_tmax_c","tot_prcp_mm","drought_proxy",
                    "Avg Fire Risk Score"]].dropna()
for ax, (wcol, wlabel) in zip(axes,
    [("avg_tmax_c","Max Temp (°C)"),
     ("tot_prcp_mm","Precipitation (mm)"),
     ("drought_proxy","Drought Proxy")]):
    xd = weather_risk[wcol].clip(weather_risk[wcol].quantile(0.01),
                                  weather_risk[wcol].quantile(0.99))
    yd = weather_risk["Avg Fire Risk Score"].clip(0, 4)
    ax.hexbin(xd, yd, gridsize=40, cmap="YlOrRd", mincnt=1)
    r, p = spearmanr(xd, yd)
    ax.set_xlabel(wlabel); ax.set_ylabel("Fire Risk Score")
    ax.set_title(f"ρ={r:.3f}, p={p:.3e}")
savefig("05b_weather_vs_risk")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SOCIOECONOMIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Socioeconomic analysis …")
socio_df = df[["median_income","housing_value","poverty_rate","edu_rate",
               "vacancy_rate","owner_rate","housing_cost_burden",
               "Avg Fire Risk Score","loss_ratio","Year"]].dropna()

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
socio_pairs = [("median_income","Median Income ($)"),
               ("housing_value","Housing Value ($)"),
               ("poverty_rate","Poverty Rate"),
               ("edu_rate","Bachelor+ Rate"),
               ("vacancy_rate","Vacancy Rate"),
               ("housing_cost_burden","Housing Cost Burden")]
for ax, (col, label) in zip(axes.flat, socio_pairs):
    data = socio_df[col].clip(socio_df[col].quantile(0.01),
                               socio_df[col].quantile(0.99))
    sns.histplot(data, kde=True, ax=ax, bins=50, color="#8E44AD")
    ax.set_title(label)
savefig("06_socioeconomic_distributions")

# Socioeconomic vs fire risk
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
for ax, (col, label) in zip(axes.flat, socio_pairs):
    xd = socio_df[col].clip(socio_df[col].quantile(0.01), socio_df[col].quantile(0.99))
    yd = socio_df["Avg Fire Risk Score"].clip(0, 4)
    ax.scatter(xd, yd, alpha=0.1, s=8, color="#8E44AD")
    m, b = np.polyfit(xd, yd, 1)
    xs = np.linspace(xd.min(), xd.max(), 200)
    ax.plot(xs, m*xs + b, "r-", linewidth=2)
    r, p = spearmanr(xd, yd)
    ax.set_xlabel(label); ax.set_ylabel("Fire Risk Score")
    ax.set_title(f"ρ={r:.3f}, p={p:.3e}")
savefig("06b_socioeconomic_vs_risk")

# ══════════════════════════════════════════════════════════════════════════════
# 7. TEMPORAL TRENDS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Temporal trend analysis …")
yearly = df.groupby("Year").agg(
    total_losses_sum      = ("total_losses",       "sum"),
    total_cat_losses_sum  = ("total_cat_losses",   "sum"),
    earned_premium_sum    = ("Earned Premium",      "sum"),
    total_claims_sum      = ("total_claims",        "sum"),
    avg_fire_risk         = ("Avg Fire Risk Score", "mean"),
    avg_drought           = ("drought_proxy",       "mean"),
    avg_tmax              = ("avg_tmax_c",          "mean"),
    avg_prcp              = ("tot_prcp_mm",         "mean"),
    fire_count            = ("fire_occurred",       "sum"),
    median_loss_ratio     = ("loss_ratio",          "median"),
).reset_index()
yearly["agg_loss_ratio"] = (yearly["total_losses_sum"]
                             / yearly["earned_premium_sum"].replace(0, np.nan))

fig, axes = plt.subplots(3, 2, figsize=(15, 14))
ax_configs = [
    ("total_losses_sum",     "Total Losses ($)",       "#E74C3C"),
    ("earned_premium_sum",   "Earned Premium ($)",     "#27AE60"),
    ("agg_loss_ratio",       "Aggregate Loss Ratio",   "#8E44AD"),
    ("fire_count",           "# Fire-linked Records",  "#E67E22"),
    ("avg_tmax",             "Avg Max Temp (°C)",      "#1A8FE3"),
    ("avg_drought",          "Avg Drought Proxy",      "#D35400"),
]
for ax, (col, label, color) in zip(axes.flat, ax_configs):
    ax.plot(yearly["Year"], yearly[col], "o-", color=color, linewidth=2.5,
            markersize=8)
    ax.set_title(label); ax.set_xlabel("Year"); ax.set_xticks(yearly["Year"])
savefig("07_temporal_trends")

# ══════════════════════════════════════════════════════════════════════════════
# 8. LOSS RATIO DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] Loss ratio analysis …")
lr_df = df[df["Earned Premium"] > 0][["loss_ratio","cat_share","claim_frequency",
                                       "Avg Fire Risk Score","Year","ZIP"]].dropna()
lr_df = lr_df[lr_df["loss_ratio"] < lr_df["loss_ratio"].quantile(0.99)]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

sns.boxplot(data=lr_df, x="Year", y="loss_ratio", ax=axes[0],
            palette="Set2", showfliers=False)
axes[0].set_title("Loss Ratio Distribution by Year")

sns.scatterplot(data=lr_df.sample(min(3000, len(lr_df))),
                x="Avg Fire Risk Score", y="loss_ratio",
                hue="Year", palette="Set1", alpha=0.6, s=25, ax=axes[1])
axes[1].set_title("Loss Ratio vs Fire Risk Score")

axes[2].hexbin(lr_df["claim_frequency"].clip(0, lr_df["claim_frequency"].quantile(0.98)),
               lr_df["loss_ratio"].clip(0, lr_df["loss_ratio"].quantile(0.98)),
               gridsize=40, cmap="Blues", mincnt=1)
axes[2].set_xlabel("Claim Frequency"); axes[2].set_ylabel("Loss Ratio")
axes[2].set_title("Claim Frequency vs Loss Ratio")
savefig("08_loss_ratio_analysis")

# ══════════════════════════════════════════════════════════════════════════════
# 9. FIRE HISTORY: ACREAGE & CAUSE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9] Fire history analysis …")
fire_df = df[df["GIS_ACRES"].notna()].copy()

cause_map = {1:"Lightning", 2:"Equipment Use", 4:"Campfire", 5:"Debris",
             7:"Arson", 8:"Playing w/ Fire", 9:"Miscellaneous",
             10:"Vehicle", 11:"Power Line", 14:"Unknown"}
fire_df["cause_label"] = fire_df["CAUSE"].map(cause_map).fillna("Other")
agency_color = {"CDF":"#E74C3C","CCO":"#3498DB","USF":"#2ECC71",
                "LRA":"#F39C12","BLM":"#9B59B6","Other":"#95A5A6"}
fire_df["agency_grp"] = fire_df["AGENCY"].where(
    fire_df["AGENCY"].isin(agency_color), "Other")

fig = plt.figure(figsize=(17, 13))
gs  = gridspec.GridSpec(2, 3, figure=fig)

# 9a. Acreage distribution
ax0 = fig.add_subplot(gs[0, 0])
sns.histplot(fire_df["log_gis_acres"].dropna(), bins=60, kde=True,
             ax=ax0, color="#E67E22")
ax0.set_title("Log(1+Acres) Distribution"); ax0.set_xlabel("log(1+GIS_ACRES)")

# 9b. Acreage by year
ax1 = fig.add_subplot(gs[0, 1])
sns.boxplot(data=fire_df, x="Year", y="log_gis_acres", ax=ax1,
            palette="Oranges", showfliers=False)
ax1.set_title("Log Acreage by Year")

# 9c. Top causes
ax2 = fig.add_subplot(gs[0, 2])
cause_counts = fire_df["cause_label"].value_counts().head(10)
sns.barplot(x=cause_counts.values, y=cause_counts.index, ax=ax2,
            palette="Reds_r")
ax2.set_title("Top Fire Causes (by count)")
ax2.set_xlabel("Count")

# 9d. Acreage by cause
ax3 = fig.add_subplot(gs[1, :2])
top_causes = cause_counts.index[:7]
sns.boxplot(data=fire_df[fire_df["cause_label"].isin(top_causes)],
            x="cause_label", y="log_gis_acres", ax=ax3,
            palette="Set2", showfliers=False)
ax3.set_title("Log Acreage by Cause"); ax3.set_xlabel("")
ax3.tick_params(axis="x", rotation=30)

# 9e. Agency breakdown
ax4 = fig.add_subplot(gs[1, 2])
agency_counts = fire_df["AGENCY"].value_counts().head(8)
ax4.pie(agency_counts.values, labels=agency_counts.index,
        autopct="%1.1f%%", startangle=140,
        colors=sns.color_palette("Set3", len(agency_counts)))
ax4.set_title("Fires by Agency")
savefig("09_fire_history_analysis")

# ══════════════════════════════════════════════════════════════════════════════
# 10. GEOGRAPHIC ANALYSIS (LAT / LON)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[10] Geographic analysis …")
geo_df = df[df["latitude"].notna() & df["longitude"].notna()].copy()
geo_df = geo_df[(geo_df["latitude"].between(32, 42))
               & (geo_df["longitude"].between(-125, -114))]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sc0 = axes[0].scatter(geo_df["longitude"], geo_df["latitude"],
                       c=geo_df["log_gis_acres"].clip(0, 15),
                       s=8, alpha=0.5, cmap="hot")
plt.colorbar(sc0, ax=axes[0], label="log(1+Acres)")
axes[0].set_title("Fire Locations (log Acreage)"); axes[0].set_xlabel("Lon")
axes[0].set_ylabel("Lat")

sc1 = axes[1].scatter(
    df.dropna(subset=["ZIP"])["ZIP"] % 1000,  # rough proxy sort
    df.dropna(subset=["ZIP"])["Avg Fire Risk Score"].clip(0,4),
    c=df.dropna(subset=["ZIP"])["Year"],
    s=8, alpha=0.3, cmap="viridis")
plt.colorbar(sc1, ax=axes[1], label="Year")
axes[1].set_title("Fire Risk Score by ZIP (×Year)")
axes[1].set_xlabel("ZIP mod 1000"); axes[1].set_ylabel("Fire Risk Score")

axes[2].hexbin(geo_df["longitude"], geo_df["latitude"],
               C=geo_df["log_gis_acres"].clip(0, 15),
               gridsize=50, cmap="YlOrRd", mincnt=1, reduce_C_function=np.mean)
axes[2].set_title("Avg log Acreage Heatmap"); axes[2].set_xlabel("Lon")
axes[2].set_ylabel("Lat")
savefig("10_geographic_analysis")

# ══════════════════════════════════════════════════════════════════════════════
# 11. CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
print("\n[11] Correlation heatmap …")
corr_feats = ["Avg Fire Risk Score","loss_ratio","cat_share","claim_frequency",
              "log_earned_premium","log_total_losses",
              "high_risk_pct","risk_concentration",
              "temp_range","drought_proxy","avg_tmax_c","log_prcp",
              "poverty_rate","edu_rate","vacancy_rate","owner_rate",
              "log_median_income","log_housing_value","housing_cost_burden",
              "log_gis_acres"]
corr_df = df[corr_feats].dropna()
corr_mat = corr_df.corr(method="spearman")

fig, ax = plt.subplots(figsize=(16, 14))
mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
sns.heatmap(corr_mat, mask=~mask,  # show lower triangle
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 7},
            linewidths=0.4, ax=ax, square=True)
ax.set_title("Spearman Correlation – Engineered Features", fontsize=14)
savefig("11_correlation_heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# 12. FEATURE IMPORTANCE VIA CORRELATION WITH LOSS RATIO
# ══════════════════════════════════════════════════════════════════════════════
print("\n[12] Feature ranking vs loss ratio …")
target   = "loss_ratio"
feat_pool = [c for c in corr_feats if c != target]
records  = []
for c in feat_pool:
    pair = df[[c, target]].dropna()
    pair = pair[pair[target] < pair[target].quantile(0.99)]
    if len(pair) < 50:
        continue
    r, p = spearmanr(pair[c], pair[target])
    records.append({"feature": c, "rho": r, "p_value": p})

imp_df = pd.DataFrame(records).sort_values("rho", key=abs, ascending=False)
imp_df["significant"] = imp_df["p_value"] < 0.05
colors = ["#E74C3C" if s else "#95A5A6" for s in imp_df["significant"]]

fig, ax = plt.subplots(figsize=(12, 9))
bars = ax.barh(imp_df["feature"][::-1], imp_df["rho"][::-1], color=colors[::-1])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Spearman ρ with Loss Ratio")
ax.set_title("Feature–Loss Ratio Correlation (red = p<0.05)")
savefig("12_feature_importance")

# ══════════════════════════════════════════════════════════════════════════════
# 13. PCA OF ENGINEERED FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[13] PCA …")
pca_cols = ["Avg Fire Risk Score","high_risk_pct","risk_concentration",
            "temp_range","drought_proxy","avg_tmax_c","log_prcp",
            "poverty_rate","edu_rate","vacancy_rate","owner_rate",
            "log_median_income","log_housing_value","housing_cost_burden"]
pca_df = df[pca_cols + ["Year"]].dropna()
X_scaled = StandardScaler().fit_transform(pca_df[pca_cols])

pca  = PCA(n_components=min(10, len(pca_cols)))
Xp   = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Scree
cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
axes[0].bar(range(1, len(cum_var)+1), pca.explained_variance_ratio_*100,
            color="#4878CF", label="Individual")
axes[0].step(range(1, len(cum_var)+1), cum_var, where="mid",
             color="#E74C3C", label="Cumulative")
axes[0].axhline(80, linestyle="--", color="gray", alpha=0.6, label="80%")
axes[0].set_xlabel("Component"); axes[0].set_ylabel("Variance (%)")
axes[0].set_title("PCA Scree Plot"); axes[0].legend()

# PC1 vs PC2
sc = axes[1].scatter(Xp[:, 0], Xp[:, 1],
                      c=pca_df["Year"], cmap="viridis", s=8, alpha=0.3)
plt.colorbar(sc, ax=axes[1], label="Year")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
axes[1].set_title("PC1 vs PC2 (coloured by Year)")

# Loadings
loadings = pd.DataFrame(pca.components_[:2].T,
                         columns=["PC1","PC2"], index=pca_cols)
loadings["magnitude"] = np.sqrt(loadings["PC1"]**2 + loadings["PC2"]**2)
loadings = loadings.sort_values("magnitude", ascending=False)
x = np.arange(len(loadings))
axes[2].bar(x - 0.2, loadings["PC1"], 0.4, label="PC1", color="#4878CF")
axes[2].bar(x + 0.2, loadings["PC2"], 0.4, label="PC2", color="#E74C3C")
axes[2].set_xticks(x); axes[2].set_xticklabels(loadings.index, rotation=45, ha="right")
axes[2].set_title("PCA Loadings (PC1 & PC2)"); axes[2].legend()
savefig("13_pca_analysis")

# ══════════════════════════════════════════════════════════════════════════════
# 14. K-MEANS CLUSTERING ON RISK PROFILE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[14] K-Means clustering …")
cluster_cols = ["Avg Fire Risk Score","drought_proxy","poverty_rate",
                "log_median_income","log_housing_value","high_risk_pct"]
clust_df = df[cluster_cols].dropna()
Xc = StandardScaler().fit_transform(clust_df)

inertias = []
K_range  = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(Xc)
    inertias.append(km.inertia_)

best_k = 4
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels   = km_final.fit_predict(Xc)
clust_df = clust_df.copy()
clust_df["cluster"] = labels

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].plot(list(K_range), inertias, "o-", color="#4878CF", linewidth=2)
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Curve – K-Means")

pca2 = PCA(n_components=2, random_state=42)
Xp2  = pca2.fit_transform(Xc)
palette = ["#E74C3C","#3498DB","#27AE60","#F39C12"]
for cl in range(best_k):
    mask = labels == cl
    axes[1].scatter(Xp2[mask, 0], Xp2[mask, 1],
                    s=12, alpha=0.4, label=f"Cluster {cl}", color=palette[cl])
axes[1].set_title(f"Clusters in PCA Space (k={best_k})")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2"); axes[1].legend()

cluster_means = clust_df.groupby("cluster")[cluster_cols].mean()
sns.heatmap(cluster_means.T, annot=True, fmt=".2f", cmap="RdYlGn_r",
            ax=axes[2], linewidths=0.5)
axes[2].set_title("Cluster Mean Profile")
savefig("14_kmeans_clustering")

# ══════════════════════════════════════════════════════════════════════════════
# 15. INSURANCE PREMIUM VS RISK SCORE STRATIFIED ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[15] Premium vs risk analysis …")
pr_df = df[["log_earned_premium","Avg Fire Risk Score","Year",
            "log_total_losses","loss_ratio"]].dropna()
pr_df = pr_df[pr_df["log_earned_premium"] > 0]
pr_df["risk_bin"] = pd.cut(pr_df["Avg Fire Risk Score"].clip(-0.2, 4),
                            bins=6, labels=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.scatterplot(data=pr_df.sample(min(4000, len(pr_df))),
                x="Avg Fire Risk Score", y="log_earned_premium",
                hue="Year", palette="viridis", s=20, alpha=0.6, ax=axes[0])
axes[0].set_title("Log Premium vs Fire Risk Score")

pr_grp = pr_df.groupby(["risk_bin","Year"])["log_earned_premium"].mean().unstack()
pr_grp.plot(ax=axes[1], marker="o")
axes[1].set_title("Avg Log Premium by Risk Bin & Year")
axes[1].set_xlabel("Risk Bin (0=low, 5=high)")

sns.boxplot(data=pr_df, x="Year", y="log_earned_premium",
            hue="risk_bin", ax=axes[2], palette="RdYlGn_r",
            showfliers=False, legend=False)
axes[2].set_title("Log Premium Distribution by Year & Risk")
savefig("15_premium_risk_analysis")

# ══════════════════════════════════════════════════════════════════════════════
# 16. STATISTICAL TESTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[16] Statistical tests summary …")

test_results = []

# Kruskal-Wallis: does loss_ratio differ across years?
yr_groups = [df[df["Year"]==y]["loss_ratio"].dropna().values for y in sorted(df["Year"].unique())]
yr_groups = [g[g < np.quantile(g, 0.99)] for g in yr_groups if len(g) > 10]
stat, pv = stats.kruskal(*yr_groups)
test_results.append({"test":"Kruskal-Wallis","H0":"loss_ratio same across years",
                     "statistic":f"{stat:.3f}","p_value":f"{pv:.3e}",
                     "reject":pv < 0.05})

# Mann-Whitney: dry_hot vs not-dry_hot on loss_ratio
g0 = df[df["dry_hot"]==0]["loss_ratio"].dropna().clip(upper=df["loss_ratio"].quantile(0.99))
g1 = df[df["dry_hot"]==1]["loss_ratio"].dropna().clip(upper=df["loss_ratio"].quantile(0.99))
stat, pv = stats.mannwhitneyu(g0, g1, alternative="two-sided")
test_results.append({"test":"Mann-Whitney","H0":"loss_ratio same for dry_hot vs not",
                     "statistic":f"{stat:.3f}","p_value":f"{pv:.3e}",
                     "reject":pv < 0.05})

# Shapiro-Wilk on log_total_losses (sample)
sample = df["log_total_losses"].dropna().sample(min(500, df["log_total_losses"].notna().sum()),
                                                  random_state=42)
stat, pv = stats.shapiro(sample)
test_results.append({"test":"Shapiro-Wilk","H0":"log_total_losses is normal",
                     "statistic":f"{stat:.4f}","p_value":f"{pv:.3e}",
                     "reject":pv < 0.05})

# Pearson correlation: drought_proxy vs log_total_losses
pair = df[["drought_proxy","log_total_losses"]].dropna()
r, pv = stats.pearsonr(pair["drought_proxy"], pair["log_total_losses"])
test_results.append({"test":"Pearson corr","H0":"drought_proxy & log_losses uncorrelated",
                     "statistic":f"r={r:.4f}","p_value":f"{pv:.3e}",
                     "reject":pv < 0.05})

# Chi-square: fire category vs dry_hot condition
ct = pd.crosstab(df["Category_HO"].astype(str), df["dry_hot"].astype(str))
stat, pv, dof, _ = stats.chi2_contingency(ct)
test_results.append({"test":"Chi-Square","H0":"policy category independent of dry_hot",
                     "statistic":f"{stat:.3f} (dof={dof})","p_value":f"{pv:.3e}",
                     "reject":pv < 0.05})

tests_df = pd.DataFrame(test_results)
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")
tbl = ax.table(cellText=tests_df.values, colLabels=tests_df.columns,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 2.0)
# colour reject column
for i, row in enumerate(tests_df.itertuples(), start=1):
    color = "#FADBD8" if row.reject else "#D5F5E3"
    tbl[(i, 4)].set_facecolor(color)
ax.set_title("Statistical Hypothesis Tests Summary", fontsize=13, pad=20)
savefig("16_statistical_tests")

# ══════════════════════════════════════════════════════════════════════════════
# 17. PAIR-PLOT OF TOP ENGINEERED FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[17] Pair plot …")
pp_cols = ["loss_ratio","Avg Fire Risk Score","drought_proxy",
           "log_median_income","poverty_rate","log_gis_acres"]
pp_df = df[pp_cols + ["Year"]].dropna()
pp_df = pp_df[pp_df["loss_ratio"] < pp_df["loss_ratio"].quantile(0.98)].sample(
    min(2000, len(pp_df)), random_state=42)

g = sns.pairplot(pp_df, vars=pp_cols, hue="Year", palette="viridis",
                 plot_kws={"alpha": 0.3, "s": 12},
                 diag_kws={"fill": True})
g.figure.suptitle("Pair Plot – Top Engineered Features", y=1.02, fontsize=13)
plt.savefig(OUT_PATH / "17_pair_plot.png", dpi=SAVE_DPI, bbox_inches="tight")
plt.close()
print("  [saved] 17_pair_plot.png")

# ══════════════════════════════════════════════════════════════════════════════
# 18. VULNERABILITY INDEX MAP
# ══════════════════════════════════════════════════════════════════════════════
print("\n[18] Vulnerability index …")
if "vulnerability_index" in df.columns:
    vi_df = df[["vulnerability_index","Avg Fire Risk Score",
                "loss_ratio","Year"]].dropna()
    vi_df = vi_df[vi_df["loss_ratio"] < vi_df["loss_ratio"].quantile(0.98)]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    sns.histplot(vi_df["vulnerability_index"], bins=60, kde=True,
                 ax=axes[0], color="#C0392B")
    axes[0].set_title("Vulnerability Index Distribution")

    axes[1].scatter(vi_df["vulnerability_index"],
                     vi_df["Avg Fire Risk Score"].clip(0, 4),
                     alpha=0.2, s=8, color="#C0392B")
    r, p = spearmanr(vi_df["vulnerability_index"], vi_df["Avg Fire Risk Score"].clip(0, 4))
    axes[1].set_xlabel("Vulnerability Index"); axes[1].set_ylabel("Fire Risk Score")
    axes[1].set_title(f"VI vs Fire Risk (ρ={r:.3f}, p={p:.2e})")

    axes[2].scatter(vi_df["vulnerability_index"],
                     vi_df["loss_ratio"],
                     alpha=0.2, s=8, color="#8B0000")
    r2, p2 = spearmanr(vi_df["vulnerability_index"], vi_df["loss_ratio"])
    axes[2].set_xlabel("Vulnerability Index"); axes[2].set_ylabel("Loss Ratio")
    axes[2].set_title(f"VI vs Loss Ratio (ρ={r2:.3f}, p={p2:.2e})")
    savefig("18_vulnerability_index")

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
saved = sorted(OUT_PATH.glob("*.png"))
print(f"\n✓ All done — {len(saved)} figures saved to {OUT_PATH}")
for p in saved:
    print(f"  {p.name}")
