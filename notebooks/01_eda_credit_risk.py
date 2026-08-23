"""
Credit Risk PD Model — Exploratory Data Analysis
=================================================
Dataset : Home Credit Default Risk (Kaggle)
Target  : TARGET = 1 → payment difficulties (default proxy), 0 → no difficulty

Sections
--------
 0. Setup & Data Loading
 1. Target Variable
 2. Missing Values
 3. Application — Numerical Features
 4. Application — Categorical Features
 5. Key Credit-Risk Variable Groups
    5a. Demographics
    5b. Financial Ratios
    5c. Employment
    5d. External Scores (EXT_SOURCE)
    5e. Social Circle & Document Flags
 6. Bureau Data Aggregation & EDA
 7. Previous Application EDA
 8. POS/Cash Balance & Installments Summary
 9. Credit Card Balance EDA
10. WoE / IV Analysis (scorecardpy)
11. Correlation & Multicollinearity
12. Summary — Top Predictors Shortlist

Run with:
    python notebooks/01_eda_credit_risk.py
Figures are saved to notebooks/figures/
"""

# =============================================================================
# 0. SETUP & DATA LOADING
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — saves to file instead of displaying
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency

# Optional WoE/IV — install with: pip install scorecardpy
try:
    import scorecardpy as sc
    SCORECARDPY_AVAILABLE = True
except ImportError:
    SCORECARDPY_AVAILABLE = False
    print("scorecardpy not found — WoE/IV section will be skipped. "
          "Install with: pip install scorecardpy")

# Paths
ROOT      = Path(__file__).resolve().parent.parent
DATA_RAW  = ROOT / "data" / "data_raw"
APP_TRAIN = DATA_RAW / "home-credit-default-risk" / "application_train.csv"
APP_TEST  = DATA_RAW / "home-credit-default-risk" / "application_test.csv"
BUREAU    = DATA_RAW / "bureau.csv"
BB        = DATA_RAW / "bureau_balance.csv"
PREV_APP  = DATA_RAW / "previous_application.csv"
POS_CASH  = DATA_RAW / "POS_CASH_balance.csv"
INSTALL   = DATA_RAW / "installments_payments.csv"
CC_BAL    = DATA_RAW / "credit_card_balance.csv"

FIG_DIR   = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = {"0 — No difficulty": "#2196F3", "1 — Difficulty": "#F44336"}
SEED = 42

print("=" * 70)
print("HOME CREDIT DEFAULT RISK — EDA")
print("=" * 70)

# Load main table
app = pd.read_csv(APP_TRAIN)
print(f"\napplication_train shape : {app.shape}")
print(f"Default rate            : {app['TARGET'].mean():.2%}  "
      f"({app['TARGET'].sum():,} defaults out of {len(app):,})")


# =============================================================================
# 1. TARGET VARIABLE
# =============================================================================
print("\n--- 1. Target Variable ---")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Count plot
counts = app["TARGET"].value_counts()
labels = ["0 — No difficulty", "1 — Difficulty"]
colors = ["#2196F3", "#F44336"]
axes[0].bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 500, f"{v:,}\n({v/len(app):.1%})",
                 ha="center", va="bottom", fontsize=11)
axes[0].set_title("Target Distribution", fontweight="bold")
axes[0].set_ylabel("Count")
axes[0].set_ylim(0, counts.max() * 1.15)

# Pie
axes[1].pie(counts.values, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1].set_title("Class Balance", fontweight="bold")

plt.tight_layout()
plt.savefig(FIG_DIR / "01_target_distribution.png", dpi=150)
plt.close()
print("  Class imbalance confirmed — ~8% default rate")
print("  → Use class_weight='balanced' or scale_pos_weight in XGBoost")


# =============================================================================
# 2. MISSING VALUES
# =============================================================================
print("\n--- 2. Missing Values ---")

miss = (app.isnull().sum() / len(app) * 100).sort_values(ascending=False)
miss = miss[miss > 0]
print(f"  Columns with missing values : {len(miss)} / {app.shape[1]}")
print(f"  Columns > 50% missing       : {(miss > 50).sum()}")
print(f"  Columns > 30% missing       : {(miss > 30).sum()}")

# Top 40 missing
top_miss = miss.head(40)
fig, ax = plt.subplots(figsize=(10, 12))
top_miss.sort_values().plot.barh(ax=ax, color="#607D8B")
ax.axvline(50, color="red", linestyle="--", linewidth=1.2, label="50% threshold")
ax.axvline(30, color="orange", linestyle="--", linewidth=1.2, label="30% threshold")
ax.set_xlabel("% Missing")
ax.set_title("Top 40 Columns by Missing Rate (application_train)", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "02_missing_values.png", dpi=150)
plt.close()

# Summary table
miss_df = pd.DataFrame({"missing_pct": miss,
                         "missing_n": app.isnull().sum()[miss.index]})
miss_df["action"] = pd.cut(miss_df["missing_pct"],
                            bins=[-1, 10, 30, 50, 100],
                            labels=["impute_ok", "impute_caution",
                                    "create_missing_flag", "consider_drop"])
print("\n  Missing value action summary:")
print(miss_df["action"].value_counts().to_string())


# =============================================================================
# 3. APPLICATION — NUMERICAL FEATURES
# =============================================================================
print("\n--- 3. Numerical Features ---")

num_cols = app.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ("SK_ID_CURR", "TARGET")]

# Descriptive stats
desc = app[num_cols].describe().T
desc["skewness"] = app[num_cols].skew()
desc["kurtosis"] = app[num_cols].kurtosis()
desc["missing_pct"] = app[num_cols].isnull().mean() * 100
print(f"\n  Numerical columns : {len(num_cols)}")
print("  Highest skewness (top 5):")
print(desc["skewness"].abs().sort_values(ascending=False).head().to_string())

# Point-biserial correlations with TARGET
pb_corr = {}
for c in num_cols:
    valid = app[[c, "TARGET"]].dropna()
    if len(valid) > 100:
        r, p = stats.pointbiserialr(valid["TARGET"], valid[c])
        pb_corr[c] = {"correlation": r, "p_value": p}

pb_df = pd.DataFrame(pb_corr).T.sort_values("correlation")
pb_df["abs_corr"] = pb_df["correlation"].abs()
top_num = pb_df.sort_values("abs_corr", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 8))
plotted_corr = top_num["correlation"].sort_values()
colors_bar = ["#F44336" if v < 0 else "#2196F3" for v in plotted_corr]
plotted_corr.plot.barh(ax=ax, color=colors_bar)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Top 20 Numerical Features — Point-Biserial Correlation with TARGET",
             fontweight="bold")
ax.set_xlabel("Correlation (positive = higher value → more defaults)")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_numerical_correlations.png", dpi=150)
plt.close()

print("\n  Top 10 numerical predictors (by |correlation|):")
print(top_num["correlation"].head(10).to_string())


# =============================================================================
# 4. APPLICATION — CATEGORICAL FEATURES
# =============================================================================
print("\n--- 4. Categorical Features ---")

cat_cols = app.select_dtypes(include=["object"]).columns.tolist()
print(f"  Categorical columns : {len(cat_cols)}")

# Default rate by category — Chi² test for association
chi2_results = {}
for c in cat_cols:
    ct = pd.crosstab(app[c], app["TARGET"])
    if ct.shape[0] > 1:
        chi2, p, dof, _ = chi2_contingency(ct)
        chi2_results[c] = {"chi2": chi2, "p_value": p, "dof": dof,
                           "cramers_v": np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))}

chi2_df = pd.DataFrame(chi2_results).T.sort_values("cramers_v", ascending=False)
print("\n  Top categorical predictors (Cramer's V):")
print(chi2_df[["cramers_v", "p_value"]].head(10).to_string())

fig, ax = plt.subplots(figsize=(9, 6))
chi2_df["cramers_v"].sort_values(ascending=True).plot.barh(ax=ax, color="#9C27B0")
ax.set_title("Categorical Features — Cramer's V with TARGET", fontweight="bold")
ax.set_xlabel("Cramer's V (0=no association, 1=perfect)")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_categorical_association.png", dpi=150)
plt.close()

# Default rate bar charts for top 5 categorical features
top_cats = chi2_df.head(5).index.tolist()
fig, axes = plt.subplots(1, len(top_cats), figsize=(20, 5))
for ax, col in zip(axes, top_cats):
    dr = app.groupby(col)["TARGET"].agg(["mean", "count"])
    dr = dr.sort_values("mean", ascending=False).head(15)
    ax.barh(dr.index.astype(str), dr["mean"] * 100, color="#E91E63", edgecolor="white")
    ax.axvline(app["TARGET"].mean() * 100, color="black",
               linestyle="--", linewidth=1, label="Portfolio avg")
    ax.set_xlabel("Default Rate (%)")
    ax.set_title(col, fontweight="bold", fontsize=9)
    ax.legend(fontsize=8)
plt.suptitle("Default Rate by Top Categorical Features", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_categorical_default_rates.png", dpi=150, bbox_inches="tight")
plt.close()


# =============================================================================
# 5a. DEMOGRAPHICS
# =============================================================================
print("\n--- 5a. Demographics ---")

# Age — convert DAYS_BIRTH to years (negative → positive)
app["AGE_YEARS"] = -app["DAYS_BIRTH"] / 365

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Age distribution by target
for tgt, label, color in [(0, "No difficulty", "#2196F3"),
                           (1, "Difficulty", "#F44336")]:
    axes[0].hist(app.loc[app["TARGET"] == tgt, "AGE_YEARS"],
                 bins=40, alpha=0.6, label=label, color=color, density=True)
axes[0].set_xlabel("Age (years)")
axes[0].set_ylabel("Density")
axes[0].set_title("Age Distribution by Default Status", fontweight="bold")
axes[0].legend()

# Default rate by age band
app["AGE_BAND"] = pd.cut(app["AGE_YEARS"],
                          bins=[19, 25, 30, 35, 40, 50, 60, 70],
                          labels=["20-25", "26-30", "31-35",
                                  "36-40", "41-50", "51-60", "61-70"])
dr_age = app.groupby("AGE_BAND", observed=True)["TARGET"].mean() * 100
dr_age.plot.bar(ax=axes[1], color="#FF7043", edgecolor="white")
axes[1].axhline(app["TARGET"].mean() * 100, color="black",
                linestyle="--", linewidth=1.2, label="Portfolio avg")
axes[1].set_xlabel("Age Band")
axes[1].set_ylabel("Default Rate (%)")
axes[1].set_title("Default Rate by Age Band", fontweight="bold")
axes[1].legend()
axes[1].tick_params(axis="x", rotation=30)

# Gender
dr_gender = app.groupby("CODE_GENDER")["TARGET"].agg(["mean", "count"])
axes[2].bar(dr_gender.index, dr_gender["mean"] * 100,
            color=["#42A5F5", "#EF5350", "#78909C"], edgecolor="white")
axes[2].axhline(app["TARGET"].mean() * 100, color="black",
                linestyle="--", linewidth=1.2)
axes[2].set_ylabel("Default Rate (%)")
axes[2].set_title("Default Rate by Gender", fontweight="bold")
for i, (idx, row) in enumerate(dr_gender.iterrows()):
    axes[2].text(i, row["mean"] * 100 + 0.1,
                 f"n={row['count']:,}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / "06_demographics.png", dpi=150)
plt.close()

# Education & Family status
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, col, title in [
    (axes[0], "NAME_EDUCATION_TYPE", "Default Rate by Education"),
    (axes[1], "NAME_FAMILY_STATUS",  "Default Rate by Family Status")
]:
    dr = app.groupby(col)["TARGET"].mean().sort_values(ascending=False) * 100
    dr.plot.bar(ax=ax, color="#5C6BC0", edgecolor="white")
    ax.axhline(app["TARGET"].mean() * 100, color="red",
               linestyle="--", linewidth=1.2, label="Portfolio avg")
    ax.set_ylabel("Default Rate (%)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(FIG_DIR / "07_education_family.png", dpi=150)
plt.close()


# =============================================================================
# 5b. FINANCIAL RATIOS
# =============================================================================
print("\n--- 5b. Financial Ratios ---")

# DAYS_EMPLOYED anomaly — 365243 is a sentinel for unemployed/pensioners
print(f"  DAYS_EMPLOYED == 365243 : {(app['DAYS_EMPLOYED'] == 365243).sum():,} rows")
app["DAYS_EMPLOYED_CLEAN"] = app["DAYS_EMPLOYED"].replace(365243, np.nan)
app["FLAG_EMPLOYED_ANOMALY"] = (app["DAYS_EMPLOYED"] == 365243).astype(int)

# Derived financial ratios
app["CREDIT_INCOME_RATIO"]  = app["AMT_CREDIT"]  / app["AMT_INCOME_TOTAL"]
app["ANNUITY_INCOME_RATIO"] = app["AMT_ANNUITY"]  / app["AMT_INCOME_TOTAL"]
app["CREDIT_TERM"]          = app["AMT_CREDIT"]   / app["AMT_ANNUITY"]
app["GOODS_CREDIT_RATIO"]   = app["AMT_GOODS_PRICE"] / app["AMT_CREDIT"]
app["INCOME_PER_PERSON"]    = (app["AMT_INCOME_TOTAL"]
                                / app["CNT_FAM_MEMBERS"].replace(0, np.nan))

ratio_cols = ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO",
              "CREDIT_TERM", "GOODS_CREDIT_RATIO", "INCOME_PER_PERSON"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ax, col in zip(axes, ratio_cols):
    g0 = app.loc[(app["TARGET"] == 0) & app[col].notna(), col]
    g1 = app.loc[(app["TARGET"] == 1) & app[col].notna(), col]
    # Clip to 99th percentile for readability
    cap = app[col].quantile(0.99)
    g0_c = g0.clip(upper=cap)
    g1_c = g1.clip(upper=cap)
    ax.hist(g0_c, bins=50, alpha=0.6, color="#2196F3",
            label="No difficulty", density=True)
    ax.hist(g1_c, bins=50, alpha=0.6, color="#F44336",
            label="Difficulty", density=True)
    ks, p = ks_2samp(g0_c, g1_c)
    ax.set_title(f"{col}\nKS={ks:.3f}, p={p:.2e}", fontweight="bold", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlabel(col)
    ax.set_ylabel("Density")

# Default rate by number of children
axes[5].cla()
dr_children = (app[app["CNT_CHILDREN"] <= 5]
               .groupby("CNT_CHILDREN")["TARGET"].mean() * 100)
dr_children.plot.bar(ax=axes[5], color="#26A69A", edgecolor="white")
axes[5].axhline(app["TARGET"].mean() * 100, color="red",
                linestyle="--", linewidth=1.2, label="Avg")
axes[5].set_title("Default Rate by No. Children", fontweight="bold")
axes[5].set_ylabel("Default Rate (%)")
axes[5].legend()

plt.suptitle("Financial Ratios — Distribution by Default Status (KS test)",
             fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "08_financial_ratios.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n  Key financial ratio default rate comparison:")
for col in ratio_cols:
    med0 = app.loc[app["TARGET"] == 0, col].median()
    med1 = app.loc[app["TARGET"] == 1, col].median()
    print(f"  {col:<28} median(0)={med0:.2f}  median(1)={med1:.2f}")


# =============================================================================
# 5c. EMPLOYMENT
# =============================================================================
print("\n--- 5c. Employment ---")

app["YEARS_EMPLOYED"] = (-app["DAYS_EMPLOYED_CLEAN"] / 365).clip(lower=0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Employment years distribution
for tgt, lbl, col in [(0, "No difficulty", "#2196F3"),
                       (1, "Difficulty",    "#F44336")]:
    axes[0].hist(app.loc[(app["TARGET"] == tgt) & app["YEARS_EMPLOYED"].notna(),
                          "YEARS_EMPLOYED"].clip(upper=40),
                 bins=40, alpha=0.6, label=lbl, color=col, density=True)
axes[0].set_xlabel("Years Employed (capped 40)")
axes[0].set_ylabel("Density")
axes[0].set_title("Employment Tenure by Default", fontweight="bold")
axes[0].legend()

# Default rate by income type
dr_inc = (app.groupby("NAME_INCOME_TYPE")["TARGET"].mean()
            .sort_values(ascending=False) * 100)
dr_inc.plot.bar(ax=axes[1], color="#AB47BC", edgecolor="white")
axes[1].axhline(app["TARGET"].mean() * 100, color="red",
                linestyle="--", linewidth=1.2, label="Portfolio avg")
axes[1].set_title("Default Rate by Income Type", fontweight="bold")
axes[1].set_ylabel("Default Rate (%)")
axes[1].legend()
axes[1].tick_params(axis="x", rotation=30)

# Employed anomaly flag
dr_flag = app.groupby("FLAG_EMPLOYED_ANOMALY")["TARGET"].mean() * 100
axes[2].bar(["Normal employment", "Anomalous flag\n(pensioner/unemployed)"],
            dr_flag.values, color=["#42A5F5", "#EF5350"], edgecolor="white")
axes[2].axhline(app["TARGET"].mean() * 100, color="black",
                linestyle="--", linewidth=1.2)
axes[2].set_ylabel("Default Rate (%)")
axes[2].set_title("Effect of DAYS_EMPLOYED Anomaly Flag", fontweight="bold")

plt.tight_layout()
plt.savefig(FIG_DIR / "09_employment.png", dpi=150)
plt.close()

# Top 15 organization types
fig, ax = plt.subplots(figsize=(10, 8))
dr_org = (app.groupby("ORGANIZATION_TYPE")["TARGET"].mean()
            .sort_values(ascending=False) * 100)
dr_org.head(20).sort_values().plot.barh(ax=ax, color="#FF7043")
ax.axvline(app["TARGET"].mean() * 100, color="black",
           linestyle="--", linewidth=1.2, label="Portfolio avg")
ax.set_title("Default Rate by Organization Type (top 20)", fontweight="bold")
ax.set_xlabel("Default Rate (%)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "10_organization_type.png", dpi=150)
plt.close()


# =============================================================================
# 5d. EXTERNAL SCORES (EXT_SOURCE_1/2/3) — usually strongest predictors
# =============================================================================
print("\n--- 5d. External Scores (EXT_SOURCE) ---")

ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

for col in ext_cols:
    valid = app[[col, "TARGET"]].dropna()
    r, p = stats.pointbiserialr(valid["TARGET"], valid[col])
    ks, _ = ks_2samp(valid.loc[valid["TARGET"] == 0, col],
                     valid.loc[valid["TARGET"] == 1, col])
    miss_rate = app[col].isna().mean()
    print(f"  {col}: corr={r:.4f}, KS={ks:.4f}, missing={miss_rate:.1%}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, col in zip(axes, ext_cols):
    g0 = app.loc[(app["TARGET"] == 0) & app[col].notna(), col]
    g1 = app.loc[(app["TARGET"] == 1) & app[col].notna(), col]
    ax.hist(g0, bins=50, alpha=0.6, color="#2196F3",
            label="No difficulty", density=True)
    ax.hist(g1, bins=50, alpha=0.6, color="#F44336",
            label="Difficulty", density=True)
    ks, _ = ks_2samp(g0, g1)
    ax.set_title(f"{col}  (KS={ks:.3f})", fontweight="bold")
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.legend()

plt.suptitle("External Credit Scores — strongest individual predictors",
             fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "11_ext_sources.png", dpi=150, bbox_inches="tight")
plt.close()

# Pairplot of EXT_SOURCE features coloured by TARGET
sample = app[ext_cols + ["TARGET"]].dropna().sample(5000, random_state=SEED)
sample["TARGET_label"] = sample["TARGET"].map({0: "No difficulty", 1: "Difficulty"})
g = sns.pairplot(sample, vars=ext_cols, hue="TARGET_label",
                 palette={"No difficulty": "#2196F3", "Difficulty": "#F44336"},
                 plot_kws=dict(alpha=0.3, s=8), diag_kind="kde")
g.figure.suptitle("EXT_SOURCE Pairplot (n=5,000 sample)", y=1.02, fontweight="bold")
g.figure.savefig(FIG_DIR / "12_ext_source_pairplot.png", dpi=120,
                 bbox_inches="tight")
plt.close()

# Combined EXT score
app["EXT_MEAN"] = app[ext_cols].mean(axis=1)
app["EXT_MIN"]  = app[ext_cols].min(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, title in [(axes[0], "EXT_MEAN", "Mean EXT Score"),
                        (axes[1], "EXT_MIN",  "Min EXT Score")]:
    g0 = app.loc[(app["TARGET"] == 0) & app[col].notna(), col]
    g1 = app.loc[(app["TARGET"] == 1) & app[col].notna(), col]
    ks, _ = ks_2samp(g0, g1)
    ax.hist(g0, bins=60, alpha=0.6, color="#2196F3",
            label="No difficulty", density=True)
    ax.hist(g1, bins=60, alpha=0.6, color="#F44336",
            label="Difficulty", density=True)
    ax.set_title(f"{title}  (KS={ks:.3f})", fontweight="bold")
    ax.legend()

plt.suptitle("Aggregated External Scores", fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "13_ext_combined.png", dpi=150)
plt.close()


# =============================================================================
# 5e. SOCIAL CIRCLE & DOCUMENT FLAGS
# =============================================================================
print("\n--- 5e. Social Circle & Document Flags ---")

social_cols = ["OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
               "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"]

doc_flag_cols = [c for c in app.columns if c.startswith("FLAG_DOCUMENT")]

# Default rate for social circle defaults
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col in [(axes[0], "DEF_30_CNT_SOCIAL_CIRCLE"),
                (axes[1], "DEF_60_CNT_SOCIAL_CIRCLE")]:
    dr = (app[app[col] <= 5]
          .groupby(col)["TARGET"].mean() * 100)
    dr.plot.bar(ax=ax, color="#26C6DA", edgecolor="white")
    ax.axhline(app["TARGET"].mean() * 100, color="red",
               linestyle="--", linewidth=1.2, label="Portfolio avg")
    ax.set_title(f"Default Rate by {col}", fontweight="bold", fontsize=9)
    ax.set_ylabel("Default Rate (%)")
    ax.legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "14_social_circle.png", dpi=150)
plt.close()

# Document flags default rate heatmap
doc_dr = {}
for c in doc_flag_cols:
    doc_dr[c] = {
        "default_rate_when_1": app.loc[app[c] == 1, "TARGET"].mean(),
        "default_rate_when_0": app.loc[app[c] == 0, "TARGET"].mean(),
        "lift": (app.loc[app[c] == 1, "TARGET"].mean()
                 / app["TARGET"].mean())
    }
doc_df = pd.DataFrame(doc_dr).T.sort_values("default_rate_when_1", ascending=False)

fig, ax = plt.subplots(figsize=(10, 7))
x = range(len(doc_df))
ax.bar(x, doc_df["default_rate_when_1"] * 100, label="Flag=1", color="#EF5350", alpha=0.8)
ax.bar(x, doc_df["default_rate_when_0"] * 100, label="Flag=0", color="#42A5F5", alpha=0.5)
ax.axhline(app["TARGET"].mean() * 100, color="black",
           linestyle="--", linewidth=1.2, label="Portfolio avg")
ax.set_xticks(list(x))
ax.set_xticklabels(doc_df.index, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Default Rate (%)")
ax.set_title("Document Flags — Default Rate when Flag = 1 vs 0", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "15_document_flags.png", dpi=150)
plt.close()

print(f"  Most impactful document flag: {doc_df['lift'].abs().idxmax()} "
      f"(lift={doc_df['lift'].abs().max():.2f})")


# =============================================================================
# 6. BUREAU DATA AGGREGATION & EDA
# =============================================================================
print("\n--- 6. Bureau Data ---")

bureau = pd.read_csv(BUREAU)
bb     = pd.read_csv(BB)
print(f"  bureau shape       : {bureau.shape}")
print(f"  bureau_balance shape: {bb.shape}")

# Aggregate bureau_balance to bureau level
# Worst ever DPD status per credit
# "X" (status unknown that month) deliberately left unmapped (-> NaN), not
# folded into "C" (confirmed closed, clean) -- see src/features.py's
# BUREAU_BALANCE_STATUS_RANK for the same fix applied to the modelling pipeline.
bb_status_map = {"C": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
bb["STATUS_NUM"] = bb["STATUS"].map(bb_status_map)
bb_agg = bb.groupby("SK_ID_BUREAU").agg(
    max_dpd_status=("STATUS_NUM", "max"),
    n_months      =("MONTHS_BALANCE", "count"),
    n_dpd_months  =("STATUS_NUM", lambda x: (x > 0).sum())
).reset_index()

bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

# Aggregate bureau to application level
bureau_app = bureau.groupby("SK_ID_CURR").agg(
    bureau_n_credits      =("SK_ID_BUREAU", "count"),
    bureau_n_active       =("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
    bureau_n_closed       =("CREDIT_ACTIVE", lambda x: (x == "Closed").sum()),
    bureau_max_overdue    =("AMT_CREDIT_MAX_OVERDUE", "max"),
    bureau_sum_debt       =("AMT_CREDIT_SUM_DEBT", "sum"),
    bureau_sum_credit     =("AMT_CREDIT_SUM", "sum"),
    bureau_avg_days_credit=("DAYS_CREDIT", "mean"),
    bureau_max_dpd        =("max_dpd_status", "max"),
    bureau_n_dpd_credits  =("n_dpd_months", lambda x: (x > 0).sum()),
    bureau_overdue_current=("AMT_CREDIT_SUM_OVERDUE", "sum"),
).reset_index()

bureau_app["bureau_dpd_rate"]  = (bureau_app["bureau_n_dpd_credits"]
                                    / bureau_app["bureau_n_credits"])
bureau_app["bureau_utilization"] = (bureau_app["bureau_sum_debt"]
                                      / bureau_app["bureau_sum_credit"]
                                        .replace(0, np.nan))

# Merge with app for analysis
app_b = app[["SK_ID_CURR", "TARGET"]].merge(bureau_app, on="SK_ID_CURR", how="left")

bureau_num_cols = [c for c in bureau_app.columns if c != "SK_ID_CURR"]
print(f"\n  Bureau features engineered : {len(bureau_num_cols)}")

# KS for each bureau feature
print("  Bureau feature KS statistics vs TARGET:")
bureau_ks = {}
for col in bureau_num_cols:
    valid = app_b[[col, "TARGET"]].dropna()
    if len(valid) > 100:
        ks, _ = ks_2samp(valid.loc[valid["TARGET"] == 0, col],
                          valid.loc[valid["TARGET"] == 1, col])
        bureau_ks[col] = ks
bureau_ks_df = pd.Series(bureau_ks).sort_values(ascending=False)
print(bureau_ks_df.to_string())

fig, ax = plt.subplots(figsize=(10, 6))
bureau_ks_df.sort_values().plot.barh(ax=ax, color="#FF7043")
ax.set_title("Bureau Aggregated Features — KS Statistic vs TARGET",
             fontweight="bold")
ax.set_xlabel("KS Statistic")
plt.tight_layout()
plt.savefig(FIG_DIR / "16_bureau_ks.png", dpi=150)
plt.close()

# Visualise top 2 bureau features
top_bureau = bureau_ks_df.head(4).index.tolist()
fig, axes = plt.subplots(1, len(top_bureau), figsize=(18, 5))
for ax, col in zip(axes, top_bureau):
    g0 = app_b.loc[(app_b["TARGET"] == 0) & app_b[col].notna(), col]
    g1 = app_b.loc[(app_b["TARGET"] == 1) & app_b[col].notna(), col]
    cap = app_b[col].quantile(0.98)
    ax.hist(g0.clip(upper=cap), bins=40, alpha=0.6, color="#2196F3",
            label="No difficulty", density=True)
    ax.hist(g1.clip(upper=cap), bins=40, alpha=0.6, color="#F44336",
            label="Difficulty", density=True)
    ks = bureau_ks_df[col]
    ax.set_title(f"{col}\n(KS={ks:.3f})", fontweight="bold", fontsize=9)
    ax.legend(fontsize=8)

plt.suptitle("Top Bureau Features by KS", fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "17_bureau_top_features.png", dpi=150)
plt.close()


# =============================================================================
# 7. PREVIOUS APPLICATION EDA
# =============================================================================
print("\n--- 7. Previous Application Data ---")

prev = pd.read_csv(PREV_APP)
print(f"  previous_application shape : {prev.shape}")
print(f"  Unique SK_ID_CURR           : {prev['SK_ID_CURR'].nunique():,}")

# Contract status breakdown
print("\n  Previous application contract status:")
print(prev["NAME_CONTRACT_STATUS"].value_counts().to_string())

# Aggregate to application level
prev["DAYS_DECISION_ABS"] = prev["DAYS_DECISION"].abs()
prev_approved = prev[prev["NAME_CONTRACT_STATUS"] == "Approved"]

prev_agg = prev.groupby("SK_ID_CURR").agg(
    prev_n_applications        =("SK_ID_PREV", "count"),
    prev_n_approved            =("NAME_CONTRACT_STATUS",
                                  lambda x: (x == "Approved").sum()),
    prev_n_refused             =("NAME_CONTRACT_STATUS",
                                  lambda x: (x == "Refused").sum()),
    prev_n_cancelled           =("NAME_CONTRACT_STATUS",
                                  lambda x: (x == "Cancelled").sum()),
    prev_avg_credit            =("AMT_CREDIT", "mean"),
    prev_max_days_decision     =("DAYS_DECISION_ABS", "max"),
    prev_avg_days_decision     =("DAYS_DECISION_ABS", "mean"),
    prev_n_cash                =("NAME_CONTRACT_TYPE",
                                  lambda x: (x == "Cash loans").sum()),
    prev_n_revolving           =("NAME_CONTRACT_TYPE",
                                  lambda x: (x == "Revolving loans").sum()),
).reset_index()

prev_agg["prev_approval_rate"] = (prev_agg["prev_n_approved"]
                                    / prev_agg["prev_n_applications"])
prev_agg["prev_refusal_rate"]  = (prev_agg["prev_n_refused"]
                                    / prev_agg["prev_n_applications"])

app_p = app[["SK_ID_CURR", "TARGET"]].merge(prev_agg, on="SK_ID_CURR", how="left")

# KS for prev features
prev_num = [c for c in prev_agg.columns if c != "SK_ID_CURR"]
prev_ks = {}
for col in prev_num:
    valid = app_p[[col, "TARGET"]].dropna()
    if len(valid) > 100:
        ks, _ = ks_2samp(valid.loc[valid["TARGET"] == 0, col],
                          valid.loc[valid["TARGET"] == 1, col])
        prev_ks[col] = ks
prev_ks_df = pd.Series(prev_ks).sort_values(ascending=False)

print("\n  Previous application KS statistics vs TARGET:")
print(prev_ks_df.to_string())

fig, ax = plt.subplots(figsize=(10, 6))
prev_ks_df.sort_values().plot.barh(ax=ax, color="#26A69A")
ax.set_title("Previous Application Features — KS Statistic vs TARGET",
             fontweight="bold")
ax.set_xlabel("KS Statistic")
plt.tight_layout()
plt.savefig(FIG_DIR / "18_prev_app_ks.png", dpi=150)
plt.close()


# =============================================================================
# 8. POS/CASH BALANCE & INSTALLMENTS SUMMARY
# =============================================================================
print("\n--- 8. POS/Cash & Installments ---")

pos  = pd.read_csv(POS_CASH)
inst = pd.read_csv(INSTALL)
print(f"  POS_CASH_balance shape     : {pos.shape}")
print(f"  installments_payments shape: {inst.shape}")

# POS: DPD behaviour
pos_agg = pos.groupby("SK_ID_CURR").agg(
    pos_n_months      =("MONTHS_BALANCE", "count"),
    pos_max_dpd       =("SK_DPD", "max"),
    pos_avg_dpd       =("SK_DPD", "mean"),
    pos_n_dpd_months  =("SK_DPD", lambda x: (x > 0).sum()),
    pos_late_payments =("SK_DPD_DEF", lambda x: (x > 0).sum()),
).reset_index()

# Installments: payment behaviour
inst["PAYMENT_DIFF"]    = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
inst["DAYS_LATE"]       = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
inst["LATE_FLAG"]       = (inst["DAYS_LATE"] > 0).astype(int)

inst_agg = inst.groupby("SK_ID_CURR").agg(
    inst_n_instalments  =("NUM_INSTALMENT_NUMBER", "count"),
    inst_max_days_late  =("DAYS_LATE", "max"),
    inst_avg_days_late  =("DAYS_LATE", "mean"),
    inst_n_late         =("LATE_FLAG", "sum"),
    inst_max_underpay   =("PAYMENT_DIFF", "max"),
    inst_total_underpay =("PAYMENT_DIFF", lambda x: x.clip(lower=0).sum()),
).reset_index()

inst_agg["inst_late_rate"] = inst_agg["inst_n_late"] / inst_agg["inst_n_instalments"]

app_i = app[["SK_ID_CURR", "TARGET"]].merge(inst_agg, on="SK_ID_CURR", how="left")

# KS
inst_ks = {}
for col in [c for c in inst_agg.columns if c != "SK_ID_CURR"]:
    valid = app_i[[col, "TARGET"]].dropna()
    if len(valid) > 100:
        ks, _ = ks_2samp(valid.loc[valid["TARGET"] == 0, col],
                          valid.loc[valid["TARGET"] == 1, col])
        inst_ks[col] = ks
inst_ks_df = pd.Series(inst_ks).sort_values(ascending=False)

print("\n  Installment feature KS statistics:")
print(inst_ks_df.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
inst_ks_df.sort_values().plot.barh(ax=ax, color="#8D6E63")
ax.set_title("Installment Payment Features — KS vs TARGET", fontweight="bold")
ax.set_xlabel("KS Statistic")
plt.tight_layout()
plt.savefig(FIG_DIR / "19_installment_ks.png", dpi=150)
plt.close()


# =============================================================================
# 9. CREDIT CARD BALANCE EDA
# =============================================================================
print("\n--- 9. Credit Card Balance ---")

cc = pd.read_csv(CC_BAL)
print(f"  credit_card_balance shape : {cc.shape}")

cc["UTILIZATION"] = (cc["AMT_BALANCE"]
                     / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan))
cc["PAYMENT_RATIO"] = (cc["AMT_PAYMENT_CURRENT"]
                        / cc["AMT_INST_MIN_REGULARITY"].replace(0, np.nan))

cc_agg = cc.groupby("SK_ID_CURR").agg(
    cc_n_months          =("MONTHS_BALANCE", "count"),
    cc_avg_balance       =("AMT_BALANCE", "mean"),
    cc_max_balance       =("AMT_BALANCE", "max"),
    cc_avg_utilization   =("UTILIZATION", "mean"),
    cc_max_utilization   =("UTILIZATION", "max"),
    cc_max_dpd           =("SK_DPD", "max"),
    cc_n_dpd             =("SK_DPD", lambda x: (x > 0).sum()),
    cc_avg_drawings      =("AMT_DRAWINGS_CURRENT", "mean"),
).reset_index()

app_c = app[["SK_ID_CURR", "TARGET"]].merge(cc_agg, on="SK_ID_CURR", how="left")

cc_ks = {}
for col in [c for c in cc_agg.columns if c != "SK_ID_CURR"]:
    valid = app_c[[col, "TARGET"]].dropna()
    if len(valid) > 100:
        ks, _ = ks_2samp(valid.loc[valid["TARGET"] == 0, col],
                          valid.loc[valid["TARGET"] == 1, col])
        cc_ks[col] = ks
cc_ks_df = pd.Series(cc_ks).sort_values(ascending=False)
print("\n  Credit card feature KS statistics:")
print(cc_ks_df.to_string())

fig, ax = plt.subplots(figsize=(9, 5))
cc_ks_df.sort_values().plot.barh(ax=ax, color="#EF9A9A")
ax.set_title("Credit Card Features — KS vs TARGET", fontweight="bold")
ax.set_xlabel("KS Statistic")
plt.tight_layout()
plt.savefig(FIG_DIR / "20_cc_balance_ks.png", dpi=150)
plt.close()


# =============================================================================
# 10. WoE / IV ANALYSIS
# =============================================================================
print("\n--- 10. WoE / IV Analysis ---")

if SCORECARDPY_AVAILABLE:
    # Select a manageable subset of application features for WoE
    woe_num_cols = [
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "DAYS_BIRTH", "DAYS_EMPLOYED_CLEAN",
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO",
        "DAYS_ID_PUBLISH", "DAYS_REGISTRATION", "DAYS_LAST_PHONE_CHANGE",
        "AMT_REQ_CREDIT_BUREAU_YEAR", "AMT_REQ_CREDIT_BUREAU_MON",
        "OWN_CAR_AGE", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
        "REGION_RATING_CLIENT",
    ]
    woe_cat_cols = [
        "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY", "NAME_TYPE_SUITE",
        "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
        "WEEKDAY_APPR_PROCESS_START",
    ]
    woe_cols = [c for c in woe_num_cols + woe_cat_cols if c in app.columns]
    woe_df = app[woe_cols + ["TARGET"]].copy()

    bins = sc.woebin(woe_df, y="TARGET", x=woe_cols,
                     positive="bad|1", print_info=False,
                     no_cores=1, check_cate_num=False)

    # IV summary
    iv_vals = {feat: bins[feat]["bin_iv"].sum() for feat in bins}
    iv_df = pd.Series(iv_vals).sort_values(ascending=False)

    print("\n  Information Value summary:")
    print("  IV interpretation: <0.02 useless | 0.02-0.1 weak | "
          "0.1-0.3 medium | >0.3 strong")
    print(iv_df.to_string())

    fig, ax = plt.subplots(figsize=(10, 8))
    iv_df.sort_values().plot.barh(
        ax=ax,
        color=[
            "#F44336" if v > 0.3 else
            "#FF9800" if v > 0.1 else
            "#4CAF50" if v > 0.02 else "#9E9E9E"
            for v in iv_df.sort_values().values
        ]
    )
    ax.axvline(0.02, color="gray",   linestyle="--", linewidth=1, label="Weak (0.02)")
    ax.axvline(0.1,  color="orange", linestyle="--", linewidth=1, label="Medium (0.10)")
    ax.axvline(0.3,  color="red",    linestyle="--", linewidth=1, label="Strong (0.30)")
    ax.set_title("Information Value (IV) by Feature", fontweight="bold")
    ax.set_xlabel("IV")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "21_iv_summary.png", dpi=150)
    plt.close()

    # WoE plots for top 4 features
    top_iv_feats = iv_df.head(4).index.tolist()
    fig, axes = plt.subplots(1, len(top_iv_feats), figsize=(20, 5))
    for ax, feat in zip(axes, top_iv_feats):
        bin_df = bins[feat].copy()
        bin_df["bin"] = bin_df["bin"].astype(str)
        ax.bar(range(len(bin_df)), bin_df["woe"], color="#7986CB", edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(bin_df)))
        ax.set_xticklabels(bin_df["bin"], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{feat}\nIV={iv_df[feat]:.4f}", fontweight="bold", fontsize=9)
        ax.set_ylabel("WoE")
    plt.suptitle("WoE Plots — Top 4 Features", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "22_woe_top_features.png", dpi=150, bbox_inches="tight")
    plt.close()
else:
    print("  scorecardpy not installed — skipping WoE/IV analysis")
    print("  Install with: pip install scorecardpy")


# =============================================================================
# 11. CORRELATION & MULTICOLLINEARITY
# =============================================================================
print("\n--- 11. Correlation Matrix ---")

# Select most predictive numericals for correlation heatmap
key_num = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "DAYS_BIRTH", "AGE_YEARS",
    "DAYS_EMPLOYED_CLEAN", "YEARS_EMPLOYED",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM",
    "DAYS_ID_PUBLISH", "DAYS_REGISTRATION",
    "EXT_MEAN", "EXT_MIN",
    "TARGET"
]
key_num = [c for c in key_num if c in app.columns]

corr_matrix = app[key_num].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.7},
            ax=ax, annot_kws={"size": 8})
ax.set_title("Correlation Matrix — Key Numerical Features", fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "23_correlation_matrix.png", dpi=150)
plt.close()

# Flag high-correlation pairs (potential multicollinearity)
print("\n  Highly correlated feature pairs (|r| > 0.70, excluding TARGET):")
corr_no_target = app[[c for c in key_num if c != "TARGET"]].corr()
for i in range(len(corr_no_target.columns)):
    for j in range(i + 1, len(corr_no_target.columns)):
        r = corr_no_target.iloc[i, j]
        if abs(r) > 0.70:
            c1 = corr_no_target.columns[i]
            c2 = corr_no_target.columns[j]
            print(f"  {c1}  ×  {c2}  →  r={r:.3f}")


# =============================================================================
# 12. SUMMARY — TOP PREDICTORS SHORTLIST
# =============================================================================
print("\n" + "=" * 70)
print("12. TOP PREDICTORS SHORTLIST")
print("=" * 70)

all_ks = {}

# Application numericals
for col in num_cols:
    valid = app[[col, "TARGET"]].dropna()
    if len(valid) > 100:
        ks, _ = ks_2samp(valid.loc[valid["TARGET"] == 0, col],
                          valid.loc[valid["TARGET"] == 1, col])
        all_ks[f"app__{col}"] = ks

# Bureau features
for col, ks in bureau_ks.items():
    all_ks[f"bureau__{col}"] = ks

# Previous application
for col, ks in prev_ks.items():
    all_ks[f"prev__{col}"] = ks

# Installments
for col, ks in inst_ks.items():
    all_ks[f"inst__{col}"] = ks

# Credit card
for col, ks in cc_ks.items():
    all_ks[f"cc__{col}"] = ks

ks_summary = pd.Series(all_ks).sort_values(ascending=False)

print(f"\nTop 30 features by KS statistic across all tables:\n")
print(ks_summary.head(30).to_string())

fig, ax = plt.subplots(figsize=(12, 10))
ks_summary.head(30).sort_values().plot.barh(
    ax=ax,
    color=["#F44336" if v > 0.20 else
           "#FF9800" if v > 0.15 else "#4CAF50"
           for v in ks_summary.head(30).sort_values().values]
)
ax.axvline(0.20, color="red",   linestyle="--", linewidth=1.2, label="KS=0.20 (good)")
ax.axvline(0.15, color="orange", linestyle="--", linewidth=1.2, label="KS=0.15 (acceptable)")
ax.set_title("Top 30 Features (All Tables) — KS Statistic vs TARGET",
             fontweight="bold")
ax.set_xlabel("KS Statistic")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "24_top_features_overall.png", dpi=150)
plt.close()

print(f"\nAll figures saved to: {FIG_DIR}")

print("\n" + "=" * 70)
print("EDA COMPLETE — Key findings:")
print("=" * 70)
print("""
1.  CLASS IMBALANCE: ~8% default rate → use class_weight='balanced'
    or scale_pos_weight ≈ 11 in XGBoost.

2.  STRONGEST PREDICTORS (application table):
    - EXT_SOURCE_2, EXT_SOURCE_3, EXT_SOURCE_1 (external credit scores)
    - DAYS_BIRTH (age) — younger applicants default more
    - DAYS_EMPLOYED — shorter tenure correlates with higher default
    - DAYS_ID_PUBLISH — recent ID change is a risk flag
    - AMT_ANNUITY / AMT_INCOME (affordability ratios)

3.  BUREAU: bureau_max_dpd, bureau_n_dpd_credits, bureau_dpd_rate
    are highly predictive. Clients with prior delinquencies default more.

4.  INSTALLMENTS: inst_max_days_late, inst_late_rate from payment history
    are strong behavioural signals.

5.  PREVIOUS APPLICATIONS: prev_refusal_rate (proportion of past
    applications refused) is an important risk signal.

6.  MISSING VALUES:
    - EXT_SOURCE_1 (~56% missing) — impute with median, add missingness flag
    - OWN_CAR_AGE (~66% missing) — create FLAG_CAR + impute
    - Building features (APARTMENTS_*, BASEMENTAREA_*, etc.) ~50% missing

7.  ANOMALIES:
    - DAYS_EMPLOYED == 365243 (pensioner/unemployed sentinel) —
      replace with NaN and add binary FLAG_EMPLOYED_ANOMALY

8.  RECOMMENDED FEATURE ENGINEERING:
    - EXT_MEAN, EXT_MIN (aggregate EXT_SOURCEs)
    - CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO
    - CREDIT_TERM = AMT_CREDIT / AMT_ANNUITY
    - AGE_YEARS, YEARS_EMPLOYED
    - Bureau: max/mean DPD, overdue amounts
    - Installments: late payment rate, max days late

Next step → 02_feature_engineering.py
""")
