"""
Credit Risk PD Model — Feature Engineering
===========================================
Builds the modelling table from application_train/test plus the 5 auxiliary
tables (bureau, bureau_balance, previous_application, POS_CASH_balance,
installments_payments, credit_card_balance).

Kaggle's application_test.csv has no TARGET, so it can't be used for
validation — we carve an internal labelled holdout out of application_train
instead. All fitted artifacts (outlier caps, WoE bins) are fit on the
internal train split only and applied to valid/test to avoid leakage.

Outputs (data/data_processed/):
    train_features.parquet / valid_features.parquet / test_features.parquet
        — raw/capped features + native categoricals, for the GBM model
    train_woe.parquet / valid_woe.parquet / test_woe.parquet
        — WoE-encoded features, for the logistic regression scorecard
Artifacts (models/artifacts/):
    outlier_caps.json, woe_bins.pkl, feature_cols.json

Run with:
    python notebooks/02_feature_engineering.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scorecardpy as sc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features import (  # noqa: E402
    compute_all_aggregates,
    merge_aggregates,
    fit_outlier_caps,
    apply_outlier_caps,
)

# Paths
DATA_RAW = ROOT / "data" / "data_raw"
APP_TRAIN = DATA_RAW / "home-credit-default-risk" / "application_train.csv"
APP_TEST  = DATA_RAW / "home-credit-default-risk" / "application_test.csv"
BUREAU    = DATA_RAW / "bureau.csv"
BB        = DATA_RAW / "bureau_balance.csv"
PREV_APP  = DATA_RAW / "previous_application.csv"
POS_CASH  = DATA_RAW / "POS_CASH_balance.csv"
INSTALL   = DATA_RAW / "installments_payments.csv"
CC_BAL    = DATA_RAW / "credit_card_balance.csv"

OUT_DIR = ROOT / "data" / "data_processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = ROOT / "models" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "TARGET"
ID_COL = "SK_ID_CURR"

# Building-quality block — 14 columns share the same missingness pattern (no building
# survey on file); one count feature instead of 14 near-duplicate flags avoids collinearity.
BUILDING_AVG_COLS = [
    "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG",
    "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG",
    "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAREA_AVG",
]

# Superseded by engineered equivalents — kept out of the modelling feature set.
DROP_RAW_COLS = ["DAYS_EMPLOYED"]


def add_curated_missingness(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["FLAG_OWN_CAR_AGE_MISSING"] = df["OWN_CAR_AGE"].isna().astype(int)
    present = [c for c in BUILDING_AVG_COLS if c in df.columns]
    df["BUILDING_INFO_MISSING_COUNT"] = df[present].isna().sum(axis=1)
    return df


print("=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

print("\nLoading raw tables...")
app_train_full = pd.read_csv(APP_TRAIN)
app_test_full = pd.read_csv(APP_TEST)
bureau = pd.read_csv(BUREAU)
bb = pd.read_csv(BB)
prev = pd.read_csv(PREV_APP)
pos = pd.read_csv(POS_CASH)
inst = pd.read_csv(INSTALL)
cc = pd.read_csv(CC_BAL)
print(f"  application_train : {app_train_full.shape}")
print(f"  application_test  : {app_test_full.shape}")
print(f"  bureau            : {bureau.shape}   bureau_balance : {bb.shape}")
print(f"  previous_app      : {prev.shape}")
print(f"  POS_CASH          : {pos.shape}")
print(f"  installments      : {inst.shape}")
print(f"  credit_card       : {cc.shape}")

# ---------------------------------------------------------------------------
# 1. Internal labelled train/valid split
# ---------------------------------------------------------------------------
train_df, valid_df = train_test_split(
    app_train_full, test_size=0.2, stratify=app_train_full[TARGET], random_state=SEED
)
print(f"\nInternal split -> train: {train_df.shape}, valid: {valid_df.shape}")
print(f"  train default rate: {train_df[TARGET].mean():.2%}")
print(f"  valid default rate: {valid_df[TARGET].mean():.2%}")

# ---------------------------------------------------------------------------
# 2. Aggregate auxiliary tables ONCE, merge onto each split
# ---------------------------------------------------------------------------
print("\nAggregating auxiliary tables to SK_ID_CURR grain (bureau_balance alone is "
      f"{len(bb):,} rows — this is the slow step)...")
aggregates = compute_all_aggregates(bureau, bb, prev, pos, inst, cc)
del bureau, bb, prev, pos, inst, cc

print("Merging onto train / valid / test...")
train_feat = merge_aggregates(train_df, aggregates)
valid_feat = merge_aggregates(valid_df, aggregates)
test_feat = merge_aggregates(app_test_full, aggregates)
del aggregates
print(f"  train_feat: {train_feat.shape}  valid_feat: {valid_feat.shape}  test_feat: {test_feat.shape}")

# ---------------------------------------------------------------------------
# 3. Curated missingness flags (fixed, no fitting needed)
# ---------------------------------------------------------------------------
train_feat = add_curated_missingness(train_feat)
valid_feat = add_curated_missingness(valid_feat)
test_feat = add_curated_missingness(test_feat)

# ---------------------------------------------------------------------------
# 4. Outlier capping — fit on train only
# ---------------------------------------------------------------------------
numeric_cols = [
    c for c in train_feat.select_dtypes(include=[np.number]).columns
    if c not in (ID_COL, TARGET)
]
print(f"\nFitting outlier caps on {len(numeric_cols)} numeric columns (train only)...")
caps = fit_outlier_caps(train_feat, numeric_cols)
train_feat = apply_outlier_caps(train_feat, caps)
valid_feat = apply_outlier_caps(valid_feat, caps)
test_feat = apply_outlier_caps(test_feat, caps)

caps_json = {k: [float(lo), float(hi)] for k, (lo, hi) in caps.items()}
with open(ARTIFACT_DIR / "outlier_caps.json", "w") as f:
    json.dump(caps_json, f, indent=2)

# ---------------------------------------------------------------------------
# 5. Drop superseded raw columns, define final feature set
# ---------------------------------------------------------------------------
for df_ in (train_feat, valid_feat, test_feat):
    df_.drop(columns=[c for c in DROP_RAW_COLS if c in df_.columns], inplace=True)

feature_cols = [c for c in train_feat.columns if c not in (ID_COL, TARGET)]
cat_cols = train_feat[feature_cols].select_dtypes(include=["object"]).columns.tolist()
print(f"\nFinal feature set: {len(feature_cols)} columns ({len(cat_cols)} categorical)")

with open(ARTIFACT_DIR / "feature_cols.json", "w") as f:
    json.dump({"feature_cols": feature_cols, "cat_cols": cat_cols}, f, indent=2)

# ---------------------------------------------------------------------------
# 6. Save GBM-ready feature sets (native categoricals, NaNs kept as-is)
# ---------------------------------------------------------------------------
# Categorical dtype categories are fit on train only, then applied as-is to
# valid/test (same fit-on-train-only discipline as outlier caps / WoE bins).
# Casting each split's categoricals independently would let pandas infer a
# different category->code mapping per split whenever the set of unique values
# differs (e.g. a rare category present in train but absent from valid/test) --
# XGBoost's enable_categorical path splits on those integer codes, so a
# per-split mismatch silently corrupts predictions for every row whose
# category's code shifted. Unseen categories in valid/test become NaN, which
# XGBoost handles natively as missing.
print("\nSaving raw/capped feature sets (GBM-ready)...")
train_out = train_feat.copy()
train_cat_categories = {}
for c in cat_cols:
    train_out[c] = train_out[c].astype("category")
    train_cat_categories[c] = train_out[c].cat.categories
train_out.to_parquet(OUT_DIR / "train_features.parquet", index=False)
print(f"  train_features.parquet -> {train_out.shape}")

for name, df_ in (("valid", valid_feat), ("test", test_feat)):
    out = df_.copy()
    for c in cat_cols:
        out[c] = pd.Categorical(out[c], categories=train_cat_categories[c])
    out.to_parquet(OUT_DIR / f"{name}_features.parquet", index=False)
    print(f"  {name}_features.parquet -> {out.shape}")

# ---------------------------------------------------------------------------
# 7. WoE binning — fit on train only, apply to valid/test (LR-ready)
# ---------------------------------------------------------------------------
print("\nPre-filtering candidate features by IV / missing-rate / identical-rate "
      "(scorecardpy.var_filter)...")
woe_input_cols = [c for c in feature_cols if c != ID_COL]
train_filtered = sc.var_filter(
    train_feat[woe_input_cols + [TARGET]], y=TARGET, iv_limit=0.02, missing_limit=0.95
)
selected_cols = [c for c in train_filtered.columns if c != TARGET]
print(f"  {len(selected_cols)} / {len(woe_input_cols)} columns survived filtering")

print("\nFitting WoE bins on train (monotonic binning via scorecardpy.woebin)...")
# no_cores=1: scorecardpy defaults to multiprocessing across all cores once x >= 10 columns,
# which on Windows re-imports this script as __main__ in each worker with no guard against
# re-running the whole pipeline — force single-core to avoid runaway process spawning.
bins = sc.woebin(train_feat[selected_cols + [TARGET]], y=TARGET, no_cores=1, check_cate_num=False)

with open(ARTIFACT_DIR / "woe_bins.pkl", "wb") as f:
    pickle.dump(bins, f)

print("Applying WoE transform to train / valid / test...")
train_woe = sc.woebin_ply(train_feat[selected_cols + [TARGET]], bins, no_cores=1)
valid_woe = sc.woebin_ply(valid_feat[selected_cols + [TARGET]], bins, no_cores=1)
test_woe = sc.woebin_ply(test_feat[selected_cols], bins, no_cores=1)

for name, df_, ids in (
    ("train", train_woe, train_feat[ID_COL]),
    ("valid", valid_woe, valid_feat[ID_COL]),
    ("test", test_woe, test_feat[ID_COL]),
):
    df_.insert(0, ID_COL, ids.values)
    df_.to_parquet(OUT_DIR / f"{name}_woe.parquet", index=False)
    print(f"  {name}_woe.parquet -> {df_.shape}")

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)
print(f"""
Outputs      : {OUT_DIR}
Artifacts    : {ARTIFACT_DIR}
GBM features : {len(feature_cols)} columns ({len(cat_cols)} native categorical)
WoE features : {len(selected_cols)} columns survived IV/missing/identical filtering

Next step -> 03_baseline_logistic_regression.py / 04_gradient_boosting_shap.py
""")
