"""
Credit Risk PD Model — Challenger: Gradient Boosting (XGBoost) with SHAP
===========================================================================
Tests how much a more expressive model improves over the WoE logistic
regression baseline (03). Trained on raw/capped features with native
categoricals and NaNs — no WoE needed, XGBoost handles both natively.

Monotonic constraints on features with a known risk direction (e.g. higher
EXT_SOURCE -> lower PD) keep the model from learning non-monotonic kinks
that wouldn't hold up under review or generalise to new applicants.

Methodology mirrors 03: a 3-way split of `train` into fit / early-stopping /
calibration folds, with `valid` (held out since 02, untouched until here)
used only for final metrics.

Run with:
    python notebooks/04_gradient_boosting_shap.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ks_2samp
import shap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "data_processed"
ARTIFACT_DIR = ROOT / "models" / "artifacts"
MODEL_DIR = ROOT / "models"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEED = 42
TARGET = "TARGET"
ID_COL = "SK_ID_CURR"

# Known risk direction from the EDA (01) and feature-engineering rationale (02).
# +1: higher value -> higher PD.  -1: higher value -> lower PD.
MONOTONE_DIRECTIONS = {
    "EXT_SOURCE_1": -1, "EXT_SOURCE_2": -1, "EXT_SOURCE_3": -1,
    "EXT_MEAN": -1, "EXT_MIN": -1, "EXT_MAX": -1,
    "CREDIT_INCOME_RATIO": 1, "ANNUITY_INCOME_RATIO": 1,
    "AGE_YEARS": -1, "YEARS_EMPLOYED": -1,
    "bureau_dpd_rate": 1, "bureau_max_dpd": 1, "bureau_worst_status_rank": 1,
    "inst_late_rate": 1, "inst_late_rate_last6": 1,
    "prev_refusal_rate": 1, "cc_dpd_rate": 1,
}


def psi(expected, actual, buckets=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_pcts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pcts = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    return np.sum(
        (actual_pcts - expected_pcts) * np.log((actual_pcts + 1e-8) / (expected_pcts + 1e-8))
    )


print("=" * 70)
print("CHALLENGER — GRADIENT BOOSTING (XGBOOST) WITH SHAP")
print("=" * 70)

print("\nLoading raw/capped features...")
train = pd.read_parquet(DATA_DIR / "train_features.parquet")
valid = pd.read_parquet(DATA_DIR / "valid_features.parquet")

with open(ARTIFACT_DIR / "feature_cols.json") as f:
    meta = json.load(f)
feature_cols, cat_cols = meta["feature_cols"], meta["cat_cols"]
print(f"  train: {train.shape}  valid: {valid.shape}  features: {len(feature_cols)} "
      f"({len(cat_cols)} categorical)")

X_train, y_train = train[feature_cols], train[TARGET]
X_valid, y_valid = valid[feature_cols], valid[TARGET]

monotone = {c: MONOTONE_DIRECTIONS[c] for c in feature_cols if c in MONOTONE_DIRECTIONS}
print(f"\nMonotonic constraints on {len(monotone)}/{len(feature_cols)} features")

# ---------------------------------------------------------------------------
# 1. Three-way split: fit (64%) / early-stopping (16%) / calibration (20%)
# ---------------------------------------------------------------------------
X_fit, X_rest, y_fit, y_rest = train_test_split(
    X_train, y_train, test_size=0.36, stratify=y_train, random_state=SEED
)
X_es, X_calib, y_es, y_calib = train_test_split(
    X_rest, y_rest, test_size=0.20 / 0.36, stratify=y_rest, random_state=SEED
)
print(f"Fit: {X_fit.shape}  Early-stop: {X_es.shape}  Calib: {X_calib.shape}")

scale_pos_weight = (1 - y_fit.mean()) / y_fit.mean()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# ---------------------------------------------------------------------------
# 2. Train with early stopping
# ---------------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    monotone_constraints=monotone,
    enable_categorical=True,
    tree_method="hist",
    eval_metric="auc",
    early_stopping_rounds=30,
    random_state=SEED,
)

print("\nTraining with early stopping (eval on early-stop fold)...")
model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
print(f"Best iteration: {model.best_iteration}")

# ---------------------------------------------------------------------------
# 3. Recalibrate on the calibration fold (raw tree-ensemble probabilities
#    are typically miscalibrated at the tails, independent of scale_pos_weight)
# ---------------------------------------------------------------------------
print("\nCalibrating (isotonic) on calib fold...")
calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
calibrated.fit(X_calib, y_calib)

# ---------------------------------------------------------------------------
# 4. Validation — on the fully held-out `valid` split only
# ---------------------------------------------------------------------------
fit_scores_raw = model.predict_proba(X_fit)[:, 1]
valid_pd = calibrated.predict_proba(X_valid)[:, 1]

auc = roc_auc_score(y_valid, valid_pd)
gini = 2 * auc - 1
ks, _ = ks_2samp(valid_pd[y_valid == 1], valid_pd[y_valid == 0])
psi_score = psi(fit_scores_raw, model.predict_proba(X_valid)[:, 1])

print("\n" + "-" * 50)
print(f"VALIDATION  |  AUC: {auc:.4f}  Gini: {gini:.4f}  KS: {ks:.4f}  PSI: {psi_score:.4f}")
print(f"Mean predicted PD (valid): {valid_pd.mean():.2%}  "
      f"True default rate (valid): {y_valid.mean():.2%}")
print("-" * 50)

fpr, tpr, _ = roc_curve(y_valid, valid_pd)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(fpr, tpr, color="#F44336", label=f"AUC = {auc:.3f}")
axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[0].set_title("ROC Curve — XGBoost Challenger", fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

axes[1].hist(valid_pd[y_valid == 0], bins=40, alpha=0.6, label="Non-default", color="#2196F3", density=True)
axes[1].hist(valid_pd[y_valid == 1], bins=40, alpha=0.6, label="Default", color="#F44336", density=True)
axes[1].set_title(f"Score Separation (KS = {ks:.3f})", fontweight="bold")
axes[1].set_xlabel("Predicted PD")
axes[1].legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "26_gbm_validation.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. SHAP explainability
# ---------------------------------------------------------------------------
print("\nComputing SHAP values on a 5,000-row sample of valid...")
shap_sample = X_valid.sample(n=min(5000, len(X_valid)), random_state=SEED)
explainer = shap.TreeExplainer(model)
shap_values = explainer(shap_sample)

plt.figure()
shap.summary_plot(shap_values, shap_sample, show=False, max_display=20)
plt.tight_layout()
plt.savefig(FIG_DIR / "27_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

mean_abs_shap = pd.Series(
    np.abs(shap_values.values).mean(axis=0), index=feature_cols
).sort_values(ascending=False)
print("\nTop 15 features by mean |SHAP|:")
print(mean_abs_shap.head(15).to_string())

top4 = mean_abs_shap.head(4).index.tolist()
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for ax, feat in zip(axes, top4):
    shap.dependence_plot(feat, shap_values.values, shap_sample, interaction_index=None, ax=ax, show=False)
    ax.set_title(feat, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "28_shap_dependence.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 6. Compare against the baseline
# ---------------------------------------------------------------------------
baseline_path = MODEL_DIR / "baseline_metrics.json"
if baseline_path.exists():
    baseline = json.loads(baseline_path.read_text())
    print("\n" + "=" * 50)
    print("BASELINE vs CHALLENGER")
    print("=" * 50)
    print(f"{'Metric':<10}{'LR (WoE)':>12}{'XGBoost':>12}{'Delta':>12}")
    for m in ("auc", "gini", "ks"):
        b, c = baseline[m], {"auc": auc, "gini": gini, "ks": ks}[m]
        print(f"{m.upper():<10}{b:>12.4f}{c:>12.4f}{c - b:>+12.4f}")
else:
    print("\nNo baseline_metrics.json found — run 03_baseline_logistic_regression.py first for comparison.")

# ---------------------------------------------------------------------------
# 7. Save artifacts
# ---------------------------------------------------------------------------
with open(MODEL_DIR / "gbm_xgboost.pkl", "wb") as f:
    pickle.dump({"model": model, "calibrated": calibrated, "feature_cols": feature_cols,
                 "cat_cols": cat_cols, "monotone_constraints": monotone}, f)

metrics = {"auc": auc, "gini": gini, "ks": ks, "psi": psi_score,
           "best_iteration": int(model.best_iteration)}
pd.Series(metrics).to_json(MODEL_DIR / "gbm_metrics.json", indent=2)

print("\n" + "=" * 70)
print("CHALLENGER COMPLETE")
print("=" * 70)
print(f"Model saved -> {MODEL_DIR / 'gbm_xgboost.pkl'}")
print(f"SHAP figures -> {FIG_DIR}")
print(f"Metrics: {metrics}")
