"""
Credit Risk PD Model — Second Challenger: Gradient Boosting (LightGBM) with SHAP
===========================================================================
Second challenger alongside 04 (XGBoost) — same raw/capped features, native
categoricals and NaNs, same monotonic-constraint rationale, same 3-way
fit / early-stopping / calibration split of `train`, with `valid` (held out
since 02) used only for final metrics. Lets us compare two different
boosting implementations on identical data/splits rather than re-deriving
methodology.

Run with:
    python notebooks/06_lightgbm_challenger.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import json
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split, ParameterSampler
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
# Isotonic calibration's lowest step can land exactly on 0.0 -- log(PD/(1-PD))
# and PD x LGD x EAD both break on an exact 0, so floor the calibrated output
# away from the boundary. Same floor as 03/04/05.
PD_FLOOR = 1e-4

# Same known-risk-direction map as 04 (CREDIT_INCOME_RATIO deliberately
# excluded -- empirical bad-rate-by-decile on train_features.parquet is
# hump-shaped, not monotonic, see 04 for the decile table).
MONOTONE_DIRECTIONS = {
    "EXT_SOURCE_1": -1, "EXT_SOURCE_2": -1, "EXT_SOURCE_3": -1,
    "EXT_MEAN": -1, "EXT_MIN": -1, "EXT_MAX": -1,
    "ANNUITY_INCOME_RATIO": 1,
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
print("SECOND CHALLENGER — GRADIENT BOOSTING (LIGHTGBM) WITH SHAP")
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

# LightGBM's monotone_constraints takes a list positional to X's columns
# (unlike XGBoost's dict-by-name) -- unconstrained features get 0.
monotone = [MONOTONE_DIRECTIONS.get(c, 0) for c in feature_cols]
print(f"\nMonotonic constraints on {sum(v != 0 for v in monotone)}/{len(feature_cols)} features")

# ---------------------------------------------------------------------------
# 1. Three-way split: fit (64%) / early-stopping (16%) / calibration (20%)
#    Identical split logic/ratios/seed to 04 for a fair, matched comparison.
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
# 2. Hyperparameter search — random search, early-stopped on the ES fold
#    (same rationale as 04: a full k-fold search would need a separate
#    early-stopped fit per fold per candidate, too slow at this scale)
# ---------------------------------------------------------------------------
# subsample_freq must be >0 for LightGBM's `subsample` (bagging_fraction) to
# actually take effect -- it's silently ignored at the default freq of 0.
# Fixed at 1 (bag every iteration) rather than searched.
SUBSAMPLE_FREQ = 1

# first_metric_only=True is required here, not optional: passing eval_metric="auc"
# to .fit() does NOT replace LightGBM's default binary_logloss eval metric, it adds
# to it -- both get tracked. Without first_metric_only, early_stopping demands ALL
# tracked metrics improve each round to reset patience. scale_pos_weight=11.4
# (below) skews predicted probabilities enough that binary_logloss gets *worse*
# almost every round even while auc keeps climbing, so early stopping fired at
# iteration 1 for every trial (valid AUC 0.729, worse than the LR baseline) until
# this was set -- confirmed by tracing the per-round auc trajectory with
# scale_pos_weight both on and off: unconstrained it climbs smoothly past 0.77.

PARAM_DIST = {
    "num_leaves": [15, 31, 63, 127],
    "learning_rate": [0.03, 0.05, 0.08, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_samples": [10, 20, 30, 50],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [0, 1.0, 1.5, 3.0],
}
N_TRIALS = 25
SEARCH_N_ESTIMATORS = 300
SEARCH_EARLY_STOP = 20

print(f"\nRandom search: {N_TRIALS} candidates, n_estimators capped at "
      f"{SEARCH_N_ESTIMATORS}, early-stopped on the ES fold...")
t0 = time.time()
search_results = []
for i, params in enumerate(ParameterSampler(PARAM_DIST, n_iter=N_TRIALS, random_state=SEED), 1):
    trial = lgb.LGBMClassifier(
        n_estimators=SEARCH_N_ESTIMATORS,
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=monotone,
        subsample_freq=SUBSAMPLE_FREQ,
        random_state=SEED,
        verbosity=-1,
        **params,
    )
    trial.fit(
        X_fit, y_fit,
        eval_set=[(X_es, y_es)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(SEARCH_EARLY_STOP, verbose=False, first_metric_only=True)],
    )
    best_iter = trial.best_iteration_
    best_auc = trial.best_score_["valid_0"]["auc"]
    search_results.append({**params, "best_iteration": best_iter, "es_auc": best_auc})
    print(f"  [{i:>2}/{N_TRIALS}] ES AUC={best_auc:.4f}  iter={best_iter:>3}  {params}")

search_df = pd.DataFrame(search_results).sort_values("es_auc", ascending=False)
best_params = search_df.iloc[0][list(PARAM_DIST)].to_dict()
best_params["num_leaves"] = int(best_params["num_leaves"])
best_params["min_child_samples"] = int(best_params["min_child_samples"])
print(f"\nSearch complete in {time.time() - t0:.0f}s")
print(f"Best candidate (ES AUC={search_df.iloc[0]['es_auc']:.4f}): {best_params}")

with open(ARTIFACT_DIR / "lgbm_hparam_search.json", "w") as f:
    json.dump({"best_params": best_params, "trials": search_results}, f, indent=2, default=float)

# ---------------------------------------------------------------------------
# 3. Train final model with the winning hyperparameters, early stopping
# ---------------------------------------------------------------------------
model = lgb.LGBMClassifier(
    n_estimators=500,
    scale_pos_weight=scale_pos_weight,
    monotone_constraints=monotone,
    subsample_freq=SUBSAMPLE_FREQ,
    random_state=SEED,
    verbosity=-1,
    **best_params,
)

print("\nTraining final model with early stopping (eval on early-stop fold)...")
model.fit(
    X_fit, y_fit,
    eval_set=[(X_es, y_es)],
    eval_metric="auc",
    categorical_feature=cat_cols,
    callbacks=[lgb.early_stopping(30, verbose=False, first_metric_only=True)],
)
print(f"Best iteration: {model.best_iteration_}")

# ---------------------------------------------------------------------------
# 4. Recalibrate on the calibration fold (raw tree-ensemble probabilities
#    are typically miscalibrated at the tails, independent of scale_pos_weight)
# ---------------------------------------------------------------------------
print("\nCalibrating (isotonic) on calib fold...")
calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
calibrated.fit(X_calib, y_calib)

# ---------------------------------------------------------------------------
# 5. Validation — on the fully held-out `valid` split only
# ---------------------------------------------------------------------------
fit_scores_raw = model.predict_proba(X_fit)[:, 1]
valid_pd = np.clip(calibrated.predict_proba(X_valid)[:, 1], PD_FLOOR, 1 - PD_FLOOR)

auc = roc_auc_score(y_valid, valid_pd)
gini = 2 * auc - 1
ks, _ = ks_2samp(valid_pd[y_valid == 1], valid_pd[y_valid == 0])
# Same caveat as 03/04: this dataset has no out-of-time/production sample --
# `fit` and `valid` are two random splits of the SAME application_train.csv
# population, so this PSI is a train-vs-valid split-stability sanity check,
# not a true population-stability test.
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
axes[0].set_title("ROC Curve — LightGBM Challenger", fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

axes[1].hist(valid_pd[y_valid == 0], bins=40, alpha=0.6, label="Non-default", color="#2196F3", density=True)
axes[1].hist(valid_pd[y_valid == 1], bins=40, alpha=0.6, label="Default", color="#F44336", density=True)
axes[1].set_title(f"Score Separation (KS = {ks:.3f})", fontweight="bold")
axes[1].set_xlabel("Predicted PD")
axes[1].legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "29_lgbm_validation.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. SHAP explainability
# ---------------------------------------------------------------------------
print("\nComputing SHAP values on a 5,000-row sample of valid...")
shap_sample = X_valid.sample(n=min(5000, len(X_valid)), random_state=SEED)
explainer = shap.TreeExplainer(model)
shap_values = explainer(shap_sample)

# LightGBM's sklearn wrapper can return per-class SHAP arrays (n, features, 2)
# for binary classification depending on shap/lightgbm version, unlike
# XGBoost's single-array output in 04 -- slice to the positive class if so.
sv = shap_values.values
if sv.ndim == 3:
    sv = sv[:, :, 1]
    shap_values = shap.Explanation(
        values=sv, base_values=shap_values.base_values[:, 1] if np.ndim(shap_values.base_values) > 1
        else shap_values.base_values,
        data=shap_values.data, feature_names=feature_cols,
    )

plt.figure()
shap.summary_plot(shap_values, shap_sample, show=False, max_display=20)
plt.tight_layout()
plt.savefig(FIG_DIR / "30_lgbm_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

mean_abs_shap = pd.Series(
    np.abs(sv).mean(axis=0), index=feature_cols
).sort_values(ascending=False)
print("\nTop 15 features by mean |SHAP|:")
print(mean_abs_shap.head(15).to_string())

top4 = mean_abs_shap.head(4).index.tolist()
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for ax, feat in zip(axes, top4):
    shap.dependence_plot(feat, sv, shap_sample, interaction_index=None, ax=ax, show=False)
    ax.set_title(feat, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "31_lgbm_shap_dependence.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 7. Compare against the baseline and the XGBoost challenger
# ---------------------------------------------------------------------------
baseline_path = MODEL_DIR / "baseline_metrics.json"
gbm_path = MODEL_DIR / "gbm_metrics.json"
if baseline_path.exists() and gbm_path.exists():
    baseline = json.loads(baseline_path.read_text())
    gbm = json.loads(gbm_path.read_text())
    lgbm_current = {"auc": auc, "gini": gini, "ks": ks}
    print("\n" + "=" * 46)
    print("LR (WoE)  vs  XGBoost  vs  LightGBM")
    print("=" * 46)
    print(f"{'Metric':<10}{'LR (WoE)':>12}{'XGBoost':>12}{'LightGBM':>12}")
    for m in ("auc", "gini", "ks"):
        print(f"{m.upper():<10}{baseline[m]:>12.4f}{gbm[m]:>12.4f}{lgbm_current[m]:>12.4f}")
else:
    print("\nRun 03 and 04 first for a full baseline/challenger comparison.")

# ---------------------------------------------------------------------------
# 8. Save artifacts
# ---------------------------------------------------------------------------
with open(MODEL_DIR / "lgbm_lightgbm.pkl", "wb") as f:
    pickle.dump({"model": model, "calibrated": calibrated, "feature_cols": feature_cols,
                 "cat_cols": cat_cols, "monotone_constraints": monotone,
                 "hyperparams": best_params}, f)

metrics = {"auc": auc, "gini": gini, "ks": ks, "psi": psi_score,
           "best_iteration": int(model.best_iteration_), **best_params}
pd.Series(metrics).to_json(MODEL_DIR / "lgbm_metrics.json", indent=2)

print("\n" + "=" * 70)
print("SECOND CHALLENGER COMPLETE")
print("=" * 70)
print(f"Model saved -> {MODEL_DIR / 'lgbm_lightgbm.pkl'}")
print(f"SHAP figures -> {FIG_DIR}")
print(f"Metrics: {metrics}")
