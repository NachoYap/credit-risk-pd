"""
Credit Risk PD Model — Baseline: Logistic Regression Scorecard
==================================================================
WoE-encoded logistic regression — the regulatory-preferred baseline.
Monotonic by construction (WoE bins are monotonically binned), naturally
handles missing values as their own bin, directly convertible to a
points-based scorecard.

Methodology:
    1. train_woe is split fit/calib (80/20) — the model is fit on `fit`,
       then recalibrated on `calib`. class_weight='balanced' shifts the
       training distribution to ~50/50, so raw predict_proba is NOT a
       true PD until corrected back to the true ~8% base rate.
    2. `valid` (the holdout carved out in 02, untouched until now) is used
       only for final metrics — never for fitting or calibration.

Run with:
    python notebooks/03_baseline_logistic_regression.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ks_2samp
import scorecardpy as sc

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


def psi(expected, actual, buckets=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_pcts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pcts = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    return np.sum(
        (actual_pcts - expected_pcts) * np.log((actual_pcts + 1e-8) / (expected_pcts + 1e-8))
    )


print("=" * 70)
print("BASELINE — LOGISTIC REGRESSION ON WOE FEATURES")
print("=" * 70)

print("\nLoading WoE-encoded features...")
train = pd.read_parquet(DATA_DIR / "train_woe.parquet")
valid = pd.read_parquet(DATA_DIR / "valid_woe.parquet")
feature_cols = [c for c in train.columns if c.endswith("_woe")]
print(f"  train: {train.shape}  valid: {valid.shape}  features: {len(feature_cols)}")

X_train, y_train = train[feature_cols], train[TARGET]
X_valid, y_valid = valid[feature_cols], valid[TARGET]

# ---------------------------------------------------------------------------
# 1. Fit / calibration split (train is never used raw for both fitting AND
#    calibration — that would let the calibration curve overfit the same
#    noise the model already fit on)
# ---------------------------------------------------------------------------
X_fit, X_calib, y_fit, y_calib = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED
)
print(f"\nFit: {X_fit.shape}  Calib: {X_calib.shape}")

# ---------------------------------------------------------------------------
# 2. Regularisation tuning — class_weight='balanced' offsets the ~8% class
#    imbalance during fitting; WoE features are already on a comparable
#    log-odds contribution scale, so no StandardScaler (it would break the
#    direct coefficient -> scorecard-points correspondence used in step 5).
# ---------------------------------------------------------------------------
print("\nTuning C via 5-fold CV (roc_auc)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
param_grid = {"C": [0.01, 0.03, 0.1, 0.3, 1, 3]}
gs = GridSearchCV(
    LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=SEED),
    param_grid, scoring="roc_auc", cv=cv, n_jobs=-1,
)
gs.fit(X_fit, y_fit)
model = gs.best_estimator_
print(f"  best C: {gs.best_params_['C']}  CV AUC: {gs.best_score_:.4f}")

# ---------------------------------------------------------------------------
# 3. Recalibrate back to true population odds on the held-out calib fold
# ---------------------------------------------------------------------------
print("\nCalibrating (isotonic) on calib fold...")
calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
calibrated.fit(X_calib, y_calib)

# ---------------------------------------------------------------------------
# 4. Validation — on the fully held-out `valid` split only
# ---------------------------------------------------------------------------
train_scores_raw = model.predict_proba(X_fit)[:, 1]
valid_pd = calibrated.predict_proba(X_valid)[:, 1]

auc = roc_auc_score(y_valid, valid_pd)
gini = 2 * auc - 1
ks, _ = ks_2samp(valid_pd[y_valid == 1], valid_pd[y_valid == 0])
psi_score = psi(train_scores_raw, model.predict_proba(X_valid)[:, 1])

print("\n" + "-" * 50)
print(f"VALIDATION  |  AUC: {auc:.4f}  Gini: {gini:.4f}  KS: {ks:.4f}  PSI: {psi_score:.4f}")
print(f"Mean predicted PD (valid): {valid_pd.mean():.2%}  "
      f"True default rate (valid): {y_valid.mean():.2%}")
print("-" * 50)

fpr, tpr, _ = roc_curve(y_valid, valid_pd)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].plot(fpr, tpr, color="#F44336", label=f"AUC = {auc:.3f}")
axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[0].set_title("ROC Curve — Baseline LR", fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

axes[1].hist(valid_pd[y_valid == 0], bins=40, alpha=0.6, label="Non-default", color="#2196F3", density=True)
axes[1].hist(valid_pd[y_valid == 1], bins=40, alpha=0.6, label="Default", color="#F44336", density=True)
axes[1].set_title(f"Score Separation (KS = {ks:.3f})", fontweight="bold")
axes[1].set_xlabel("Predicted PD")
axes[1].legend()

grade_cuts = np.unique(np.percentile(valid_pd, np.linspace(0, 100, 11)))
valid_grade = pd.cut(valid_pd, bins=grade_cuts, include_lowest=True, labels=False)
grade_dr = pd.Series(y_valid.values).groupby(valid_grade).mean()
axes[2].bar(grade_dr.index.astype(str), grade_dr.values, color="#4CAF50")
axes[2].set_title("Observed Default Rate by PD Decile", fontweight="bold")
axes[2].set_xlabel("PD Decile (0=lowest risk)")
axes[2].set_ylabel("Observed Default Rate")

plt.tight_layout()
plt.savefig(FIG_DIR / "25_baseline_lr_validation.png", dpi=150)
plt.close()

is_monotonic = grade_dr.is_monotonic_increasing
print(f"\nDefault rate by decile monotonically increasing: {is_monotonic}")
print(grade_dr.to_string())

# ---------------------------------------------------------------------------
# 5. Points-based scorecard
# ---------------------------------------------------------------------------
print("\nBuilding points-based scorecard (600 pts = 1:1 odds, PDO=20)...")
with open(ARTIFACT_DIR / "woe_bins.pkl", "rb") as f:
    bins = pickle.load(f)

card = sc.scorecard(bins, model, feature_cols, points0=600, odds0=1, pdo=20)

train_raw = pd.read_parquet(DATA_DIR / "train_features.parquet")
valid_raw = pd.read_parquet(DATA_DIR / "valid_features.parquet")
valid_scores = sc.scorecard_ply(valid_raw, card, print_step=0)
print("\nScore distribution (valid):")
print(valid_scores["score"].describe().to_string())

# ---------------------------------------------------------------------------
# 6. Save artifacts
# ---------------------------------------------------------------------------
with open(MODEL_DIR / "baseline_logistic_woe.pkl", "wb") as f:
    pickle.dump({"model": model, "calibrated": calibrated, "feature_cols": feature_cols}, f)
with open(MODEL_DIR / "baseline_scorecard.pkl", "wb") as f:
    pickle.dump(card, f)

metrics = {"auc": auc, "gini": gini, "ks": ks, "psi": psi_score, "best_C": gs.best_params_["C"]}
pd.Series(metrics).to_json(MODEL_DIR / "baseline_metrics.json", indent=2)

print("\n" + "=" * 70)
print("BASELINE COMPLETE")
print("=" * 70)
print(f"Model saved -> {MODEL_DIR / 'baseline_logistic_woe.pkl'}")
print(f"Scorecard saved -> {MODEL_DIR / 'baseline_scorecard.pkl'}")
print(f"Metrics: {metrics}")
