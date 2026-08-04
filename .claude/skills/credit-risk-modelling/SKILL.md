---
name: credit-risk-modelling
description: >
  Expert guidance for building credit risk models from scratch in Python, covering
  Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD).
  Use this skill whenever the user mentions PD models, LGD models, EAD models, credit
  scoring, default prediction, logistic regression for credit, survival models, scorecard
  development, weight of evidence (WoE), information value (IV), calibration, Gini
  coefficient, KS statistic, IFRS 9 staging, Basel IRB models, or any task involving
  building, tuning, or evaluating credit risk models in Python. Trigger even for partial
  requests like "help me build a PD model" or "how do I encode WoE features".
---

# Credit Risk Modelling Skill

End-to-end guidance for building PD, LGD, and EAD models in Python using scikit-learn,
statsmodels, and supporting libraries. Follows Basel II/III IRB and IFRS 9 conventions
where relevant.

---

## Workflow Overview

Every model follows this sequence. Jump in at the relevant step based on what the user
has already done.

```
1. Problem Framing      → Define target, observation window, performance window
2. Data Preparation     → Handle missings, outliers, time-based splits
3. Feature Engineering  → WoE/IV for PD; clipping/transformations for LGD/EAD
4. Modelling            → Fit appropriate model class per component
5. Calibration          → Align predicted probabilities to observed default rates
6. Validation           → Discriminatory power, calibration, stability metrics
7. Documentation        → Model card, Gini/KS table, PSI report
```

For details on each step, see the reference files below.

---

## Which Model Are We Building?

Always clarify this first if the user hasn't said.

| Component | Target Variable | Typical Model Class |
|-----------|----------------|---------------------|
| **PD** | Binary: 1 = defaulted within performance window, 0 = not | Logistic regression (scorecard), XGBoost |
| **LGD** | Continuous [0,1]: (EAD − Recovery) / EAD | Beta regression, two-stage (classifier + regressor), XGBoost |
| **EAD** | Continuous ≥ 0: outstanding balance at default | Linear regression on CCF, XGBoost |

Read the relevant reference file before proceeding:
- `references/pd.md` — PD modelling (WoE, scorecard, logistic regression)
- `references/lgd.md` — LGD modelling (two-stage, beta regression)
- `references/ead.md` — EAD / CCF modelling

---

## Universal Rules (apply to all components)

### Train/Test Split
**Always use a time-based split** — never random. Credit data has temporal structure;
random splits cause data leakage.

```python
df = df.sort_values('observation_date')
cutoff = df['observation_date'].quantile(0.8)
train = df[df['observation_date'] < cutoff]
test  = df[df['observation_date'] >= cutoff]
```

### Missing Values
- Treat missing as a separate category for categorical features.
- For numerical: impute with median on train, apply same median to test.
- Document missingness rates — regulators will ask.

```python
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='median')
X_train_num = imp.fit_transform(X_train[num_cols])
X_test_num  = imp.transform(X_test[num_cols])
```

### Class Imbalance (PD / LGD two-stage)
- Default rates are typically 1–10%. Do **not** oversample in credit risk — it distorts
  calibration. Instead use `class_weight='balanced'` or adjust decision thresholds.
- For XGBoost use `scale_pos_weight = n_non_default / n_default`.

---

## Standard Python Stack

```python
# Core
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import statsmodels.api as sm

# Evaluation
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# WoE / IV (for PD scorecards)
# pip install scorecardpy
import scorecardpy as sc
```

---

## Quick Reference: Key Metrics

| Metric | What it measures | Acceptable range |
|--------|-----------------|-----------------|
| Gini coefficient | Discriminatory power | > 0.30 (retail), > 0.25 (wholesale) |
| KS statistic | Max separation between good/bad | > 0.20 |
| AUC-ROC | Discriminatory power | > 0.65 |
| Brier Score | Calibration quality | Lower is better; compare to null model |
| PSI | Population stability | < 0.10 stable, 0.10–0.25 monitor, > 0.25 investigate |

```python
# Gini from AUC
gini = 2 * roc_auc_score(y_true, y_score) - 1

# KS
from scipy.stats import ks_2samp
ks_stat, _ = ks_2samp(y_score[y_true==1], y_score[y_true==0])
```

---

## Calibration (Critical for IRB / IFRS 9)

Raw model scores must be calibrated to long-run average default rates.

```python
from sklearn.calibration import CalibratedClassifierCV

# Platt scaling
calibrated = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
calibrated.fit(X_val, y_val)
probs = calibrated.predict_proba(X_test)[:, 1]

# Visual check
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
plt.plot(mean_pred, frac_pos, marker='o')
plt.plot([0,1],[0,1], '--', label='Perfect calibration')
plt.xlabel('Mean predicted probability'); plt.ylabel('Fraction of positives')
plt.title('Calibration plot'); plt.legend(); plt.show()
```

---

## Output Format

When delivering a model, always provide:
1. **Feature importance / coefficient table** with signs and magnitudes
2. **Gini / KS / AUC on train and test** (flag if gap > 5 points = overfitting)
3. **Calibration plot**
4. **PSI table** if a holdout or OOT (out-of-time) period is available
5. **Model card** summarising assumptions, data period, exclusions

---

## Common Pitfalls to Flag

- **Look-ahead bias**: features computed using information after observation date
- **Through-the-cycle vs point-in-time**: PD for IFRS 9 staging needs PiT; IRB capital
  uses TtC. Ask the user which they need.
- **LGD flooring**: LGD must be ≥ 0 and ≤ 1 (or regulatory floor, e.g. 0.45 for unsecured)
- **EAD ≥ current exposure**: CCF should not produce EAD below drawn balance
- **Segment-level calibration**: a model well-calibrated at portfolio level can be poorly
  calibrated within product/risk grade buckets — always check by segment
