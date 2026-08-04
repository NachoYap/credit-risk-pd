# PD Modelling Reference

## 1. Problem Framing

- **Observation date**: the date at which you observe a borrower's characteristics
- **Performance window**: how far forward you look to determine default (typically 12 months)
- **Default definition**: follow Basel/IFRS 9 — 90 days past due, or unlikeliness to pay

```python
# Example: flag defaults within 12-month window
df['default_flag'] = (
    (df['days_past_due_max_12m'] >= 90) |
    (df['write_off_12m'] == 1)
).astype(int)
```

---

## 2. WoE / IV Analysis

Weight of Evidence (WoE) encodes the predictive power of each bin relative to the
overall default rate. Information Value (IV) summarises the overall feature predictiveness.

```python
import scorecardpy as sc

# Automatic binning with monotonicity constraint
bins = sc.woebin(train, y='default_flag', x=feature_cols)

# IV summary — use to select features
iv_summary = sc.woebin_ply(train, bins)  # transforms to WoE
# Rule of thumb: IV < 0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium, >0.3 strong

# Apply WoE transformation to train and test
train_woe = sc.woebin_ply(train, bins)
test_woe  = sc.woebin_ply(test, bins)
```

---

## 3. Logistic Regression Scorecard

The standard regulatory-preferred approach: interpretable, auditable, stable.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

feature_cols_woe = [c + '_woe' for c in feature_cols]

model = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(
        C=0.1,               # regularisation — tune via CV
        class_weight='balanced',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    ))
])
model.fit(train_woe[feature_cols_woe], train['default_flag'])
```

### Convert to Points-Based Scorecard

```python
# Standard scorecard scaling: Score = Offset + Factor * ln(odds)
# Target: 600 points = odds 1:1 (PDO = 20 points to double odds)
factor = 20 / np.log(2)
offset = 600 - factor * np.log(1)

coefs = model.named_steps['lr'].coef_[0]
intercept = model.named_steps['lr'].intercept_[0]

# Build score table per feature and bin
score_card = sc.scorecard(bins, model.named_steps['lr'],
                           feature_cols_woe, points0=600, odds0=1, pdo=20)
sc.scorecard_ply(test, score_card)  # apply to get scores
```

---

## 4. XGBoost PD Model (when scorecard is not required)

```python
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

default_rate = train['default_flag'].mean()
scale_pos_weight = (1 - default_rate) / default_rate

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=False)  # no shuffle — temporal order
cv_auc = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              early_stopping_rounds=30,
              verbose=False)
```

---

## 5. Validation

```python
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ks_2samp

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_true = test['default_flag']

# Discrimination
auc  = roc_auc_score(y_true, y_pred_proba)
gini = 2 * auc - 1
ks, _ = ks_2samp(y_pred_proba[y_true==1], y_pred_proba[y_true==0])

print(f"AUC: {auc:.4f} | Gini: {gini:.4f} | KS: {ks:.4f}")

# Stability (PSI) — compare train score distribution to test
def psi(expected, actual, buckets=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets+1))
    expected_pcts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pcts   = np.histogram(actual,   bins=breakpoints)[0] / len(actual)
    psi_val = np.sum((actual_pcts - expected_pcts) * np.log(
        (actual_pcts + 1e-8) / (expected_pcts + 1e-8)))
    return psi_val

psi_score = psi(train_scores, test_scores)
print(f"PSI: {psi_score:.4f}")
```

---

## 6. Rating Grade Mapping

After calibration, map continuous PD to discrete rating grades (e.g. 1–10 or AAA–CCC).

```python
# Quantile-based grade assignment
grade_cuts = np.percentile(train_pd, np.linspace(0, 100, 11))
train['pd_grade'] = pd.cut(train_pd, bins=grade_cuts, labels=range(1, 11))

# Validate: each grade should have monotonically increasing observed default rate
grade_dr = train.groupby('pd_grade')['default_flag'].mean()
print(grade_dr)  # should be monotonically increasing
```
