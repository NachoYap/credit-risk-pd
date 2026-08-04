# EAD Modelling Reference

## 1. Problem Framing

- **EAD** = estimated outstanding balance at the time of default
- For **term loans**: EAD ≈ current outstanding (little uncertainty)
- For **revolving facilities** (credit cards, overdrafts, credit lines): EAD depends on
  how much the borrower draws down before defaulting — this is uncertain and must be modelled
- **CCF (Credit Conversion Factor)**: the key intermediate quantity for revolvers

```
EAD = Current Drawn Balance + CCF × Undrawn Commitment
CCF ∈ [0, 1]: 0 = no further draw-down, 1 = full draw-down of limit
```

```python
# Compute observed CCF from defaulted accounts
df['ccf'] = (df['ead_at_default'] - df['drawn_at_observation']) / \
             df['undrawn_at_observation'].replace(0, np.nan)
df['ccf'] = df['ccf'].clip(0, 1)
# Drop rows where undrawn = 0 (no commitment to convert)
df_ccf = df.dropna(subset=['ccf'])
```

---

## 2. CCF Distribution

Like LGD, CCF is bounded [0,1] and often bimodal. Plot before modelling.

```python
import matplotlib.pyplot as plt
plt.hist(df_ccf['ccf'], bins=50, edgecolor='k')
plt.xlabel('CCF'); plt.ylabel('Count')
plt.title('CCF Distribution (Revolving Facilities)')
plt.show()
```

---

## 3. OLS / Ridge Regression on CCF

Simple and interpretable. Use as a baseline.

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ead_model = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])
ead_model.fit(X_train, train['ccf'])

ccf_pred = ead_model.predict(X_test)
ccf_pred = np.clip(ccf_pred, 0, 1)  # enforce [0,1]

# Compute final EAD
ead_pred = (test['drawn_at_observation'] +
            ccf_pred * test['undrawn_at_observation'])
```

---

## 4. XGBoost CCF Model

```python
import xgboost as xgb

xgb_ead = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_ead.fit(X_train, train['ccf'],
            eval_set=[(X_test, test['ccf'])],
            early_stopping_rounds=30,
            verbose=False)

ccf_pred = xgb_ead.predict(X_test).clip(0, 1)
ead_pred = test['drawn_at_observation'] + ccf_pred * test['undrawn_at_observation']
```

---

## 5. Validation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Validate at CCF level
mae_ccf  = mean_absolute_error(test['ccf'], ccf_pred)
rmse_ccf = mean_squared_error(test['ccf'], ccf_pred, squared=False)
print(f"CCF MAE: {mae_ccf:.4f} | CCF RMSE: {rmse_ccf:.4f}")

# Validate at EAD level (most important for business)
mae_ead  = mean_absolute_error(test['ead_at_default'], ead_pred)
rmse_ead = mean_squared_error(test['ead_at_default'], ead_pred, squared=False)
print(f"EAD MAE: {mae_ead:.0f} | EAD RMSE: {rmse_ead:.0f}")

# Calibration: EAD should not be systematically below observed
bias = (ead_pred - test['ead_at_default']).mean()
print(f"EAD Bias (positive = conservative): {bias:.0f}")
```

---

## 6. Regulatory Constraints

- EAD must be **≥ current drawn balance** (cannot be less than what is already outstanding)
- For Basel IRB, EAD for off-balance sheet items uses **regulatory CCF floors** defined
  by commitment type (e.g. unconditionally cancellable = 0%, other commitments = 20–75%)
- Ask the user whether they need a Basel regulatory CCF or a model-estimated CCF

```python
# Enforce EAD floor = current drawn
ead_pred_floored = np.maximum(ead_pred, test['drawn_at_observation'])
```

---

## 7. Term Loans (simple case)

For fully drawn term loans, EAD = amortised balance at default. Model the remaining
balance using the contractual amortisation schedule.

```python
# Amortised balance at time t
def amortised_balance(principal, annual_rate, term_months, t):
    r = annual_rate / 12
    balance = principal * ((1+r)**term_months - (1+r)**t) / ((1+r)**term_months - 1)
    return balance
```
