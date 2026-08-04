# LGD Modelling Reference

## 1. Problem Framing

- **LGD** = (EAD − Recovery) / EAD, bounded [0, 1]
- Observed only on **defaulted accounts** — you need a default cohort
- **Workout LGD**: wait until recovery process is complete (can be 3–7 years)
- **Market LGD**: use market prices of distressed debt (less common for retail)

```python
# Compute realised LGD
df['lgd'] = (df['ead_at_default'] - df['total_recovery']) / df['ead_at_default']
df['lgd'] = df['lgd'].clip(lower=0, upper=1)  # enforce [0,1] bounds
```

---

## 2. LGD Distribution

LGD is bimodal: many accounts recover fully (LGD ≈ 0) or recover nothing (LGD ≈ 1).
Always plot the distribution first.

```python
import matplotlib.pyplot as plt
plt.hist(df['lgd'], bins=50, edgecolor='k')
plt.xlabel('LGD'); plt.ylabel('Count')
plt.title('LGD Distribution')
plt.show()
```

---

## 3. Two-Stage Model (recommended)

**Stage 1**: Predict whether LGD = 0 (full recovery) — binary classifier
**Stage 2**: Predict LGD conditional on partial/total loss — regressor on (0, 1]

```python
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor

# Stage 1: full-recovery flag
df['full_recovery'] = (df['lgd'] == 0).astype(int)
clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500)
clf.fit(X_train, train['full_recovery'])
p_full_recovery = clf.predict_proba(X_test)[:, 1]

# Stage 2: LGD among partial/total losses
mask_train = train['full_recovery'] == 0
mask_test  = test['full_recovery']  == 0

reg = GradientBoostingRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, random_state=42
)
reg.fit(X_train[mask_train], train.loc[mask_train, 'lgd'])
lgd_partial = reg.predict(X_test[mask_test])
lgd_partial = np.clip(lgd_partial, 0, 1)

# Combine
lgd_pred = np.zeros(len(X_test))
lgd_pred[~mask_test] = (1 - p_full_recovery[~mask_test]) * lgd_partial
```

---

## 4. Beta Regression (alternative)

Beta regression is well-suited for (0, 1) bounded targets with a natural S-shape.

```python
# statsmodels has BetaModel in formula API
import statsmodels.formula.api as smf

# Requires LGD strictly in (0, 1) — apply small adjustment for boundary values
df['lgd_adj'] = df['lgd'].clip(0.001, 0.999)

formula = 'lgd_adj ~ feature1 + feature2 + feature3'
beta_model = smf.beta(formula=formula, data=train).fit()
print(beta_model.summary())

lgd_pred = beta_model.predict(test)
```

---

## 5. Validation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae  = mean_absolute_error(test['lgd'], lgd_pred)
rmse = mean_squared_error(test['lgd'], lgd_pred, squared=False)
print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f}")

# Calibration by decile
test['lgd_pred'] = lgd_pred
test['decile'] = pd.qcut(lgd_pred, 10, labels=False)
calib = test.groupby('decile')[['lgd', 'lgd_pred']].mean()
print(calib)

# Plot
import matplotlib.pyplot as plt
plt.scatter(calib['lgd_pred'], calib['lgd'], marker='o')
plt.plot([0,1],[0,1],'--', label='Perfect calibration')
plt.xlabel('Predicted LGD'); plt.ylabel('Observed LGD')
plt.title('LGD Calibration by Decile')
plt.legend(); plt.show()
```

---

## 6. Regulatory Floors

Under Basel IRB:
- Unsecured retail: LGD floor = **45%**
- Secured (residential mortgage): LGD floor = **10–20%** depending on jurisdiction
- Subordinated debt: LGD floor = **75%**

```python
# Apply regulatory floor
lgd_regulatory = np.maximum(lgd_pred, 0.45)  # example: unsecured
```

Always confirm with the user which jurisdiction and product type applies.
