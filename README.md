# Credit Risk PD Model

Probability of Default (PD) model built on the Kaggle [Home Credit Default
Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset. Currently
**PD-only** — LGD and EAD models are not started.

## What's done

**Pipeline** (run in order: `01` → `02` → `03` and/or `04`):

| Notebook | Purpose |
|---|---|
| `notebooks/01_eda_credit_risk.py`/`.ipynb` | EDA — target imbalance, missingness, per-table KS/IV, top predictors |
| `notebooks/02_feature_engineering.py` | Builds the modelling table, fits WoE bins + outlier caps (train only) |
| `notebooks/03_baseline_logistic_regression.py` | WoE + logistic regression scorecard (baseline) |
| `notebooks/04_gradient_boosting_shap.py` | XGBoost + SHAP (challenger) |

Both `03` and `04` depend on `02`'s output but are independent of each other.

**Two trained, calibrated models** (committed under `models/`), evaluated on
an internal 80/20 train/valid split carved out of `application_train.csv`
(Kaggle's `application_test.csv` has no `TARGET`, so it's feature-engineered
but never used for metrics):

| Metric | LR (WoE baseline) | XGBoost (challenger) |
|---|---|---|
| AUC  | 0.7650 | 0.7761 |
| Gini | 0.5300 | 0.5523 |
| KS   | 0.3976 | 0.4179 |
| PSI  | 0.0002 | 0.0001 |

Both models are calibrated (predicted PD ≈8.0% vs true 8.07% base rate) via
isotonic regression on a held-out fold — raw `predict_proba` from either
model is not a usable PD without this step. XGBoost's top SHAP drivers:
`EXT_MEAN` (dominant), `ORGANIZATION_TYPE`, `CREDIT_TERM`,
`GOODS_CREDIT_RATIO`, `bureau_debt_credit_ratio`, `inst_late_rate_last6`.

**Reusable feature aggregations** for the bureau, previous_application,
POS_CASH, installments, and credit_card auxiliary tables live in
`src/features.py`, computed once and merged onto all splits.

## Pending

- **No inference/scoring script** — `test_features.parquet` / `test_woe.parquet`
  are already built but nothing produces a Kaggle submission file from them.
- **No hyperparameter search for XGBoost** — currently fixed defaults
  (depth=4, lr=0.05), only `n_estimators` is tuned via early stopping.
  LightGBM is installed but untried as a second challenger.
- **LGD and EAD models not started** — skill references exist at
  `.claude/skills/credit-risk-modelling/references/{lgd,ead}.md`.
- **`app/` and `tests/` are empty** — no serving layer or unit tests exist,
  including for `src/features.py`.

## Environment

`requirements.txt` is accurate but not yet enforced via a venv. `numpy` is
pinned to `2.1.3` (downgraded from 2.2.6) because `shap` (via `numba`)
doesn't support numpy 2.2+.

## Data

`data/data_raw/` (raw Kaggle CSVs, ~2.5GB) and `data/data_processed/`
(engineered train/valid/test parquet) are gitignored. Raw data is not
reproducible from code — download it from Kaggle. Processed data
regenerates by running `02_feature_engineering.py`.

See `CLAUDE.md` for detailed gotchas, methodology notes, and Claude Code
guidance for this repo.
