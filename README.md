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
| `notebooks/05_score_and_submit.py` | Scores the Kaggle test set with a trained model, writes a submission CSV |

Both `03` and `04` depend on `02`'s output but are independent of each other.
`05` depends on either (pass `--model xgboost` or `--model logistic`).

**Two trained, calibrated models** (committed under `models/`), evaluated on
an internal 80/20 train/valid split carved out of `application_train.csv`
(Kaggle's `application_test.csv` has no `TARGET`, so it's feature-engineered
but never used for metrics). **The table below is STALE** — a round of bug
fixes (see `CLAUDE.md`'s "Known gotchas") landed after these numbers were
generated and the pipeline hasn't been rerun since; treat these as
placeholders until `02`→`03`→`04` is rerun:

| Metric | LR (WoE baseline) | XGBoost (challenger) |
|---|---|---|
| AUC  | 0.7649 | 0.7767 |
| Gini | 0.5298 | 0.5534 |
| KS   | 0.3972 | 0.4178 |
| PSI  | 0.0002 | 0.0002 |

Both models are calibrated (predicted PD ≈8.0% vs true 8.07% base rate) via
isotonic regression on a held-out fold — raw `predict_proba` from either
model is not a usable PD without this step. `TARGET` itself is Kaggle's raw
label from `application_train.csv`, used as-is (never derived). XGBoost's top
SHAP drivers: `EXT_MEAN` (dominant), `ORGANIZATION_TYPE`, `CREDIT_TERM`,
`GOODS_CREDIT_RATIO`, `bureau_debt_credit_ratio`, `inst_late_rate_last6`.

**Reusable feature aggregations** live in `src/features.py`, computed once and
merged onto all splits. Every raw auxiliary table contributes surviving
features to both models — none are EDA-only:

| Dataset | Aggregation function | Features in final model |
|---|---|---|
| `application_{train,test}.csv` | `clean_application` | base + engineered (ratios, `EXT_*`, ...) |
| `bureau.csv` | `aggregate_bureau` | 13 |
| `bureau_balance.csv` | `aggregate_bureau_balance` | folded into `bureau_*` stats above, no separate block |
| `previous_application.csv` | `aggregate_previous_application` | 7 |
| `POS_CASH_balance.csv` | `aggregate_pos_cash` | 5 |
| `installments_payments.csv` | `aggregate_installments` | 8 |
| `credit_card_balance.csv` | `aggregate_credit_card` | 9 |

(177 features total, per `models/artifacts/feature_cols.json`.)

## Pending

- **Rerun `02` → `03` → `04` → `05`** to regenerate models/metrics/submissions
  reflecting this session's bug fixes — not yet done (see the metrics table's
  staleness note above and `CLAUDE.md`'s "Known gotchas").
- **`notebooks/01_eda_credit_risk.ipynb`** has independently diverged from the
  `.py` version and still has bugs the `.py` had this session (e.g. a bureau
  merge-key typo) fixed — not reconciled, see `CLAUDE.md`.
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
