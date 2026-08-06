# credit-risk-pd

Credit risk PD (Probability of Default) model on the Kaggle Home Credit
Default Risk dataset. Currently PD-only — LGD/EAD are not started.

## Structure

```
notebooks/
  01_eda_credit_risk.py/.ipynb   EDA — target imbalance, missingness, per-table KS/IV, top predictors
  02_feature_engineering.py      Builds the modelling table, fits WoE bins + outlier caps (train only)
  03_baseline_logistic_regression.py   WoE + LR scorecard (baseline)
  04_gradient_boosting_shap.py         XGBoost + SHAP (challenger)
  05_score_and_submit.py               Scores test_features/test_woe with a trained model, writes Kaggle submission CSV
  figures/                       All output plots, numbered sequentially across scripts (next free: 29)
submissions/                    Kaggle submission CSVs from 05 — gitignored, regenerate via 05
src/features.py                  Reusable aggregation functions (bureau, prev_app, POS, installments, cc)
data/data_raw/                   Raw Kaggle CSVs — gitignored, not reproducible from code, ~2.5GB
data/data_processed/             Engineered train/valid/test parquet — gitignored, regenerate via 02
models/                          Saved models + metrics (committed — small)
models/artifacts/                Fitted WoE bins, outlier caps, feature column lists (committed)
.claude/skills/credit-risk-modelling/   PD/LGD/EAD methodology reference skill
```

Run order: `01` → `02` → `03` and/or `04` (03/04 both depend on 02's output, independent of each other).

## Environment

No project-local venv — packages are installed in the **global** Python 3.12
interpreter, which other projects on this machine also use. `requirements.txt`
is accurate but not yet enforced via a venv; consider creating one before
further dependency changes so this repo stops touching unrelated projects.
`numpy` is pinned to `2.1.3` (downgraded from 2.2.6) because `shap` (via
`numba`) doesn't support numpy 2.2+.

## Known gotchas (hit and fixed this session)

- `scorecardpy.woebin` defaults to multiprocessing across all CPU cores once
  x-columns ≥ 10. On Windows, without a `__main__` guard, that can re-execute
  the whole calling script inside each worker. Always pass `no_cores=1`.
- `scorecardpy.woebin` also prompts interactively (`input()`) when a
  categorical column has many unique values (e.g. `ORGANIZATION_TYPE`, 58
  categories) — hangs/crashes under non-interactive execution. Pass
  `check_cate_num=False`.
- `shap.dependence_plot`'s automatic interaction-partner search breaks when
  scanning across categorical (string) columns mixed with numeric ones. Pass
  `interaction_index=None` explicitly.
- `02_feature_engineering.py` takes ~25–30 min end to end — `scorecardpy.var_filter`
  alone is ~20 min over 246k rows × 178 columns. This is the library being slow,
  not a bug; run it in the background.
- Casting categorical columns to pandas `category` dtype **per split**
  (`train_feat.astype("category")`, `valid_feat.astype("category")`, ... done
  independently) lets pandas infer a different category→code mapping per split
  whenever the set of unique values differs (e.g. a rare category present in
  train but absent from valid/test) — XGBoost's `enable_categorical=True` path
  splits on those integer codes, so a per-split mismatch silently corrupts
  predictions for every row whose category's code shifted (confirmed: up to
  0.087 absolute PD shift for individual applicants). Fit categories on train
  only, then apply that exact category set to valid/test via
  `pd.Categorical(col, categories=train_categories)` — unseen categories become
  NaN, which XGBoost handles natively.
- `BUREAU_BALANCE_STATUS_RANK` must not map `"X"` (status unknown that month)
  to the same rank as `"C"` (confirmed closed, clean) — they're opposite ends
  of information quality, and 24.8% of bureau tradelines report `"X"` as their
  most recent status. Leave `"X"` unmapped (→ NaN) so `bb_recent_status_rank`/
  `bb_worst_status_rank` correctly signal "unknown" rather than "as safe as
  closed" for a quarter of all credit lines.

## Methodology notes

- `TARGET` is read as-is from `application_train.csv` (`02:93`) — it is Kaggle's
  raw label (0 = repaid, 1 = default), never derived or constructed from other
  columns anywhere in the pipeline.
- Kaggle's `application_test.csv` has no `TARGET`, so it can't be used for
  validation. `02` carves an internal labelled 80/20 train/valid split out of
  `application_train.csv` instead; `application_test.csv` (48,744 rows) is still
  feature-engineered into `test_features.parquet`/`test_woe.parquet` (for Kaggle
  submission via `05`) but never used for metrics — all reported AUC/Gini/KS/PSI
  come from the internal `valid` split only. Don't confuse pipeline "test" (Kaggle's
  unlabeled set) with "valid" (the labelled holdout used for metrics).
- Auxiliary-table aggregations (bureau, bureau_balance, previous_application,
  POS_CASH_balance, installments_payments, credit_card_balance) are computed
  **once** and merged onto train/valid/test — do not call `build_feature_matrix`
  separately per split, it re-aggregates ~58M rows each time. All 6 raw tables
  contribute surviving features to both models (177 total, per
  `models/artifacts/feature_cols.json`): bureau 13 (bureau_balance's stats are
  folded in under the `bureau_` prefix — there's no separate `bb_*` block),
  previous_application 7, POS_CASH 5, installments 8, credit_card 9, plus the
  application-table base/engineered columns. None of the 6 tables are EDA-only —
  e.g. `bureau_debt_credit_ratio` (bureau) and `cc_avg_utilization`/`cc_utilization_trend`
  (credit_card) all reach the trained models.
- Outlier caps and WoE bins are fit on the train split only, then applied to
  valid/test — this is deliberate, don't "simplify" by fitting on the full data.
- Both models use `class_weight='balanced'` / `scale_pos_weight` during fitting,
  then isotonic recalibration on a held-out calibration fold to correct
  predicted PD back to the true ~8% base rate. Raw `predict_proba` from either
  model is not a usable PD without this step.

## Current results (internal valid split)

| Metric | LR (WoE baseline) | XGBoost (challenger) |
|---|---|---|
| AUC  | 0.7649 | 0.7767 |
| Gini | 0.5298 | 0.5534 |
| KS   | 0.3972 | 0.4178 |
| PSI  | 0.0002 | 0.0002 |

Both calibrated (predicted PD ≈8.0% vs true 8.07%). XGBoost's top SHAP drivers:
`EXT_MEAN` (dominant), `ORGANIZATION_TYPE`, `CREDIT_TERM`, `GOODS_CREDIT_RATIO`,
`bureau_debt_credit_ratio`, `inst_late_rate_last6`.

## Pending / next steps

- No hyperparameter search for XGBoost — currently fixed defaults
  (depth=4, lr=0.05) matching the skill's reference values, only `n_estimators`
  is tuned via early stopping. LightGBM is installed but untried as a second
  challenger.
- LGD and EAD models not started (skill references exist at
  `.claude/skills/credit-risk-modelling/references/{lgd,ead}.md`).
- `app/`, `tests/` are empty scaffolding — no serving layer or unit tests exist,
  including for `src/features.py`.
- `Dockerfile` is empty.
- No project-local venv (see Environment above).
