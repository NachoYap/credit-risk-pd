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
  closed" for a quarter of all credit lines. This fix has to be applied to
  **every** downstream comparison against `STATUS_RANK`, not just the max/tail
  aggregates — `IS_DPD = (STATUS_RANK > 0)` silently re-introduces the same bug
  one line down, since `NaN > 0` evaluates to `False`; 85,569 tradelines
  (10.47%) are 100% `"X"` months and were getting `bb_dpd_rate = 0.0` (as
  clean as a confirmed on-time payer) until this was caught on a second,
  more thorough review pass.
- An installment with no `DAYS_ENTRY_PAYMENT`/`AMT_PAYMENT` (never paid at
  all — the worst-case outcome) must not be coded as "on time" just because
  `NaN > 0` is `False`. `aggregate_installments`'s `LATE_FLAG` explicitly
  OR's in `DAYS_ENTRY_PAYMENT.isna()`, and `PAYMENT_SHORTFALL` treats a
  missing `AMT_PAYMENT` as 0 paid, not as a row to silently exclude from the
  shortfall sum while still counting it in the denominator.
- XGBoost's `monotone_constraints` should only be set on features with a
  genuinely monotonic empirical relationship to `TARGET` — verify with a
  decile bad-rate table before adding one, don't assume from intuition.
  `CREDIT_INCOME_RATIO` looked like it should be "higher → riskier" but the
  real relationship is hump-shaped (deciles: 6.84%, 7.80%, 8.09%, 9.16%,
  8.68%, 9.14%, 8.71%, 7.79%, 7.39%, 7.14%), so it's deliberately left
  unconstrained.
- `sc.scorecard()` needs a model whose raw coefficients/intercept represent
  the actual log-odds it's converting to points — it can't take the isotonic
  `calibrated` wrapper (no linear coefficients to build points from), but
  passing the `class_weight='balanced'`-fit `model` directly silently anchors
  "points0 = odds0" at the ~50/50 training-loss prior instead of the true ~8%
  base rate. Correct the intercept analytically before building the card
  (`03:177-187`, King & Zeng 2001 prior correction) rather than either option.
- Isotonic calibration's lowest step can land exactly on `PD = 0.0` (confirmed:
  24 valid rows / 34 Kaggle-test rows for XGBoost). Any log-odds transform or
  `PD × LGD × EAD` calc breaks on an exact zero — clip calibrated output to
  `[PD_FLOOR, 1 - PD_FLOOR]` (`PD_FLOOR = 1e-4`) wherever it's consumed as a
  final PD, not just internally during fitting.
- `sc.woebin_ply()` silently returns `NaN` (no error, no warning) for any
  category value absent from the fitted train-only bins — the WoE-path analog
  of the XGBoost categorical-code bug above, except fail-loud instead of
  fail-silent: a downstream `predict_proba` on WoE features containing NaN
  raises `ValueError`. Currently latent (0 NaN in any committed `*_woe.parquet`
  — Kaggle's test set happens to cover every category train saw) but
  `05_score_and_submit.py`'s `score_logistic()` defensively `fillna(0)`s
  before scoring (0 = population-average log-odds contribution) in case a
  future scoring batch introduces a genuinely new category.
- The PSI reported in `03`/`04` compares two random splits of the *same*
  `application_train.csv` file (`fit` vs `valid`) — there's no out-of-time or
  production sample in this dataset to compare against. It's a train-vs-valid
  split-stability sanity check, not a true population-stability test, and will
  be near-zero by construction. Don't read "PSI: 0.0002" as evidence of
  validated production stability.
- `01_eda_credit_risk.py` isn't covered by `02`/`03`/`04`'s data (it reads the
  raw CSVs itself and isn't imported anywhere), so it had accumulated its own,
  separate bugs: the wrong bureau merge key (`SK_BUREAU_ID` instead of
  `SK_ID_BUREAU`), which silently killed sections 6–12 (bureau/prev/POS/
  installments/credit-card KS, WoE/IV, correlation matrix, top-predictors
  shortlist) with a `KeyError` — confirmed via `notebooks/figures/` jumping
  straight from `15_*` to `25_*`, i.e. `16_bureau_ks.png` through
  `24_top_features_overall.png` had never once been generated; a wrong column
  name in the credit-card section (`AMT_INSTALMENT`, which belongs to
  `installments_payments.csv`, instead of `AMT_INST_MIN_REGULARITY`); a
  missing `no_cores=1`/`check_cate_num=False` on its own separate
  `sc.woebin()` call (the fix below was applied to `02` but not here); its own
  copy of the `"X": -1` bureau-balance conflation (same bug as
  `BUREAU_BALANCE_STATUS_RANK` above); a bar-chart color/data
  misalignment in the correlation plot (`colors_bar` was built from one sort
  order, the bars plotted in a different one); and a Cramer's V denominator
  bug — `np.sqrt(chi2 / (len(app) * (min(ct.shape) - 1)))` used the full
  `application_train` row count as `n`, but `pd.crosstab` drops rows with
  NaN in either column, so `len(app)` overcounts `n` whenever a categorical
  column has missing values, understating every Cramer's V. Fixed to
  `ct.values.sum()`, the actual number of observations in the contingency
  table. All fixed — but the `.ipynb` twin was not, see Pending below.
- `notebooks/KaggledataDownload.py` had two separate bugs: it called
  `kagglehub.dataset_download(..., output_dir=...)`, a kwarg the installed
  `kagglehub==0.3.5` doesn't have (fixed by downloading to kagglehub's own
  cache dir, then `shutil.copytree`-ing into `data/data_raw`); and its
  destination path was `Path.cwd().parent` (cwd-relative) instead of
  repo-relative, so it only worked if run from inside `notebooks/` — run from
  the repo root, like every other script in this project, it silently wrote
  ~2.5GB outside the repo and outside `.gitignore`'s reach. Fixed to
  `Path(__file__).resolve().parent.parent`.

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

Rerun on 2026-08-23 after the second-pass bug fixes (`bb_dpd_rate`'s
incomplete NaN fix, unpaid-installment coding, the `CREDIT_INCOME_RATIO`
monotone constraint, scorecard odds anchoring, PD flooring) — these numbers
reflect the fixed code (`02`→`03`→`04`→`05` all rerun; commit `cdc22ec`).

| Metric | LR (WoE baseline) | XGBoost (challenger) |
|---|---|---|
| AUC  | 0.7651 | 0.7770 |
| Gini | 0.5302 | 0.5540 |
| KS   | 0.3997 | 0.4190 |
| PSI  | 0.0002 | 0.0002 |

Both calibrated (predicted PD: LR 7.98%, XGBoost 8.02%, vs true 8.07%).
XGBoost's top SHAP drivers: `EXT_MEAN` (dominant), `ORGANIZATION_TYPE`,
`inst_late_rate_last6`, `CREDIT_TERM`, `GOODS_CREDIT_RATIO`,
`bureau_debt_credit_ratio`. Kaggle submissions regenerated at
`submissions/submission_xgboost.csv` (mean PD 7.57%) and
`submissions/submission_logistic.csv` (mean PD 8.12%).

## Pending / next steps

- `notebooks/01_eda_credit_risk.ipynb` has independently diverged from the
  `.py` version (e.g. an extra `02b_missing_info_gain.png` cell not in the
  `.py`) and still has the `SK_BUREAU_ID`/`SK_ID_BUREAU` typo fixed in the
  `.py` this session — not reconciled, since `.py` is the documented entry
  point and the notebook is large enough that Read/NotebookEdit tooling
  couldn't load it in one pass.
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
