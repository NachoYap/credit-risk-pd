"""
Credit Risk PD Model — Inference: score the Kaggle test set and write a submission
====================================================================================
Loads the already feature-engineered Kaggle test set (built by 02 from
application_test.csv, which has no TARGET and was never used for metrics) and
a trained + calibrated model, scores every applicant's PD, and writes a
submission CSV in the SK_ID_CURR,TARGET format Kaggle expects.

Scoring mirrors exactly how 03/04 score their `valid` split — no new
calibration logic here, just predict_proba on the calibrated model.

Run with:
    python notebooks/05_score_and_submit.py --model xgboost
    python notebooks/05_score_and_submit.py --model logistic
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "data_processed"
MODEL_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "submissions"

ID_COL = "SK_ID_CURR"
TARGET = "TARGET"
# Isotonic calibration's lowest step can land exactly on 0.0 -- log(PD/(1-PD))
# and PD x LGD x EAD both break on an exact 0, so floor the calibrated output
# away from the boundary.
PD_FLOOR = 1e-4


def score_xgboost() -> pd.DataFrame:
    test = pd.read_parquet(DATA_DIR / "test_features.parquet")
    with open(MODEL_DIR / "gbm_xgboost.pkl", "rb") as f:
        artifact = pickle.load(f)
    feature_cols = artifact["feature_cols"]
    pd_scores = artifact["calibrated"].predict_proba(test[feature_cols])[:, 1]
    pd_scores = np.clip(pd_scores, PD_FLOOR, 1 - PD_FLOOR)
    return pd.DataFrame({ID_COL: test[ID_COL], TARGET: pd_scores})


def score_logistic() -> pd.DataFrame:
    test = pd.read_parquet(DATA_DIR / "test_woe.parquet")
    with open(MODEL_DIR / "baseline_logistic_woe.pkl", "rb") as f:
        artifact = pickle.load(f)
    feature_cols = artifact["feature_cols"]
    # sc.woebin_ply() silently returns NaN for any category value it never
    # saw while fitting bins on train (confirmed: currently 0 NaN in this
    # repo's own test_woe.parquet, but a future batch with a genuinely new
    # category would otherwise crash predict_proba with "Input X contains
    # NaN"). 0 = population-average log-odds contribution in WoE encoding, a
    # safe default for a code path we can't currently exercise against real data.
    X = test[feature_cols].fillna(0)
    pd_scores = artifact["calibrated"].predict_proba(X)[:, 1]
    pd_scores = np.clip(pd_scores, PD_FLOOR, 1 - PD_FLOOR)
    return pd.DataFrame({ID_COL: test[ID_COL], TARGET: pd_scores})


SCORERS = {"xgboost": score_xgboost, "logistic": score_logistic}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=sorted(SCORERS), default="xgboost",
        help="Which trained model to score with (default: xgboost, the better-performing challenger)",
    )
    args = parser.parse_args()

    print(f"Scoring Kaggle test set with: {args.model}")
    submission = SCORERS[args.model]()

    print(f"  rows: {len(submission)}  mean predicted PD: {submission[TARGET].mean():.2%}")

    SUBMISSION_DIR.mkdir(exist_ok=True)
    out_path = SUBMISSION_DIR / f"submission_{args.model}.csv"
    submission.to_csv(out_path, index=False)
    print(f"Submission written -> {out_path}")


if __name__ == "__main__":
    main()
