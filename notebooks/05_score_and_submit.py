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

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "data_processed"
MODEL_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "submissions"

ID_COL = "SK_ID_CURR"
TARGET = "TARGET"


def score_xgboost() -> pd.DataFrame:
    test = pd.read_parquet(DATA_DIR / "test_features.parquet")
    with open(MODEL_DIR / "gbm_xgboost.pkl", "rb") as f:
        artifact = pickle.load(f)
    feature_cols = artifact["feature_cols"]
    pd_scores = artifact["calibrated"].predict_proba(test[feature_cols])[:, 1]
    return pd.DataFrame({ID_COL: test[ID_COL], TARGET: pd_scores})


def score_logistic() -> pd.DataFrame:
    test = pd.read_parquet(DATA_DIR / "test_woe.parquet")
    with open(MODEL_DIR / "baseline_logistic_woe.pkl", "rb") as f:
        artifact = pickle.load(f)
    feature_cols = artifact["feature_cols"]
    pd_scores = artifact["calibrated"].predict_proba(test[feature_cols])[:, 1]
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
