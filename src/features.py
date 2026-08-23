"""Feature engineering for the Home Credit Default Risk PD model — fit-on-train, apply-to-test."""

import numpy as np
import pandas as pd

EXT_SOURCE_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

# "X" (status unknown that month) is deliberately left unmapped (-> NaN), not
# folded into "C" (confirmed closed, clean) — the two are opposite ends of
# information quality, and 24.8% of bureau tradelines report "X" as their most
# recent status, so conflating them would bias bb_recent_status_rank optimistic
# for a quarter of all credit lines.
BUREAU_BALANCE_STATUS_RANK = {"C": -1, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}


def clean_application(app: pd.DataFrame) -> pd.DataFrame:
    df = app.copy()

    df["FLAG_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED_CLEAN"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED_CLEAN"] / 365).clip(lower=0)

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["CREDIT_TERM"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"].replace(0, np.nan)
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)

    df["EXT_MEAN"] = df[EXT_SOURCE_COLS].mean(axis=1)
    df["EXT_MIN"] = df[EXT_SOURCE_COLS].min(axis=1)
    df["EXT_MAX"] = df[EXT_SOURCE_COLS].max(axis=1)
    df["EXT_STD"] = df[EXT_SOURCE_COLS].std(axis=1)
    df["EXT_SOURCE_MISSING_COUNT"] = df[EXT_SOURCE_COLS].isna().sum(axis=1)

    return df


def aggregate_bureau_balance(bb: pd.DataFrame) -> pd.DataFrame:
    bb = bb.copy()
    bb["STATUS_RANK"] = bb["STATUS"].map(BUREAU_BALANCE_STATUS_RANK)
    # Keep "X" (unknown status) as NaN here too, not 0 -- (NaN > 0) silently
    # evaluates to False, which would conflate "unknown" with "confirmed not
    # delinquent" in bb_dpd_rate, the same conflation already fixed for the
    # rank columns below.
    bb["IS_DPD"] = (bb["STATUS_RANK"] > 0).astype(float)
    bb.loc[bb["STATUS_RANK"].isna(), "IS_DPD"] = np.nan

    agg = bb.groupby("SK_ID_BUREAU").agg(
        bb_months_count=("MONTHS_BALANCE", "count"),
        bb_dpd_rate=("IS_DPD", "mean"),
        bb_worst_status_rank=("STATUS_RANK", "max"),
    ).reset_index()

    recent = (
        bb.sort_values(["SK_ID_BUREAU", "MONTHS_BALANCE"])
        .groupby("SK_ID_BUREAU")
        .tail(1)[["SK_ID_BUREAU", "STATUS_RANK"]]
        .rename(columns={"STATUS_RANK": "bb_recent_status_rank"})
    )
    return agg.merge(recent, on="SK_ID_BUREAU", how="left")


def aggregate_bureau(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    bb_agg = aggregate_bureau_balance(bureau_balance)
    b = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    b["IS_ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    b["HAS_OVERDUE"] = (b["CREDIT_DAY_OVERDUE"] > 0).astype(int)

    agg = b.groupby("SK_ID_CURR").agg(
        bureau_n_credits=("SK_ID_BUREAU", "count"),
        bureau_active_ratio=("IS_ACTIVE", "mean"),
        bureau_max_overdue=("AMT_CREDIT_MAX_OVERDUE", "max"),
        bureau_sum_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_sum_credit=("AMT_CREDIT_SUM", "sum"),
        bureau_avg_days_credit=("DAYS_CREDIT", "mean"),
        bureau_overdue_current=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        bureau_max_dpd=("CREDIT_DAY_OVERDUE", "max"),
        bureau_n_dpd_credits=("HAS_OVERDUE", "sum"),
        bureau_dpd_rate=("bb_dpd_rate", "mean"),
        bureau_worst_status_rank=("bb_worst_status_rank", "max"),
        bureau_recent_status_rank=("bb_recent_status_rank", "mean"),
    ).reset_index()

    agg["bureau_debt_credit_ratio"] = agg["bureau_sum_debt"] / agg["bureau_sum_credit"].replace(0, np.nan)
    return agg


def aggregate_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    prev = prev.copy()
    prev["DAYS_DECISION_ABS"] = prev["DAYS_DECISION"].abs()
    prev["IS_REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    prev["IS_APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)

    agg = prev.groupby("SK_ID_CURR").agg(
        prev_n_applications=("SK_ID_PREV", "count"),
        prev_avg_credit=("AMT_CREDIT", "mean"),
        prev_max_days_decision=("DAYS_DECISION_ABS", "max"),
        prev_avg_days_decision=("DAYS_DECISION_ABS", "mean"),
        prev_days_since_last_application=("DAYS_DECISION_ABS", "min"),
        prev_refusal_rate=("IS_REFUSED", "mean"),
        prev_approval_rate=("IS_APPROVED", "mean"),
    ).reset_index()
    return agg


def aggregate_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    pos = pos.copy()
    pos["IS_DPD"] = (pos["SK_DPD"] > 0).astype(int)
    pos["IS_COMPLETED"] = (pos["NAME_CONTRACT_STATUS"] == "Completed").astype(int)

    agg = pos.groupby("SK_ID_CURR").agg(
        pos_n_records=("SK_ID_PREV", "count"),
        pos_avg_dpd=("SK_DPD", "mean"),
        pos_max_dpd=("SK_DPD", "max"),
        pos_dpd_rate=("IS_DPD", "mean"),
        pos_completion_rate=("IS_COMPLETED", "mean"),
    ).reset_index()
    return agg


def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    inst = inst.copy()
    inst["DAYS_LATE"] = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
    # An installment that was never paid (no DAYS_ENTRY_PAYMENT/AMT_PAYMENT
    # record at all) is the worst-case outcome, not "on time" -- NaN > 0 is
    # False, so without the explicit isna() check these rows would silently
    # read as clean payments instead of the strongest delinquency signal.
    inst["LATE_FLAG"] = ((inst["DAYS_LATE"] > 0) | inst["DAYS_ENTRY_PAYMENT"].isna()).astype(int)
    inst["PAYMENT_SHORTFALL"] = (inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"].fillna(0)).clip(lower=0)

    lifetime = inst.groupby("SK_ID_CURR").agg(
        inst_n_payments=("NUM_INSTALMENT_NUMBER", "count"),
        inst_max_days_late=("DAYS_LATE", "max"),
        inst_avg_days_late=("DAYS_LATE", "mean"),
        inst_late_rate=("LATE_FLAG", "mean"),
        inst_shortfall_sum=("PAYMENT_SHORTFALL", "sum"),
        inst_instalment_sum=("AMT_INSTALMENT", "sum"),
    ).reset_index()
    lifetime["inst_payment_shortfall_ratio"] = (
        lifetime["inst_shortfall_sum"] / lifetime["inst_instalment_sum"].replace(0, np.nan)
    )

    # recency ordered by DAYS_INSTALMENT (closer to 0 = more recent), across all previous loans
    recent = (
        inst.sort_values(["SK_ID_CURR", "DAYS_INSTALMENT"], ascending=[True, False])
        .groupby("SK_ID_CURR")
        .head(6)
        .groupby("SK_ID_CURR")["LATE_FLAG"]
        .mean()
        .rename("inst_late_rate_last6")
        .reset_index()
    )

    return lifetime.merge(recent, on="SK_ID_CURR", how="left")


def _utilization_trend(g: pd.DataFrame) -> float:
    g = g.dropna(subset=["UTILIZATION"])
    if len(g) < 3:
        return np.nan
    return np.polyfit(g["MONTHS_BALANCE"], g["UTILIZATION"], 1)[0]


def aggregate_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    cc = cc.copy()
    cc["UTILIZATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    cc["PAYMENT_RATIO"] = cc["AMT_PAYMENT_CURRENT"] / cc["AMT_INST_MIN_REGULARITY"].replace(0, np.nan)
    cc["CASH_ADVANCE_RATIO"] = cc["AMT_DRAWINGS_ATM_CURRENT"] / cc["AMT_DRAWINGS_CURRENT"].replace(0, np.nan)
    cc["IS_DPD"] = (cc["SK_DPD"] > 0).astype(int)

    agg = cc.groupby("SK_ID_CURR").agg(
        cc_avg_balance=("AMT_BALANCE", "mean"),
        cc_max_balance=("AMT_BALANCE", "max"),
        cc_avg_utilization=("UTILIZATION", "mean"),
        cc_max_utilization=("UTILIZATION", "max"),
        cc_avg_payment_ratio=("PAYMENT_RATIO", "mean"),
        cc_avg_cash_advance_ratio=("CASH_ADVANCE_RATIO", "mean"),
        cc_dpd_rate=("IS_DPD", "mean"),
        cc_max_dpd=("SK_DPD", "max"),
    ).reset_index()

    trend = (
        cc.groupby("SK_ID_CURR")
        .apply(_utilization_trend, include_groups=False)
        .rename("cc_utilization_trend")
        .reset_index()
    )
    return agg.merge(trend, on="SK_ID_CURR", how="left")


def compute_all_aggregates(
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
    prev: pd.DataFrame,
    pos: pd.DataFrame,
    inst: pd.DataFrame,
    cc: pd.DataFrame,
) -> list[pd.DataFrame]:
    """Aggregate every auxiliary table to SK_ID_CURR grain once — reused across train/valid/test splits."""
    return [
        aggregate_bureau(bureau, bureau_balance),
        aggregate_previous_application(prev),
        aggregate_pos_cash(pos),
        aggregate_installments(inst),
        aggregate_credit_card(cc),
    ]


def merge_aggregates(app: pd.DataFrame, aggregates: list[pd.DataFrame]) -> pd.DataFrame:
    df = clean_application(app)
    for agg_df in aggregates:
        df = df.merge(agg_df, on="SK_ID_CURR", how="left")
    return df


def build_feature_matrix(
    app: pd.DataFrame,
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
    prev: pd.DataFrame,
    pos: pd.DataFrame,
    inst: pd.DataFrame,
    cc: pd.DataFrame,
) -> pd.DataFrame:
    aggregates = compute_all_aggregates(bureau, bureau_balance, prev, pos, inst, cc)
    return merge_aggregates(app, aggregates)


def fit_outlier_caps(df: pd.DataFrame, cols, lower_q: float = 0.005, upper_q: float = 0.995) -> dict:
    caps = {}
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            lo, hi = df[c].quantile([lower_q, upper_q])
            if pd.notna(lo) and pd.notna(hi) and lo < hi:
                caps[c] = (lo, hi)
    return caps


def apply_outlier_caps(df: pd.DataFrame, caps: dict) -> pd.DataFrame:
    df = df.copy()
    for c, (lo, hi) in caps.items():
        if c in df.columns:
            df[c] = df[c].clip(lower=lo, upper=hi)
    return df


def add_missingness_flags(df: pd.DataFrame, cols, threshold: float = 0.2):
    df = df.copy()
    flag_cols = []
    for c in cols:
        if c in df.columns and df[c].isna().mean() > threshold:
            flag_col = f"FLAG_{c}_MISSING"
            df[flag_col] = df[c].isna().astype(int)
            flag_cols.append(flag_col)
    return df, flag_cols
