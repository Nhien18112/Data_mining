"""Preprocessing ported from bank_account_fraud_complete-v2.ipynb (cells 9, 21, 23, 80, 82).

Keep this file behaviour-identical to the notebook: the saved models expect exactly
these 42 columns in exactly this order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "fraud_bool"

# Notebook cell 9
CAT_COLS = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
BIN_COLS = ["email_is_free", "phone_home_valid", "phone_mobile_valid",
            "has_other_cards", "foreign_request", "keep_alive_session"]

# Notebook cell 23 - columns whose negative codes mean "missing" (Datasheet Q7)
MASKED_COLS = ["prev_address_months_count", "intended_balcon_amount", "bank_months_count",
               "current_address_months_count", "session_length_in_minutes",
               "device_distinct_emails_8w"]
FLAG_COLS = [f"{c}_flag" for c in MASKED_COLS]

# Notebook cell 21 - constant column dropped
CONST_COLS = ["device_fraud_count"]

# Notebook cell 80
BASE_NUM = ["income", "name_email_similarity", "customer_age", "days_since_request",
            "prev_address_months_count", "current_address_months_count", "intended_balcon_amount",
            "bank_months_count", "zip_count_4w", "velocity_6h", "velocity_24h", "velocity_4w",
            "bank_branch_count_8w", "date_of_birth_distinct_emails_4w", "credit_risk_score",
            "proposed_credit_limit", "session_length_in_minutes", "device_distinct_emails_8w"]
NUM_FEATURES = BASE_NUM + BIN_COLS + FLAG_COLS
RATIO_FEATURES = ["vel_6h_24h", "vel_24h_4w", "addr_per_age",
                  "bank_per_age", "limit_per_income", "balcon_per_income"]
FEATURE_COLS = NUM_FEATURES + RATIO_FEATURES + CAT_COLS + ["month"]
LR_NUM_COLS = NUM_FEATURES + RATIO_FEATURES + ["month"]

# Raw input schema expected from an uploaded application file (target and month optional)
RAW_INPUT_COLS = BASE_NUM + BIN_COLS + CAT_COLS
OPTIONAL_COLS = [TARGET, "month", "device_fraud_count"]

# For new applications the month is unknown; trees treat any month >= 5 the same as 5.
DEFAULT_MONTH = 5


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop constant column, turn negative missing codes into NaN and add *_flag columns."""
    df = df.drop(columns=[c for c in CONST_COLS if c in df.columns]).copy()
    if "month" not in df.columns:
        df["month"] = DEFAULT_MONTH
    for col in MASKED_COLS:
        df[col] = df[col].astype(float)
        missing = df[col] < 0 if col == "intended_balcon_amount" else df[col] == -1
        df[f"{col}_flag"] = missing.astype(int)
        df.loc[missing, col] = np.nan
    return df


def make_X(d: pd.DataFrame) -> pd.DataFrame:
    """Original features + 6 ratio features + categoricals + month (notebook cell 80)."""
    X = d[NUM_FEATURES + CAT_COLS + ["month"]].copy()
    X["vel_6h_24h"] = d["velocity_6h"].clip(lower=0) / (d["velocity_24h"].clip(lower=0) + 1)
    X["vel_24h_4w"] = d["velocity_24h"] / (d["velocity_4w"] + 1)
    X["addr_per_age"] = d["current_address_months_count"] / d["customer_age"]
    X["bank_per_age"] = d["bank_months_count"] / d["customer_age"]
    X["limit_per_income"] = d["proposed_credit_limit"] / (d["income"] + 0.1)
    X["balcon_per_income"] = d["intended_balcon_amount"] / (d["income"] + 0.1)
    return X[FEATURE_COLS]


def to_lgbm(X: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Cast categoricals using the category lists learned on Train (notebook cell 82)."""
    X = X.copy()
    for c in CAT_COLS:
        values = X[c].astype(str)
        values = values.where(values.isin(categories[c]))  # unseen levels -> missing
        X[c] = values.astype(pd.CategoricalDtype(categories[c]))
    return X


def validate_raw(df: pd.DataFrame, categories: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for an uploaded raw application table."""
    errors, warnings = [], []
    missing = [c for c in RAW_INPUT_COLS if c not in df.columns]
    if missing:
        errors.append("Thiếu cột bắt buộc: " + ", ".join(missing))
    extra = [c for c in df.columns if c not in RAW_INPUT_COLS + OPTIONAL_COLS]
    if extra:
        warnings.append("Bỏ qua các cột không dùng: " + ", ".join(extra))
    if errors:
        return errors, warnings
    for c in BASE_NUM + BIN_COLS:
        if not pd.api.types.is_numeric_dtype(df[c]):
            errors.append(f"Cột `{c}` phải là số.")
    for c in BIN_COLS:
        if pd.api.types.is_numeric_dtype(df[c]) and not df[c].dropna().isin([0, 1]).all():
            errors.append(f"Cột `{c}` chỉ nhận giá trị 0/1.")
    for c in CAT_COLS:
        unknown = sorted(set(df[c].dropna().astype(str)) - set(categories[c]))
        if unknown:
            warnings.append(f"Cột `{c}` có giá trị chưa gặp khi huấn luyện ({', '.join(unknown[:5])}) "
                            "- mô hình coi là thiếu dữ liệu.")
    if df[RAW_INPUT_COLS].isna().any().any():
        warnings.append("Có ô trống (NaN) - LightGBM xử lý như dữ liệu thiếu.")
    return errors, warnings
