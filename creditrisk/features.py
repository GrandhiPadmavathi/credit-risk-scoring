"""Feature engineering in DuckDB SQL, shared by training and serving."""

from pathlib import Path

import duckdb
import pandas as pd

FEATURE_SQL = (Path(__file__).with_name("features.sql")).read_text(encoding="utf-8")

# Protected and socioeconomic-proxy attributes are kept out of the model and used only for the fairness audit.
PROTECTED = ["sex", "age", "marriage", "education"]
FEATURES = [
    "limit_bal",
    "max_delay",
    "recent_delay",
    "months_delayed",
    "months_delayed_last3",
    "months_paid_full",
    "months_no_balance",
    "months_no_payment",
    "utilization",
    "avg_utilization",
    "max_utilization",
    "pay_to_bill_ratio",
    "last_payment_to_limit",
    "avg_payment_to_limit",
    "bill_trend",
]

FEATURE_LABELS = {
    "limit_bal": "Credit limit",
    "max_delay": "Worst payment delay (months)",
    "recent_delay": "Payment delay on latest statement (months)",
    "months_delayed": "Months with a late payment (of 6)",
    "months_delayed_last3": "Late months in the last 3",
    "months_paid_full": "Months paid in full (of 6)",
    "months_no_balance": "Months with no balance (of 6)",
    "months_no_payment": "Months with a bill but no payment",
    "utilization": "Current balance / limit",
    "avg_utilization": "Average balance / limit",
    "max_utilization": "Peak balance / limit",
    "pay_to_bill_ratio": "Payments / bills (last 5 months)",
    "last_payment_to_limit": "Latest payment / limit",
    "avg_payment_to_limit": "Average payment / limit",
    "bill_trend": "Monthly balance growth / limit",
}


def build_features(applicants: pd.DataFrame) -> pd.DataFrame:
    """Raw applicant rows (id, limit_bal and the six monthly statements) -> one feature row per id, in input order."""
    con = duckdb.connect()
    try:
        con.register("applicants", applicants)
        feats = con.execute(FEATURE_SQL).df()
    finally:
        con.close()
    return feats.set_index("id").loc[applicants["id"].to_numpy()].reset_index()
