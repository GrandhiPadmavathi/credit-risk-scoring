import pandas as pd
import pytest

from creditrisk.data import clean
from creditrisk.features import FEATURES, build_features


def applicant(id_, limit, pay, bill, paid):
    row = {"id": id_, "limit_bal": limit}
    for m in range(6):
        row[f"pay_{m + 1}"], row[f"bill_amt{m + 1}"], row[f"pay_amt{m + 1}"] = pay[m], bill[m], paid[m]
    return row


def test_hand_computed_features():
    raw = pd.DataFrame([
        applicant(7, 10000, [2, 1, 0, -1, -1, -2], [9000, 8000, 7000, 6000, 5000, 4000], [0, 1000, 2000, 1000, 0, 0]),
        applicant(3, 20000, [-1] * 6, [1000] * 6, [1000] * 6),
    ])
    f = build_features(raw).set_index("id")
    late = f.loc[7]
    assert late["max_delay"] == 2 and late["recent_delay"] == 2
    assert late["months_delayed"] == 2 and late["months_delayed_last3"] == 2
    assert late["months_paid_full"] == 2 and late["months_no_balance"] == 1
    assert late["utilization"] == pytest.approx(0.9)
    assert late["max_utilization"] == pytest.approx(0.9)
    # payments in months 1-5 settle bills of months 2-6: 4000 / (8000+7000+6000+5000+4000)
    assert late["pay_to_bill_ratio"] == pytest.approx(4000 / 30000)
    # months 1 and 5 had a bill to settle but no payment
    assert late["months_no_payment"] == 2
    assert late["bill_trend"] > 0  # balance has been rising toward the latest month

    clean_payer = f.loc[3]
    assert clean_payer["max_delay"] == 0 and clean_payer["months_delayed"] == 0
    assert clean_payer["pay_to_bill_ratio"] == pytest.approx(1.0)
    assert clean_payer["bill_trend"] == pytest.approx(0.0)


def test_output_order_and_columns_match_input(raw):
    shuffled = raw.sample(frac=1, random_state=1).head(50)
    f = build_features(shuffled)
    assert list(f["id"]) == list(shuffled["id"])
    assert set(FEATURES) <= set(f.columns)
    assert not f[FEATURES].isna().any().any()


def test_zero_bills_do_not_divide_by_zero():
    raw = pd.DataFrame([applicant(1, 10000, [-2] * 6, [0] * 6, [0] * 6)])
    f = build_features(raw).iloc[0]
    assert f["pay_to_bill_ratio"] == 1.0 and f["utilization"] == 0.0


def test_clean_folds_unknown_codes():
    df = clean(pd.DataFrame({"education": [0, 1, 5, 6, 3], "marriage": [0, 1, 2, 3, 0]}))
    assert list(df["education"]) == [4, 1, 4, 4, 3]
    assert list(df["marriage"]) == [3, 1, 2, 3, 3]
