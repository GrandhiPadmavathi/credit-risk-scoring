"""Turn per-feature log-odds contributions into ranked, human-readable reason codes."""

import pandas as pd

from creditrisk.features import FEATURE_LABELS


def describe(feature: str, value: float) -> str:
    v = float(value)
    texts = {
        "limit_bal": f"Credit limit of NT${v:,.0f}",
        "max_delay": f"Worst payment delay in 6 months: {v:.0f} month(s)" if v else "No late payments in 6 months",
        "recent_delay": f"Latest statement is {v:.0f} month(s) overdue" if v else "Latest statement is not overdue",
        "months_delayed": f"Paid late in {v:.0f} of the last 6 months" if v else "No late payments in 6 months",
        "months_delayed_last3": f"Paid late in {v:.0f} of the last 3 months" if v else "No late payments in 3 months",
        "months_paid_full": f"Paid the full balance in {v:.0f} of the last 6 months" if v
        else "Did not pay the full balance in any of the last 6 months",
        "months_no_balance": f"No balance in {v:.0f} of the last 6 months" if v
        else "Carried a balance in each of the last 6 months",
        "months_no_payment": f"Made no payment on {v:.0f} bill(s) in the last 5 months" if v
        else "Made a payment on every bill in the last 5 months",
        "utilization": f"Current balance is {v:.0%} of the credit limit",
        "avg_utilization": f"Average balance is {v:.0%} of the credit limit",
        "max_utilization": f"Peak balance reached {v:.0%} of the credit limit",
        "pay_to_bill_ratio": f"Repaid {v:.0%} of billed amounts over the last 5 months",
        "last_payment_to_limit": f"Latest payment was {v:.1%} of the credit limit",
        "avg_payment_to_limit": f"Average monthly payment is {v:.1%} of the credit limit",
        "bill_trend": f"Balance {'grew' if v > 0 else 'fell'} by {abs(v):.1%} of the limit per month",
    }
    return texts.get(feature, f"{FEATURE_LABELS.get(feature, feature)}: {v:g}")


def reason_codes(features: pd.Series, contributions: pd.Series, n: int = 4) -> dict:
    """Top factors raising risk (adverse) and lowering it (favourable) for one applicant."""
    ranked = contributions.sort_values()
    adverse = [f for f in ranked.index[::-1] if ranked[f] > 0][:n]
    favourable = [f for f in ranked.index if ranked[f] < 0][:n]

    def item(f):
        return {
            "feature": f,
            "label": FEATURE_LABELS.get(f, f),
            "value": float(features[f]),
            "impact": round(float(contributions[f]), 4),
            "text": describe(f, features[f]),
        }

    return {"adverse": [item(f) for f in adverse], "favourable": [item(f) for f in favourable]}
