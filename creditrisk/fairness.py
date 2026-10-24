"""Group-level audit of decisions across protected attributes the model never sees."""

import numpy as np
import pandas as pd

SEX = {1: "male", 2: "female"}
MARRIAGE = {1: "married", 2: "single", 3: "other"}
EDUCATION = {1: "graduate school", 2: "university", 3: "high school", 4: "other"}


def protected_groups(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sex": raw["sex"].map(SEX).fillna("unknown"),
            "age_band": pd.cut(raw["age"], [0, 29, 44, 200], labels=["under 30", "30-44", "45+"]).astype(str),
            "marriage": raw["marriage"].map(MARRIAGE).fillna("other"),
            "education": raw["education"].map(EDUCATION).fillna("other"),
        },
        index=raw.index,
    )


MIN_GROUP = 100


def audit(groups: pd.DataFrame, y, pd_hat, approve, min_group: int = MIN_GROUP) -> dict:
    """Per attribute: approval rate, observed default rate, mean predicted PD, and approval rate among
    customers who actually repaid (equal opportunity). The adverse impact ratio compares each group's
    approval rate with the most-approved group; below 0.8 fails the common four-fifths rule.
    Groups smaller than `min_group` are reported but excluded from the summary gaps (too noisy to judge)."""
    frame = groups.assign(y=np.asarray(y), pd_hat=np.asarray(pd_hat), approve=np.asarray(approve))
    report = {}
    for attr in groups.columns:
        rows = []
        for value, g in frame.groupby(attr):
            good = g[g["y"] == 0]
            rows.append(
                {
                    "group": str(value),
                    "n": int(len(g)),
                    "default_rate": float(g["y"].mean()),
                    "mean_pd": float(g["pd_hat"].mean()),
                    "approval_rate": float(g["approve"].mean()),
                    "good_customer_approval": float(good["approve"].mean()) if len(good) else float("nan"),
                }
            )
        sized = [r for r in rows if r["n"] >= min_group] or rows
        best = max(r["approval_rate"] for r in sized)
        best_good = max(r["good_customer_approval"] for r in sized)
        for r in rows:
            r["small_sample"] = r["n"] < min_group
            r["adverse_impact_ratio"] = r["approval_rate"] / best if best else float("nan")
            r["equal_opportunity_gap"] = best_good - r["good_customer_approval"]
        report[attr] = {
            "groups": rows,
            "min_adverse_impact_ratio": min(r["adverse_impact_ratio"] for r in sized),
            "max_equal_opportunity_gap": max(r["equal_opportunity_gap"] for r in sized),
            "calibration_gap": max(abs(r["mean_pd"] - r["default_rate"]) for r in sized),
            "excluded_small_groups": [r["group"] for r in rows if r["small_sample"]],
        }
    return report
