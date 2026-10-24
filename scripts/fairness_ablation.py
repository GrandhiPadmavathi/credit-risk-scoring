"""Proxy ablation: how much of the education approval gap comes from the credit limit feature?

The bank sets credit limits partly from income and education, so `limit_bal` (and features scaled by it) can act
as a proxy. This retrains LightGBM without the limit-driven inputs and compares accuracy, profit and fairness on the
same test split. Writes reports/fairness_ablation.json.

Usage: python scripts/fairness_ablation.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from creditrisk import fairness, metrics, models  # noqa: E402
from creditrisk.config import SETTINGS  # noqa: E402
from creditrisk.data import TARGET  # noqa: E402
from creditrisk.features import FEATURES  # noqa: E402
from creditrisk.train import load_dataset, split  # noqa: E402

VARIANTS = {
    "all features": FEATURES,
    "without credit limit": [f for f in FEATURES if f != "limit_bal"],
}


def main() -> None:
    s, costs = SETTINGS, SETTINGS.costs
    raw, feats = load_dataset()
    y = raw[TARGET].to_numpy()
    tr, va, te = split(raw, s.seed)
    groups = fairness.protected_groups(raw.iloc[te])
    rows = []
    for name, cols in VARIANTS.items():
        model = models.lightgbm(feats.iloc[tr], y[tr], s.seed, features=cols)
        model.calibrate(feats.iloc[va], y[va], seed=s.seed)
        threshold = metrics.best_threshold(y[va], model.predict_pd(feats.iloc[va]), feats["limit_bal"].iloc[va], costs)
        p = model.predict_pd(feats.iloc[te])
        audit = fairness.audit(groups, y[te], p, p < threshold)
        outcome = metrics.decision_outcome(y[te], p, feats["limit_bal"].iloc[te], threshold, costs)
        rows.append({
            "variant": name,
            "auc": metrics.classification_report(y[te], p)["auc"],
            "threshold": threshold,
            "approval_rate": outcome["approval_rate"],
            "profit": outcome["profit"],
            **{f"{attr}_min_air": audit[attr]["min_adverse_impact_ratio"] for attr in audit},
            **{f"{attr}_max_eo_gap": audit[attr]["max_equal_opportunity_gap"] for attr in audit},
        })
        r = rows[-1]
        print(f"{name:22s} AUC={r['auc']:.4f} approve={r['approval_rate']:.1%} profit={r['profit']:,.0f} "
              f"education AIR={r['education_min_air']:.3f} EO gap={r['education_max_eo_gap']:.3f} "
              f"age AIR={r['age_band_min_air']:.3f}")
    out = s.paths.reports / "fairness_ablation.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
