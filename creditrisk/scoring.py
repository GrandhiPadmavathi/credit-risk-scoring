"""Load the trained bundle and score raw applicant records."""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from creditrisk.config import SETTINGS
from creditrisk.explain import reason_codes
from creditrisk.features import build_features

RISK_BANDS = [(0.05, "A"), (0.10, "B"), (0.20, "C"), (0.35, "D"), (1.01, "E")]


def risk_band(p: float) -> str:
    return next(band for cutoff, band in RISK_BANDS if p < cutoff)


class Scorer:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.model = bundle["model"]
        self.threshold = bundle["threshold"]
        self.costs = bundle["costs"]

    @classmethod
    def load(cls, path: Path = SETTINGS.paths.bundle) -> "Scorer":
        if not Path(path).exists():
            raise FileNotFoundError(f"No trained model at {path}. Run: python -m creditrisk.train")
        return cls(joblib.load(path))

    def features(self, raw: pd.DataFrame) -> pd.DataFrame:
        raw = raw.copy()
        if "id" not in raw:
            raw.insert(0, "id", np.arange(1, len(raw) + 1))
        return build_features(raw)

    def score(self, raw: pd.DataFrame, n_reasons: int = 4) -> list[dict]:
        feats = self.features(raw)
        pd_hat = self.model.predict_pd(feats)
        contrib = self.model.contributions(feats)
        results = []
        for i in range(len(feats)):
            p = float(pd_hat[i])
            limit = float(feats["limit_bal"].iloc[i])
            expected_profit = limit * (self.costs.margin_good * (1 - p) - self.costs.loss_default * p)
            results.append(
                {
                    "id": int(feats["id"].iloc[i]),
                    "probability_of_default": round(p, 4),
                    "risk_band": risk_band(p),
                    "decision": "approve" if p < self.threshold else "decline",
                    "threshold": self.threshold,
                    "expected_profit": round(expected_profit, 2),
                    "reasons": reason_codes(feats.iloc[i], contrib.iloc[i], n=n_reasons),
                }
            )
        return results


@lru_cache(maxsize=1)
def default_scorer() -> Scorer:
    return Scorer.load()
