"""Logistic regression baseline and LightGBM challenger, each with isotonic calibration.

Both expose the same interface: raw log-odds, calibrated probability of default, and per-feature
contributions to the log-odds (used for reason codes).
"""

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from creditrisk.features import FEATURES
from creditrisk.metrics import expected_calibration_error

HEAVY_TAILED = ["limit_bal", "last_payment_to_limit", "avg_payment_to_limit", "bill_trend"]

# Business constraints that keep reason codes defensible: late or missed payments can never lower
# predicted risk, and repaying more (or in full) can never raise it.
MONOTONE = {
    "max_delay": 1,
    "recent_delay": 1,
    "months_delayed": 1,
    "months_delayed_last3": 1,
    "months_no_payment": 1,
    "months_paid_full": -1,
    "pay_to_bill_ratio": -1,
}


def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _signed_log_names(transformer, names):
    return names


class PlattScaler:
    """Logistic recalibration on the log-odds: two parameters, so it cannot overfit like isotonic can."""

    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e4).fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self.lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


CALIBRATORS = {
    "none": lambda: None,
    "platt": PlattScaler,
    "isotonic": lambda: IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4),
}


@dataclass
class CalibratedModel:
    name: str
    estimator: object
    calibrator: object | None = None
    calibration_method: str = "none"
    calibration_cv_ece: dict | None = None
    features: list[str] = field(default_factory=lambda: list(FEATURES))

    def log_odds(self, X: pd.DataFrame) -> np.ndarray:
        if self.name == "lightgbm":
            return self.estimator.predict(X[self.features], raw_score=True)
        return self.estimator.decision_function(X[self.features])

    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        return 1 / (1 + np.exp(-self.log_odds(X)))

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.raw_proba(X)
        return self.calibrator.predict(raw) if self.calibrator is not None else raw

    def calibrate(self, X: pd.DataFrame, y, seed: int = 0, folds: int = 5) -> None:
        """Pick none / Platt / isotonic by cross-validated ECE on the calibration set, then refit on all of it."""
        p, y = self.raw_proba(X), np.asarray(y)
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
        scores = {}
        for method, make in CALIBRATORS.items():
            out = np.empty_like(p)
            for fit_idx, eval_idx in cv.split(p.reshape(-1, 1), y):
                cal = make()
                out[eval_idx] = p[eval_idx] if cal is None else cal.fit(p[fit_idx], y[fit_idx]).predict(p[eval_idx])
            scores[method] = expected_calibration_error(y, out)
        self.calibration_method = min(scores, key=scores.get)
        self.calibration_cv_ece = scores
        cal = CALIBRATORS[self.calibration_method]()
        self.calibrator = None if cal is None else cal.fit(p, y)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Additive contribution of each input feature to the log-odds (positive = riskier)."""
        if self.name == "lightgbm":
            contrib = self.estimator.predict(X[self.features], pred_contrib=True)[:, :-1]
            return pd.DataFrame(contrib, columns=self.features, index=X.index)
        pre = self.estimator.named_steps["pre"]
        coef = self.estimator.named_steps["clf"].coef_[0]
        centered = (pre.transform(X[self.features]) - self.reference_mean_) * coef
        names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
        return pd.DataFrame(centered, columns=names, index=X.index)[self.features]


def logistic_regression(X: pd.DataFrame, y, seed: int, features: list[str] = FEATURES) -> CalibratedModel:
    heavy = [f for f in HEAVY_TAILED if f in features]
    light = [f for f in features if f not in HEAVY_TAILED]
    pre = ColumnTransformer(
        [
            ("log", Pipeline([("log", FunctionTransformer(_signed_log, feature_names_out=_signed_log_names)),
                              ("scale", StandardScaler())]), heavy),
            ("num", StandardScaler(), light),
        ]
    )
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.5, max_iter=2000, random_state=seed))])
    pipe.fit(X[features], y)
    model = CalibratedModel("logistic_regression", pipe, features=list(features))
    model.reference_mean_ = np.asarray(pre.transform(X[features])).mean(axis=0)
    return model


def lightgbm(
    X: pd.DataFrame, y, seed: int, features: list[str] = FEATURES, early_stop_frac: float = 0.15
) -> CalibratedModel:
    rng = np.random.default_rng(seed)
    holdout = rng.random(len(X)) < early_stop_frac
    y = np.asarray(y)
    clf = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=15,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        monotone_constraints=[MONOTONE.get(f, 0) for f in features],
        random_state=seed,
        verbose=-1,
    )
    clf.fit(
        X.loc[~holdout, features],
        y[~holdout],
        eval_X=(X.loc[holdout, features],),
        eval_y=(y[holdout],),
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return CalibratedModel("lightgbm", clf, features=list(features))
