import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from creditrisk.config import CostModel
from creditrisk.features import build_features
from creditrisk.models import lightgbm, logistic_regression
from creditrisk.scoring import Scorer
from creditrisk.security import SCOPES, KeyStore, new_key

KEY_SCOPES = {"full": list(SCOPES), "portal": ["score"], "monitoring": ["monitor", "model"]}


def synthetic_applicants(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Fake raw records in the source schema; default risk rises with late payments and utilisation."""
    rng = np.random.default_rng(seed)
    limit = rng.choice([20000, 50000, 80000, 120000, 200000, 360000], n)
    lateness = rng.random(n)
    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "limit_bal": limit,
        "sex": rng.choice([1, 2], n),
        "education": rng.choice([1, 2, 3, 4], n, p=[0.35, 0.45, 0.15, 0.05]),
        "marriage": rng.choice([1, 2, 3], n, p=[0.45, 0.5, 0.05]),
        "age": rng.integers(21, 70, n),
    })
    util = np.clip(rng.beta(2, 3, n) + 0.3 * lateness, 0, 1.1)
    for m in range(1, 7):
        late = rng.random(n) < lateness * 0.5
        df[f"pay_{m}"] = np.where(late, rng.integers(1, 4, n), rng.choice([-2, -1, 0], n, p=[0.1, 0.3, 0.6]))
        df[f"bill_amt{m}"] = (limit * util * rng.uniform(0.85, 1.05, n)).round()
        df[f"pay_amt{m}"] = np.where(late, 0, (df[f"bill_amt{m}"] * rng.uniform(0.05, 0.6, n)).round())
    risk = -3.2 + 4.0 * (df[[f"pay_{m}" for m in range(1, 4)]] >= 1).sum(axis=1) / 3 + 1.5 * util
    df["defaulted"] = (rng.random(n) < 1 / (1 + np.exp(-risk))).astype(int)
    return df


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return synthetic_applicants()


@pytest.fixture(scope="session")
def feats(raw) -> pd.DataFrame:
    return build_features(raw)


@pytest.fixture(scope="session")
def fitted(raw, feats):
    y = raw["defaulted"].to_numpy()
    half = len(raw) // 2
    lgbm = lightgbm(feats.iloc[:half], y[:half], seed=0)
    lgbm.calibrate(feats.iloc[half:], y[half:])
    lr = logistic_regression(feats.iloc[:half], y[:half], seed=0)
    lr.calibrate(feats.iloc[half:], y[half:])
    return {"lightgbm": lgbm, "logistic_regression": lr}


@pytest.fixture(scope="session")
def scorer(fitted, feats) -> Scorer:
    model = fitted["lightgbm"]
    reference = feats.iloc[:1000]
    return Scorer({
        "model": model,
        "threshold": 0.3,
        "costs": CostModel(),
        "reference": reference,
        "reference_pd": model.predict_pd(reference),
        "trained_at": "test",
    })


@pytest.fixture
def issued() -> dict:
    """name -> (presented key, stored record) for each entry in KEY_SCOPES."""
    return {name: new_key(name, scopes) for name, scopes in KEY_SCOPES.items()}


@pytest.fixture
def keystore(issued) -> KeyStore:
    return KeyStore([record for _, record in issued.values()])


@pytest.fixture
def headers(issued) -> dict:
    return {name: {"X-API-Key": key} for name, (key, _) in issued.items()}


@pytest.fixture
def client(scorer, keystore, headers) -> TestClient:
    """Client authenticated with a key that has every scope."""
    return TestClient(create_app(scorer, keystore), headers=headers["full"])
