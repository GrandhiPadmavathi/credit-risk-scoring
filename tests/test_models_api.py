import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from creditrisk.features import FEATURES
from creditrisk.metrics import classification_report

RISKY = {
    "limit_bal": 30000,
    "pay_1": 3, "pay_2": 2, "pay_3": 2, "pay_4": 2, "pay_5": 0, "pay_6": 0,
    "bill_amt1": 31500, "bill_amt2": 30800, "bill_amt3": 30100,
    "bill_amt4": 29500, "bill_amt5": 27000, "bill_amt6": 25000,
    "pay_amt1": 0, "pay_amt2": 0, "pay_amt3": 0, "pay_amt4": 500, "pay_amt5": 1000, "pay_amt6": 1000,
}
SAFE = {
    "limit_bal": 200000,
    **{f"pay_{m}": -1 for m in range(1, 7)},
    **{f"bill_amt{m}": 10000 for m in range(1, 7)},
    **{f"pay_amt{m}": 10000 for m in range(1, 7)},
}


def test_models_learn_signal(raw, feats, fitted):
    y = raw["defaulted"].to_numpy()
    half = len(raw) // 2
    for model in fitted.values():
        assert classification_report(y[half:], model.predict_pd(feats.iloc[half:]))["auc"] > 0.7


@pytest.mark.parametrize("name", ["lightgbm", "logistic_regression"])
def test_contributions_are_additive(feats, fitted, name):
    """Contributions must add up to the model's log-odds (up to a constant), so reason codes are faithful."""
    model = fitted[name]
    X = feats.iloc[:200]
    gap = model.log_odds(X) - model.contributions(X)[FEATURES].sum(axis=1).to_numpy()
    assert np.ptp(gap) < 1e-6


def test_lightgbm_is_monotone_in_payment_delay(fitted, feats):
    model = fitted["lightgbm"]
    base = feats.iloc[[0] * 9].reset_index(drop=True)
    base["recent_delay"] = np.arange(9)
    base["max_delay"] = np.maximum(base["max_delay"], base["recent_delay"])
    scores = model.log_odds(base)
    assert np.all(np.diff(scores) >= -1e-9)


def test_score_endpoint(scorer):
    client = TestClient(create_app(scorer))
    risky = client.post("/score", json=RISKY).json()
    safe = client.post("/score", json=SAFE).json()
    assert risky["probability_of_default"] > safe["probability_of_default"]
    assert risky["decision"] == "decline" and safe["decision"] == "approve"
    adverse = {r["feature"] for r in risky["reasons"]["adverse"]}
    assert adverse & {"recent_delay", "max_delay", "months_delayed", "months_delayed_last3", "months_no_payment"}
    assert all(r["impact"] > 0 for r in risky["reasons"]["adverse"])


def test_score_with_letter_falls_back_without_llm(scorer, monkeypatch):
    monkeypatch.setattr("creditrisk.letter.SETTINGS", type("S", (), {"llm_enabled": False})())
    body = TestClient(create_app(scorer)).post("/score?letter=true", json=RISKY).json()
    assert body["letter"]["source"] == "template" and "unable to approve" in body["letter"]["text"]


def test_validation_rejects_bad_input(scorer):
    client = TestClient(create_app(scorer))
    assert client.post("/score", json={**RISKY, "pay_1": 12}).status_code == 422
    assert client.post("/score", json={**RISKY, "limit_bal": 0}).status_code == 422


def test_batch_and_drift(scorer, raw):
    client = TestClient(create_app(scorer))
    cols = list(RISKY)
    sample = raw[cols].head(300)
    batch = client.post("/score/batch", json={"applicants": sample.to_dict("records")}).json()
    assert batch["count"] == 300 and 0 < batch["approval_rate"] < 1

    assert client.post("/monitor/drift", json={"applicants": sample.head(50).to_dict("records")}).status_code == 422
    stable = client.post("/monitor/drift", json={"applicants": raw[cols].iloc[1500:2500].to_dict("records")}).json()
    assert stable["score_status"] == "stable"

    shifted = pd.DataFrame([RISKY] * 300)
    drifted = client.post("/monitor/drift", json={"applicants": shifted.to_dict("records")}).json()
    assert drifted["score_status"] == "significant" and drifted["features_flagged"]


def test_health(scorer):
    body = TestClient(create_app(scorer)).get("/health").json()
    assert body["status"] == "ok" and body["model"] == "lightgbm"
