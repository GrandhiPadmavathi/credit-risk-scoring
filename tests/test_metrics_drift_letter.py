import dataclasses

import numpy as np
import pandas as pd
import pytest

from creditrisk import drift, fairness, letter, metrics
from creditrisk.config import CostModel


def test_fairness_audit_ratios_and_small_groups():
    groups = pd.DataFrame({"sex": ["male"] * 200 + ["female"] * 200 + ["unknown"] * 10})
    y = np.zeros(410)
    approve = np.r_[np.ones(200), np.r_[np.ones(120), np.zeros(80)], np.zeros(10)].astype(bool)
    rep = fairness.audit(groups, y, np.full(410, 0.1), approve)["sex"]
    by = {g["group"]: g for g in rep["groups"]}
    assert by["female"]["adverse_impact_ratio"] == pytest.approx(0.6)
    assert by["female"]["equal_opportunity_gap"] == pytest.approx(0.4)
    assert rep["excluded_small_groups"] == ["unknown"]
    assert rep["min_adverse_impact_ratio"] == pytest.approx(0.6)


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 3000)
    y = (rng.random(3000) < p).astype(int)
    ci = metrics.bootstrap_ci(y, p, p_other=rng.uniform(0, 1, 3000), n_boot=100)
    auc = metrics.classification_report(y, p)["auc"]
    assert ci["auc"][0] < auc < ci["auc"][1]
    assert ci["auc_diff"][0] > 0  # informative scores beat random ones


def test_profit_and_breakeven_threshold():
    costs = CostModel(margin_good=0.04, loss_default=0.30)
    assert costs.breakeven_pd == pytest.approx(0.04 / 0.34)
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.2, 0.1, 0.9])
    out = metrics.decision_outcome(y, p, [1000] * 4, threshold=0.15, costs=costs)
    # approves rows 0 and 2: +40 from the good payer, -300 from the defaulter
    assert out["profit"] == pytest.approx(40 - 300)
    assert out["approval_rate"] == 0.5 and out["defaulters_caught"] == 0.5


def test_best_threshold_is_near_breakeven_for_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50000)
    y = (rng.random(50000) < p).astype(int)
    costs = CostModel()
    t = metrics.best_threshold(y, p, np.full(len(p), 1000.0), costs)
    assert abs(t - costs.breakeven_pd) < 0.03


def test_ks_and_ece():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert metrics.ks_statistic(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == 1.0
    y = (np.arange(1000) < 200).astype(int)
    assert metrics.expected_calibration_error(y, np.full(1000, 0.2)) == pytest.approx(0.0)
    assert metrics.expected_calibration_error(y, np.full(1000, 0.5)) == pytest.approx(0.3)


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20000)
    assert drift.psi(ref, rng.normal(0, 1, 20000)) < 0.01
    assert drift.psi(ref, rng.normal(1, 1, 20000)) > 0.25
    assert drift.status(0.05) == "stable" and drift.status(0.2) == "moderate" and drift.status(0.4) == "significant"


def test_psi_handles_discrete_features():
    ref = np.array([0] * 900 + [1] * 100)
    assert drift.psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert drift.psi(ref, np.array([0] * 500 + [1] * 500)) > 0.25


def test_stress_scenario_increases_delinquency(raw):
    stressed = drift.simulate_stress(raw, seed=0, severity=0.3)
    assert (stressed["pay_1"] >= 1).mean() > (raw["pay_1"] >= 1).mean()
    assert stressed["pay_amt1"].sum() < raw["pay_amt1"].sum()


DECLINE = {
    "decision": "decline",
    "reasons": {
        "adverse": [
            {"text": "Latest statement is 2 month(s) overdue"},
            {"text": "Current balance is 97% of the credit limit"},
        ],
        "favourable": [],
    },
}


def test_grounding_rejects_invented_numbers():
    facts = [r["text"] for r in DECLINE["reasons"]["adverse"]]
    assert letter._grounded("Your latest statement is 2 months overdue and the balance is 97% of the limit.", facts)
    assert not letter._grounded("Your balance is 97% of the limit. Call 1800 555 0100 for help.", facts)


def test_template_letter_lists_reasons():
    text = letter.template_letter(DECLINE)
    assert "unable to approve" in text
    assert "- Latest statement is 2 month(s) overdue" in text


def test_llm_output_used_only_when_grounded(monkeypatch):
    monkeypatch.setattr(letter, "SETTINGS", dataclasses.replace(letter.SETTINGS, llm_enabled=True))

    class Resp:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": self.content}}]}

    monkeypatch.setattr(letter.requests, "post", lambda *a, **k: Resp("Declined: 2 months overdue, 97% balance."))
    assert letter.compose_letter(DECLINE)["source"] == "llm"
    monkeypatch.setattr(letter.requests, "post", lambda *a, **k: Resp("Declined. Your new rate would be 24.9%."))
    out = letter.compose_letter(DECLINE)
    assert out["source"] == "template" and "grounding" in out["note"]

    def down(*a, **k):
        raise letter.requests.ConnectionError("no server")

    monkeypatch.setattr(letter.requests, "post", down)
    assert letter.compose_letter(DECLINE)["source"] == "template"
