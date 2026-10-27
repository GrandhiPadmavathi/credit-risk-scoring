import json
import logging
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from creditrisk import security
from creditrisk.security import KeyStore, RateLimiter, new_key
from tests.test_models_api import RISKY

BATCH = {"applicants": [RISKY] * 100}


def test_every_endpoint_but_health_needs_a_key(scorer, keystore):
    client = TestClient(create_app(scorer, keystore))
    assert client.get("/health").status_code == 200
    for method, path, body in [("get", "/model", None), ("get", "/auth/key", None), ("post", "/score", RISKY),
                               ("post", "/score/batch", BATCH), ("post", "/monitor/drift", BATCH)]:
        resp = client.request(method, path, json=body)
        assert resp.status_code == 401, path
        assert resp.headers["WWW-Authenticate"] == "ApiKey"


@pytest.mark.parametrize("presented", ["", "not-a-key", "crk_00000000_secret", "crk_x"])
def test_malformed_or_unknown_keys_are_rejected(scorer, keystore, presented):
    resp = TestClient(create_app(scorer, keystore)).post("/score", json=RISKY, headers={"X-API-Key": presented})
    assert resp.status_code == 401


def test_right_id_wrong_secret_is_rejected(scorer, keystore, issued):
    key, record = issued["full"]
    forged = f"crk_{record.id}_{'A' * 43}"
    assert forged != key
    resp = TestClient(create_app(scorer, keystore)).post("/score", json=RISKY, headers={"X-API-Key": forged})
    assert resp.status_code == 401 and resp.json()["detail"] == "Invalid API key"


def test_fails_closed_when_no_keys_are_configured(scorer, tmp_path):
    client = TestClient(create_app(scorer, KeyStore(path=tmp_path / "missing.json")))
    resp = client.post("/score", json=RISKY, headers={"X-API-Key": "crk_abc_def"})
    assert resp.status_code == 503 and "No API keys" in resp.json()["detail"]


def test_scopes_limit_what_each_key_can_do(scorer, keystore, headers):
    client = TestClient(create_app(scorer, keystore))
    portal, monitoring = headers["portal"], headers["monitoring"]

    assert client.post("/score", json=RISKY, headers=portal).status_code == 200
    assert client.post("/score/batch", json=BATCH, headers=portal).status_code == 200
    denied = client.post("/score?letter=true", json=RISKY, headers=portal)
    assert denied.status_code == 403 and "'letter'" in denied.json()["detail"]
    assert client.post("/monitor/drift", json=BATCH, headers=portal).status_code == 403

    assert client.post("/monitor/drift", json=BATCH, headers=monitoring).status_code == 200
    assert client.post("/score", json=RISKY, headers=monitoring).status_code == 403
    me = client.get("/auth/key", headers=monitoring).json()
    assert me["name"] == "monitoring" and me["scopes"] == ["model", "monitor"] and "sha256" not in me


def test_revoked_and_expired_keys_are_rejected(scorer):
    key, record = new_key("old", ["score"])
    client = TestClient(create_app(scorer, KeyStore([record])), headers={"X-API-Key": key})
    assert client.post("/score", json=RISKY).status_code == 200

    record.expires = (date.today() - timedelta(days=1)).isoformat()
    assert client.post("/score", json=RISKY).status_code == 401
    record.expires, record.disabled = None, True
    assert client.post("/score", json=RISKY).json()["detail"] == "API key revoked or expired"


def test_cli_stores_only_a_hash_and_revocation_needs_no_restart(scorer, tmp_path, capsys):
    path = tmp_path / "keys.json"
    security.main(["--file", str(path), "create", "--name", "loan-portal", "--scopes", "score"])
    key = capsys.readouterr().out.strip().splitlines()[-1]
    assert key.startswith("crk_")
    stored = path.read_text(encoding="utf-8")
    assert key.split("_", 2)[2] not in stored and json.loads(stored)["keys"][0]["name"] == "loan-portal"

    client = TestClient(create_app(scorer, KeyStore(path=path)), headers={"X-API-Key": key})
    assert client.post("/score", json=RISKY).status_code == 200
    security.main(["--file", str(path), "revoke", key.split("_")[1]])
    assert client.post("/score", json=RISKY).status_code == 401

    with pytest.raises(ValueError, match="Unknown scope"):
        new_key("x", ["score", "admin"])


def test_rate_limit_is_per_key_and_recovers(scorer):
    now = [0.0]
    limited_key, limited = new_key("limited", ["score"], rate_per_minute=2)
    other_key, other = new_key("other", ["score"], rate_per_minute=2)
    store = KeyStore([limited, other], limiter=RateLimiter(clock=lambda: now[0]))
    client = TestClient(create_app(scorer, store))

    for _ in range(2):
        assert client.post("/score", json=RISKY, headers={"X-API-Key": limited_key}).status_code == 200
    blocked = client.post("/score", json=RISKY, headers={"X-API-Key": limited_key})
    assert blocked.status_code == 429 and int(blocked.headers["Retry-After"]) > 0
    assert client.post("/score", json=RISKY, headers={"X-API-Key": other_key}).status_code == 200

    now[0] = 61.0
    assert client.post("/score", json=RISKY, headers={"X-API-Key": limited_key}).status_code == 200


def test_decisions_are_audited_without_applicant_figures(client, headers, caplog):
    caplog.set_level(logging.INFO, logger="creditrisk.audit")
    resp = client.post("/score", json=RISKY)
    client.post("/monitor/drift", json=BATCH, headers=headers["portal"])

    events = [json.loads(r.getMessage()) for r in caplog.records if r.name == "creditrisk.audit"]
    decision = next(e for e in events if e["event"] == "decision")
    assert decision["client"] == "full" and decision["decision"] == resp.json()["decision"]
    assert decision["request_id"] == resp.headers["X-Request-ID"]
    assert not {"limit_bal", "bill_amt1", "pay_amt1"} & set(decision)
    assert str(RISKY["bill_amt1"]) not in caplog.text
    denied = next(e for e in events if e["event"] == "access_denied")
    assert denied["client"] == "portal" and denied["scope"] == "monitor"
