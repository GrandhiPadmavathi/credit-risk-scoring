"""Smoke test against the real trained bundle: API endpoints and a headless dashboard render.

Usage: python scripts/smoke.py   (after `python -m creditrisk.train`)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from api.main import app  # noqa: E402
from creditrisk import data, drift  # noqa: E402
from creditrisk.config import ROOT  # noqa: E402
from creditrisk.profiles import PRESETS  # noqa: E402
from creditrisk.profiles import to_applicant as applicant  # noqa: E402


def main() -> None:
    client = TestClient(app)
    print("health:", client.get("/health").json())
    print("model:", json.dumps(client.get("/model").json()["test_metrics"]["lightgbm"], indent=None)[:200])
    for name, preset in PRESETS.items():
        r = client.post("/score", json=applicant(preset)).json()
        top = [x["text"] for x in r["reasons"]["adverse"][:2]]
        print(f"{name:28s} PD={r['probability_of_default']:.3f} band={r['risk_band']} {r['decision']:8s} {top}")

    letter = client.post("/score?letter=true", json=applicant(PRESETS["Recently late"])).json()["letter"]
    print(f"\nletter ({letter['source']}{', ' + letter['note'] if letter.get('note') else ''}):\n{letter['text']}\n")

    raw = data.load_applicants().sample(1000, random_state=3)
    cols = [c for c in applicant(PRESETS["Reliable payer"])]
    for label, frame in (("holdout sample", raw), ("stress scenario", drift.simulate_stress(raw, seed=3))):
        d = client.post("/monitor/drift", json={"applicants": frame[cols].to_dict("records")}).json()
        print(f"drift {label:16s} score PSI={d['score_psi']:.3f} ({d['score_status']}) "
              f"flagged={d['features_flagged'][:4]}")

    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    print("\ndashboard tabs:", [t.label for t in at.tabs])
    print("dashboard metrics:", [(m.label, m.value) for m in at.metric])
    print("dashboard alerts:", [s.value[:60] for s in (*at.success, *at.error)])


if __name__ == "__main__":
    main()
