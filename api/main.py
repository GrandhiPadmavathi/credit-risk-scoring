"""Scoring service: probability of default, decision, reason codes, decision letter and drift checks."""

import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from creditrisk import drift
from creditrisk.config import SETTINGS
from creditrisk.features import FEATURES
from creditrisk.letter import compose_letter
from creditrisk.scoring import Scorer, default_scorer

Status = Field(ge=-2, le=9, description="-2 no balance, -1 paid in full, 0 revolving, 1-9 months overdue")
Amount = Field(description="NT$")


class Applicant(BaseModel):
    """Credit limit and six monthly statements; index 1 is the most recent month.
    Demographics (sex, age, marital status, education) are not model inputs and are not requested."""

    limit_bal: float = Field(gt=0, description="Credit limit (NT$)")
    pay_1: int = Status
    pay_2: int = Status
    pay_3: int = Status
    pay_4: int = Status
    pay_5: int = Status
    pay_6: int = Status
    bill_amt1: float = Amount
    bill_amt2: float = Amount
    bill_amt3: float = Amount
    bill_amt4: float = Amount
    bill_amt5: float = Amount
    bill_amt6: float = Amount
    pay_amt1: float = Field(ge=0)
    pay_amt2: float = Field(ge=0)
    pay_amt3: float = Field(ge=0)
    pay_amt4: float = Field(ge=0)
    pay_amt5: float = Field(ge=0)
    pay_amt6: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "limit_bal": 50000,
                "pay_1": 2, "pay_2": 2, "pay_3": 0, "pay_4": 0, "pay_5": 0, "pay_6": 0,
                "bill_amt1": 48500, "bill_amt2": 47200, "bill_amt3": 45100, "bill_amt4": 30200,
                "bill_amt5": 29000, "bill_amt6": 28100,
                "pay_amt1": 0, "pay_amt2": 1500, "pay_amt3": 1800, "pay_amt4": 1200, "pay_amt5": 1100, "pay_amt6": 1000,
            }
        }
    }


class Batch(BaseModel):
    applicants: list[Applicant] = Field(min_length=1, max_length=5000)


def frame(applicants: list[Applicant]) -> pd.DataFrame:
    return pd.DataFrame([a.model_dump() for a in applicants])


def create_app(scorer: Scorer | None = None) -> FastAPI:
    app = FastAPI(title="Credit Risk Scoring API", version="1.0")

    def get_scorer() -> Scorer:
        try:
            return scorer or default_scorer()
        except FileNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/health")
    def health():
        s = get_scorer()
        return {"status": "ok", "model": s.model.name, "threshold": s.threshold, "trained_at": s.bundle["trained_at"]}

    @app.get("/model")
    def model_card_metrics():
        path = SETTINGS.paths.metrics
        if not path.exists():
            raise HTTPException(404, "metrics.json not found; train the model first")
        m = json.loads(path.read_text(encoding="utf-8"))
        return {k: m[k] for k in ("champion", "threshold", "costs", "dataset", "baselines_test")} | {
            "test_metrics": {n: {k: v for k, v in r["test"].items() if k != "decision"} for n, r in m["models"].items()}
        }

    @app.post("/score")
    def score(applicant: Applicant, letter: bool = False):
        result = get_scorer().score(frame([applicant]))[0]
        result.pop("id")
        if letter:
            result["letter"] = compose_letter(result)
        return result

    @app.post("/score/batch")
    def score_batch(batch: Batch):
        results = get_scorer().score(frame(batch.applicants), n_reasons=2)
        approved = sum(r["decision"] == "approve" for r in results)
        return {"count": len(results), "approval_rate": approved / len(results), "results": results}

    @app.post("/monitor/drift")
    def monitor_drift(batch: Batch):
        if len(batch.applicants) < 100:
            raise HTTPException(422, "Send at least 100 applicants for a meaningful PSI estimate")
        s = get_scorer()
        current = s.features(frame(batch.applicants))
        features = drift.drift_report(s.bundle["reference"], current, FEATURES)
        score_psi = drift.psi(s.bundle["reference_pd"], s.model.predict_pd(current))
        return {
            "score_psi": score_psi,
            "score_status": drift.status(score_psi),
            "features_flagged": [f["feature"] for f in features if f["status"] != "stable"],
            "features": features,
        }

    return app


app = create_app()
