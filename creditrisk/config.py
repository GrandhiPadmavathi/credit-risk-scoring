import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    raw_xls: Path = ROOT / "data" / "raw" / "default of credit card clients.xls"
    db: Path = ROOT / "data" / "credit.duckdb"
    artifacts: Path = ROOT / "artifacts"
    reports: Path = ROOT / "reports"

    @property
    def bundle(self) -> Path:
        return self.artifacts / "model_bundle.joblib"

    @property
    def metrics(self) -> Path:
        return self.reports / "metrics.json"

    @property
    def api_keys(self) -> Path:
        return Path(os.getenv("API_KEYS_FILE", ROOT / "secrets" / "api_keys.json"))


@dataclass(frozen=True)
class CostModel:
    """Per-applicant economics of approving, as fractions of the credit limit (NT$).

    Approving a customer who repays earns `margin_good` x limit over the year (interest and fees, net of cost
    of funds). Approving one who defaults loses `loss_default` x limit (drawn balance x loss given default).
    Declining earns and loses nothing. These are illustrative assumptions, documented in the model card.
    """

    margin_good: float = 0.04
    loss_default: float = 0.30

    def profit(self, limit, defaulted):
        return limit * (self.margin_good * (1 - defaulted) - self.loss_default * defaulted)

    @property
    def breakeven_pd(self) -> float:
        return self.margin_good / (self.margin_good + self.loss_default)


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    paths: Paths = field(default_factory=Paths)
    costs: CostModel = field(default_factory=CostModel)
    mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:3b")
    llm_api_key: str = os.getenv("LLM_API_KEY", "ollama")
    llm_enabled: bool = os.getenv("LLM_ENABLED", "1").lower() in ("1", "true", "yes")


SETTINGS = Settings()
