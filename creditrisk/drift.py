"""Population Stability Index (PSI) drift monitoring, the standard check in credit risk model monitoring."""

import numpy as np
import pandas as pd

STABLE, MODERATE = 0.10, 0.25


def _edges(reference: np.ndarray, bins: int) -> np.ndarray:
    uniques = np.unique(reference)
    if len(uniques) <= bins:
        mids = (uniques[:-1] + uniques[1:]) / 2
        return np.concatenate([[-np.inf], mids, [np.inf]])
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.concatenate([[-np.inf], edges, [np.inf]])


def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = _edges(reference, bins)
    expected = np.histogram(reference, edges)[0] / len(reference)
    actual = np.histogram(current, edges)[0] / len(current)
    expected, actual = np.clip(expected, eps, None), np.clip(actual, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def status(value: float) -> str:
    return "stable" if value < STABLE else "moderate" if value < MODERATE else "significant"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> list[dict]:
    rows = [{"feature": c, "psi": psi(reference[c], current[c])} for c in columns]
    for r in rows:
        r["status"] = status(r["psi"])
    return sorted(rows, key=lambda r: -r["psi"])


def simulate_stress(raw: pd.DataFrame, seed: int = 0, severity: float = 0.3) -> pd.DataFrame:
    """A downturn scenario on raw statements: more customers slip a month behind, balances grow and
    repayments shrink. Used to show what the monitor flags; not a forecast."""
    rng = np.random.default_rng(seed)
    out = raw.copy()
    slip = rng.random(len(out)) < severity
    for m in range(1, 7):
        col = f"pay_{m}"
        out.loc[slip, col] = np.where(out.loc[slip, col] >= 1, out.loc[slip, col] + 1, 1).clip(max=8)
        out[f"bill_amt{m}"] = out[f"bill_amt{m}"] * (1 + 0.5 * severity)
        out[f"pay_amt{m}"] = out[f"pay_amt{m}"] * (1 - severity)
    return out
