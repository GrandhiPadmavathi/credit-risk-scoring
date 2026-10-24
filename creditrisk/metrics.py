import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

from creditrisk.config import CostModel


def ks_statistic(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Weighted gap between predicted and observed default rate over quantile bins (ties share a bin)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)[1:-1]))
    which = np.searchsorted(edges, p, side="right")
    ece = 0.0
    for b in np.unique(which):
        mask = which == b
        ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def classification_report(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "ece": expected_calibration_error(y, p),
    }


def bootstrap_ci(y, p, p_other=None, n_boot: int = 300, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap intervals for AUC and ECE; with `p_other`, also the paired AUC difference."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    stats = {"auc": [], "ece": [], "auc_diff": []}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        auc = roc_auc_score(y[i], p[i])
        stats["auc"].append(auc)
        stats["ece"].append(expected_calibration_error(y[i], p[i]))
        if p_other is not None:
            stats["auc_diff"].append(auc - roc_auc_score(y[i], np.asarray(p_other)[i]))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for k, v in stats.items() if v}


def decision_outcome(y, p, limit, threshold: float, costs: CostModel) -> dict:
    """Approve when predicted default probability is below the threshold."""
    y, p, limit = np.asarray(y), np.asarray(p), np.asarray(limit, dtype=float)
    approve = p < threshold
    profit = costs.profit(limit[approve], y[approve]).sum()
    n_good = max(int((y == 0).sum()), 1)
    return {
        "threshold": float(threshold),
        "approval_rate": float(approve.mean()),
        "default_rate_approved": float(y[approve].mean()) if approve.any() else 0.0,
        "defaulters_caught": float(1 - y[approve].sum() / max(y.sum(), 1)),
        "good_customers_declined": float(((~approve) & (y == 0)).sum() / n_good),
        "profit": float(profit),
        "profit_per_applicant": float(profit / len(y)),
    }


THRESHOLD_GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def profit_curve(y, p, limit, costs: CostModel, grid=THRESHOLD_GRID) -> list[dict]:
    return [decision_outcome(y, p, limit, t, costs) for t in grid]


def best_threshold(y, p, limit, costs: CostModel) -> float:
    curve = profit_curve(y, p, limit, costs)
    return max(curve, key=lambda r: r["profit"])["threshold"]
