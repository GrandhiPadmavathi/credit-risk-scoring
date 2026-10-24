"""Static report figures (PNG) for the README and MLflow."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from creditrisk import metrics  # noqa: E402
from creditrisk.features import FEATURE_LABELS  # noqa: E402

COLORS = {"logistic_regression": "#6b7280", "lightgbm": "#2563eb"}
LABELS = {"logistic_regression": "Logistic regression", "lightgbm": "LightGBM"}


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def roc(models, X, y, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, m in models.items():
        p = m.predict_pd(X)
        fpr, tpr, _ = roc_curve(y, p)
        auc = metrics.classification_report(y, p)["auc"]
        ax.plot(fpr, tpr, color=COLORS[name], label=f"{LABELS[name]} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="#9ca3af")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    ax.legend(loc="lower right")
    return _save(fig, out / "roc.png")


def calibration(models, X, y, out: Path, bins: int = 10) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, m in models.items():
        for p, style, suffix in ((m.raw_proba(X), "--", " (raw)"), (m.predict_pd(X), "-", " (calibrated)")):
            order = np.argsort(p)
            chunks = np.array_split(order, bins)
            ax.plot([p[c].mean() for c in chunks], [y[c].mean() for c in chunks], style, marker="o", ms=3,
                    color=COLORS[name], label=LABELS[name] + suffix, alpha=1 if style == "-" else 0.5)
    ax.plot([0, 1], [0, 1], ":", color="#9ca3af")
    ax.set(xlabel="Predicted default probability", ylabel="Observed default rate", title="Calibration (test set)",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    return _save(fig, out / "calibration.png")


def profit(models, X, y, costs, threshold, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, m in models.items():
        curve = metrics.profit_curve(y, m.predict_pd(X), X["limit_bal"], costs)
        ax.plot([r["threshold"] for r in curve], [r["profit"] / 1e6 for r in curve], color=COLORS[name],
                label=LABELS[name])
    ax.axvline(threshold, color="#16a34a", ls="--", label=f"Chosen threshold {threshold:.2f}")
    ax.axvline(0.5, color="#dc2626", ls=":", label="Default 0.5 cut-off")
    ax.axhline(0, color="#9ca3af", lw=0.8)
    ax.set(xlabel="Decline if PD is at least", ylabel="Portfolio profit (NT$ millions)",
           title="Profit vs approval threshold (test set)")
    ax.legend(fontsize=8)
    return _save(fig, out / "profit_curve.png")


def importance(imp, out: Path, top: int = 12) -> Path:
    imp = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh([FEATURE_LABELS.get(f, f) for f in imp.index], imp.values, color="#2563eb")
    ax.set(xlabel="Mean |log-odds contribution| (TreeSHAP)", title="What drives the score")
    return _save(fig, out / "importance.png")


def save_all(models, X, y, costs, threshold, imp, out: Path) -> list[Path]:
    y = np.asarray(y)
    return [roc(models, X, y, out), calibration(models, X, y, out), profit(models, X, y, costs, threshold, out),
            importance(imp, out)]
