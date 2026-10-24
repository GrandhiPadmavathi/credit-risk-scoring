"""End-to-end training: data -> SQL features -> models -> calibration -> threshold -> audit -> artifacts.

Usage: python -m creditrisk.train [--no-mlflow]
"""

import argparse
import json
import time
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from creditrisk import data, drift, fairness, metrics, models, plots
from creditrisk.config import SETTINGS
from creditrisk.features import FEATURES, build_features

SPLIT = {"train": 0.6, "validation": 0.2, "test": 0.2}


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = SETTINGS.paths
    if not paths.db.exists():
        data.write_db(data.read_raw(paths.raw_xls), paths.db)
    raw = data.load_applicants(paths.db)
    return raw, build_features(raw)


def split(raw: pd.DataFrame, seed: int):
    idx = np.arange(len(raw))
    y = raw[data.TARGET].to_numpy()
    train, rest = train_test_split(idx, test_size=1 - SPLIT["train"], stratify=y, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    return train, val, test


def evaluate(model, X, y, threshold, costs) -> dict:
    p_raw, p = model.raw_proba(X), model.predict_pd(X)
    report = metrics.classification_report(y, p)
    report["ece_uncalibrated"] = metrics.expected_calibration_error(y, p_raw)
    report["decision"] = metrics.decision_outcome(y, p, X["limit_bal"], threshold, costs)
    return report


def main(use_mlflow: bool = True) -> dict:
    started = time.time()
    s, costs = SETTINGS, SETTINGS.costs
    raw, feats = load_dataset()
    y = raw[data.TARGET].to_numpy()
    tr, va, te = split(raw, s.seed)
    X_tr, X_va, X_te = feats.iloc[tr], feats.iloc[va], feats.iloc[te]

    candidates = {
        "logistic_regression": models.logistic_regression(X_tr, y[tr], s.seed),
        "lightgbm": models.lightgbm(X_tr, y[tr], s.seed),
    }
    results = {}
    for name, model in candidates.items():
        model.calibrate(X_va, y[va], seed=s.seed)
        threshold = metrics.best_threshold(y[va], model.predict_pd(X_va), X_va["limit_bal"], costs)
        results[name] = {
            "threshold": threshold,
            "calibration": {"method": model.calibration_method, "cv_ece": model.calibration_cv_ece},
            "validation": evaluate(model, X_va, y[va], threshold, costs),
            "test": evaluate(model, X_te, y[te], threshold, costs),
        }
    champion_name = max(results, key=lambda n: results[n]["validation"]["auc"])
    champion, threshold = candidates[champion_name], results[champion_name]["threshold"]
    challenger_name = next(n for n in candidates if n != champion_name)

    pd_te = champion.predict_pd(X_te)
    pd_te_challenger = candidates[challenger_name].predict_pd(X_te)
    results[champion_name]["test"]["ci95"] = metrics.bootstrap_ci(y[te], pd_te, pd_te_challenger, seed=s.seed)
    results[challenger_name]["test"]["ci95"] = metrics.bootstrap_ci(y[te], pd_te_challenger, seed=s.seed)
    lim_te = X_te["limit_bal"]
    baselines = {
        "approve_everyone": metrics.decision_outcome(y[te], pd_te, lim_te, 1.01, costs),
        "threshold_0.5": metrics.decision_outcome(y[te], pd_te, lim_te, 0.5, costs),
        "breakeven_pd": metrics.decision_outcome(y[te], pd_te, lim_te, costs.breakeven_pd, costs),
        "profit_optimal": metrics.decision_outcome(y[te], pd_te, lim_te, threshold, costs),
    }

    approve_te = pd_te < threshold
    fairness_report = fairness.audit(fairness.protected_groups(raw.iloc[te]), y[te], pd_te, approve_te)

    stressed = build_features(drift.simulate_stress(raw.iloc[te], seed=s.seed))
    drift_reports = {
        "test_vs_train": drift.drift_report(X_tr, X_te, FEATURES),
        "stress_vs_train": drift.drift_report(X_tr, stressed, FEATURES),
    }
    score_psi = {
        "test_vs_train": drift.psi(champion.predict_pd(X_tr), pd_te),
        "stress_vs_train": drift.psi(champion.predict_pd(X_tr), champion.predict_pd(stressed)),
    }

    contrib = champion.contributions(X_te)
    importance = contrib.abs().mean().sort_values(ascending=False)

    summary = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {"rows": len(raw), "default_rate": float(y.mean()), "split": SPLIT,
                    "sizes": {"train": len(tr), "validation": len(va), "test": len(te)}},
        "costs": {"margin_good": costs.margin_good, "loss_default": costs.loss_default,
                  "breakeven_pd": costs.breakeven_pd},
        "champion": champion_name,
        "threshold": threshold,
        "models": results,
        "baselines_test": baselines,
        "fairness_test": fairness_report,
        "drift": {"features": drift_reports, "score_psi": score_psi},
        "importance_mean_abs_contribution": importance.round(4).to_dict(),
        "lightgbm_trees": int(candidates["lightgbm"].estimator.best_iteration_ or 0),
    }

    paths = s.paths
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)
    reference = X_tr.sample(n=min(5000, len(X_tr)), random_state=s.seed)
    joblib.dump(
        {
            "model": champion,
            "challenger": candidates[challenger_name],
            "threshold": threshold,
            "costs": costs,
            "features": FEATURES,
            "reference": reference,
            "reference_pd": champion.predict_pd(reference),
            "trained_at": summary["trained_at"],
        },
        paths.bundle,
    )
    paths.metrics.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figures = plots.save_all(candidates, X_te, y[te], costs, threshold, importance, paths.reports)

    if use_mlflow:
        log_mlflow(summary, figures)
    summary["elapsed_s"] = round(time.time() - started, 1)
    return summary


def log_mlflow(summary: dict, figures: list) -> None:
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed; skipping experiment tracking")
        return
    mlflow.set_tracking_uri(SETTINGS.mlflow_uri)
    mlflow.set_experiment("credit-risk-scoring")
    for name, res in summary["models"].items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "threshold": res["threshold"], "champion": name == summary["champion"],
                               **summary["costs"]})
            for split_name in ("validation", "test"):
                r = res[split_name]
                mlflow.log_metrics({f"{split_name}_{k}": v for k, v in r.items() if isinstance(v, float)})
                mlflow.log_metrics({f"{split_name}_{k}": v for k, v in r["decision"].items()})
            if name == summary["champion"]:
                mlflow.log_artifact(str(SETTINGS.paths.metrics))
                for fig in figures:
                    mlflow.log_artifact(str(fig))


def _print_summary(s: dict) -> None:
    print(f"rows={s['dataset']['rows']} default_rate={s['dataset']['default_rate']:.3f} champion={s['champion']}")
    for name, res in s["models"].items():
        t = res["test"]
        print(f"  {name:20s} test AUC={t['auc']:.4f} KS={t['ks']:.3f} Brier={t['brier']:.4f} "
              f"ECE={t['ece']:.4f} (raw {t['ece_uncalibrated']:.4f}, {res['calibration']['method']}) "
              f"thr={res['threshold']:.2f} "
              f"profit={t['decision']['profit']:,.0f} ci95={t['ci95']}")
    for name, b in s["baselines_test"].items():
        print(f"  policy {name:18s} approve={b['approval_rate']:.1%} bad_rate={b['default_rate_approved']:.1%} "
              f"profit={b['profit']:,.0f}")
    for attr, rep in s["fairness_test"].items():
        print(f"  fairness {attr:10s} min AIR={rep['min_adverse_impact_ratio']:.3f} "
              f"max EO gap={rep['max_equal_opportunity_gap']:.3f}")
    print(f"  score PSI: {s['drift']['score_psi']}")
    print(f"  elapsed {s['elapsed_s']}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mlflow", action="store_true")
    _print_summary(main(use_mlflow=not parser.parse_args().no_mlflow))
