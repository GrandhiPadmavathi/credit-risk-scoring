"""Credit risk dashboard: score an applicant, inspect model quality, fairness and drift."""

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from creditrisk import data, drift  # noqa: E402
from creditrisk.config import SETTINGS  # noqa: E402
from creditrisk.features import FEATURE_LABELS, FEATURES  # noqa: E402
from creditrisk.letter import compose_letter  # noqa: E402
from creditrisk.profiles import PRESETS  # noqa: E402
from creditrisk.scoring import Scorer  # noqa: E402

st.set_page_config(page_title="Credit Risk Scoring", page_icon="💳", layout="wide")

MONTHS = ["Latest", "-1", "-2", "-3", "-4", "-5"]
STATUS = {-2: "No balance", -1: "Paid in full", 0: "Revolving", **{k: f"{k} month(s) late" for k in range(1, 9)}}


@st.cache_resource
def scorer() -> Scorer:
    return Scorer.load()


@st.cache_data
def summary() -> dict:
    return json.loads(SETTINGS.paths.metrics.read_text(encoding="utf-8"))


@st.cache_data
def raw_sample(n: int = 3000) -> pd.DataFrame | None:
    if not SETTINGS.paths.db.exists():
        return None
    return data.load_applicants().sample(n=n, random_state=7)


try:
    model, metrics = scorer(), summary()
except FileNotFoundError:
    st.error("No trained model found. Run `python -m creditrisk.train` first.")
    st.stop()

st.title("Credit Risk Scoring")
champ = metrics["models"][metrics["champion"]]["test"]
st.caption(
    f"Model: **{metrics['champion'].replace('_', ' ')}** · test AUC {champ['auc']:.3f} · "
    f"decline when PD ≥ **{model.threshold:.2f}** (profit-optimal) · trained {metrics['trained_at'][:10]}"
)

score_tab, perf_tab, fair_tab, drift_tab = st.tabs(["Score an applicant", "Model performance", "Fairness", "Drift"])

with score_tab:
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        preset = PRESETS[st.selectbox("Start from a profile", list(PRESETS))]
        limit = st.number_input("Credit limit (NT$)", 10000, 1000000, preset["limit_bal"], 10000)
        st.caption("Demographics are not model inputs; only the limit and statement history are used.")
        st.markdown("**Last six statements** (edit any cell)")
        table = st.data_editor(
            pd.DataFrame({"month": MONTHS, "status": preset["pay"], "bill": preset["bill"], "paid": preset["paid"]}),
            column_config={
                "month": st.column_config.TextColumn(disabled=True),
                "status": st.column_config.SelectboxColumn(options=list(STATUS), help="-2 no balance, -1 paid in "
                                                           "full, 0 revolving, 1-8 months overdue"),
                "bill": st.column_config.NumberColumn("bill (NT$)", format="%d"),
                "paid": st.column_config.NumberColumn("paid (NT$)", min_value=0, format="%d"),
            },
            hide_index=True,
            width="stretch",
            key=f"editor-{limit}-{preset['pay']}",
        )
    applicant = {"limit_bal": limit}
    for m in range(6):
        applicant[f"pay_{m + 1}"] = int(table["status"].iloc[m])
        applicant[f"bill_amt{m + 1}"] = float(table["bill"].iloc[m])
        applicant[f"pay_amt{m + 1}"] = float(table["paid"].iloc[m])
    result = model.score(pd.DataFrame([applicant]))[0]

    with right:
        approve = result["decision"] == "approve"
        a, b, c = st.columns(3)
        a.metric("Probability of default", f"{result['probability_of_default']:.1%}")
        b.metric("Risk band", result["risk_band"])
        c.metric("Expected profit", f"NT${result['expected_profit']:,.0f}")
        (st.success if approve else st.error)(
            f"**{'APPROVE' if approve else 'DECLINE'}**: PD {'below' if approve else 'at or above'} the "
            f"{model.threshold:.0%} cut-off"
        )
        reasons = result["reasons"]["adverse"] + result["reasons"]["favourable"]
        if reasons:
            chart_df = pd.DataFrame(
                {"reason": [r["text"] for r in reasons], "impact": [r["impact"] for r in reasons],
                 "direction": ["raises risk" if r["impact"] > 0 else "lowers risk" for r in reasons]}
            )
            st.markdown("**Why** (contribution to the risk score)")
            st.altair_chart(
                alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X("impact:Q", title="← lowers risk · raises risk →"),
                    y=alt.Y("reason:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
                    color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                                   range=["#dc2626", "#16a34a"]), legend=None),
                ).properties(height=34 * len(chart_df)),
                width="stretch",
            )
        if st.button("Draft customer letter"):
            with st.spinner("Writing letter from the reason codes..."):
                letter = compose_letter(result)
            st.text_area("Letter", letter["text"], height=220)
            st.caption(f"Source: {letter['source']}" + (f" ({letter['note']})" if letter.get("note") else ""))

with perf_tab:
    rows = []
    for name, res in metrics["models"].items():
        t = res["test"]
        rows.append({"model": name.replace("_", " "), "AUC": t["auc"], "Gini": t["gini"], "KS": t["ks"],
                     "Brier": t["brier"], "ECE (calibrated)": t["ece"], "ECE (raw)": t["ece_uncalibrated"],
                     "threshold": res["threshold"], "profit (NT$ M)": t["decision"]["profit"] / 1e6})
    st.subheader("Test-set metrics")
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
    st.subheader("Approval policy comparison (champion model, test set)")
    pol = pd.DataFrame(
        [{"policy": k.replace("_", " "), "approval rate": v["approval_rate"], "bad rate of approved":
          v["default_rate_approved"], "good customers declined": v["good_customers_declined"],
          "profit (NT$ M)": v["profit"] / 1e6} for k, v in metrics["baselines_test"].items()]
    )
    st.dataframe(pol.round(3), hide_index=True, width="stretch")
    cols = st.columns(2)
    for i, name in enumerate(["profit_curve", "calibration", "roc", "importance"]):
        path = SETTINGS.paths.reports / f"{name}.png"
        if path.exists():
            cols[i % 2].image(str(path), width="stretch")

with fair_tab:
    st.markdown(
        "Sex, age and marital status are **not model inputs**. This audit checks whether decisions still differ "
        "across those groups. *Adverse impact ratio* = group approval rate / highest group approval rate "
        "(below 0.80 fails the four-fifths rule). *Good-customer approval* = share of customers who actually "
        "repaid that were approved (equal opportunity)."
    )
    for attr, rep in metrics["fairness_test"].items():
        df = pd.DataFrame(rep["groups"])
        st.subheader(attr.replace("_", " ").title())
        a, b = st.columns([1.4, 1])
        a.dataframe(df[["group", "n", "default_rate", "mean_pd", "approval_rate", "good_customer_approval",
                        "adverse_impact_ratio", "small_sample"]].round(3), hide_index=True, width="stretch")
        if rep.get("excluded_small_groups"):
            small = ", ".join(rep["excluded_small_groups"])
            a.caption(f"Fewer than 100 people, excluded from the summary gaps: {small}")
        b.altair_chart(
            alt.Chart(df).mark_bar().encode(
                x=alt.X("adverse_impact_ratio:Q", scale=alt.Scale(domain=[0, 1.05]), title="Adverse impact ratio"),
                y=alt.Y("group:N", title=None),
                color=alt.condition(alt.datum.adverse_impact_ratio < 0.8, alt.value("#dc2626"), alt.value("#2563eb")),
            ).properties(height=40 * len(df))
            + alt.Chart(pd.DataFrame({"x": [0.8]})).mark_rule(strokeDash=[4, 4], color="#6b7280").encode(x="x:Q"),
            width="stretch",
        )

with drift_tab:
    st.markdown("Population Stability Index (PSI) against the training population: below 0.10 is stable, "
                "0.10-0.25 moderate, above 0.25 significant.")
    raw = raw_sample()
    if raw is None:
        st.info("Raw data not available; showing the drift results recorded at training time.")
        report = pd.DataFrame(metrics["drift"]["features"]["stress_vs_train"])
        score_psi = metrics["drift"]["score_psi"]["stress_vs_train"]
    else:
        severity = st.slider("Simulated downturn severity", 0.0, 0.6, 0.3, 0.05,
                             help="Share of customers slipping one month behind; balances grow and payments shrink")
        current = model.features(drift.simulate_stress(raw, seed=1, severity=severity))
        report = pd.DataFrame(drift.drift_report(model.bundle["reference"], current, FEATURES))
        score_psi = drift.psi(model.bundle["reference_pd"], model.model.predict_pd(current))
        a, b = st.columns(2)
        a.metric("Score PSI", f"{score_psi:.3f}", drift.status(score_psi), delta_color="off")
        b.metric("Mean PD", f"{model.model.predict_pd(current).mean():.1%}",
                 f"{model.model.predict_pd(current).mean() - model.bundle['reference_pd'].mean():+.1%} vs training",
                 delta_color="inverse")
    report["label"] = report["feature"].map(FEATURE_LABELS)
    st.altair_chart(
        alt.Chart(report).mark_bar().encode(
            x=alt.X("psi:Q", title="PSI"),
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)),
            color=alt.Color("status:N", scale=alt.Scale(domain=["stable", "moderate", "significant"],
                                                        range=["#16a34a", "#f59e0b", "#dc2626"])),
        ).properties(height=26 * len(report)),
        width="stretch",
    )
