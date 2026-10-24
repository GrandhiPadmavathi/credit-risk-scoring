"""Customer-facing decision letters grounded in the computed reason codes.

The LLM only rewrites facts it is given. Any number in its output that is not in those facts makes the
letter fall back to the deterministic template, so the model cannot invent amounts, rates or reasons.
"""

import re

import requests

from creditrisk.config import SETTINGS

SYSTEM = (
    "You write short, plain-English decision letters for a credit card issuer. Use ONLY the facts provided. "
    "Do not invent numbers, policies, offers, rates or contact details. Do not mention models, scores or "
    "probabilities. For a decline, list the main reasons and say the customer may ask for a review. "
    "At most 120 words. Plain text, no greeting placeholder brackets."
)

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _facts(result: dict) -> list[str]:
    key = "adverse" if result["decision"] == "decline" else "favourable"
    return [r["text"] for r in result["reasons"][key]]


def template_letter(result: dict) -> str:
    facts = _facts(result)
    if result["decision"] == "decline":
        lines = ["We are unable to approve your credit application at this time.", "", "The main reasons were:"]
        lines += [f"- {f}" for f in facts] or ["- Your recent repayment history"]
        lines += ["", "You may request a review of this decision, and we encourage you to reapply once these "
                  "factors improve."]
    else:
        lines = ["Good news: your credit application has been approved.", ""]
        if facts:
            lines += ["Factors that supported this decision:"] + [f"- {f}" for f in facts]
        lines += ["", "Thank you for banking with us."]
    return "\n".join(lines)


def _grounded(text: str, facts: list[str]) -> bool:
    allowed = {n.replace(",", "") for f in facts for n in NUMBER.findall(f)}
    used = {n.replace(",", "").rstrip(".") for n in NUMBER.findall(text)}
    return used <= allowed


def compose_letter(result: dict, timeout: float = 60) -> dict:
    facts = _facts(result)
    if not SETTINGS.llm_enabled:
        return {"text": template_letter(result), "source": "template"}
    prompt = f"Decision: {result['decision'].upper()}\nFacts:\n" + "\n".join(f"- {f}" for f in facts)
    try:
        resp = requests.post(
            f"{SETTINGS.llm_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {SETTINGS.llm_api_key}"},
            json={
                "model": SETTINGS.llm_model,
                "temperature": 0.2,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, ValueError) as exc:
        return {"text": template_letter(result), "source": "template", "note": f"LLM unavailable: {type(exc).__name__}"}
    if not text or not _grounded(text, facts):
        return {"text": template_letter(result), "source": "template", "note": "LLM output failed grounding check"}
    return {"text": text, "source": "llm", "model": SETTINGS.llm_model}
