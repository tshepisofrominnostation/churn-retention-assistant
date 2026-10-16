"""Generative AI Retention Assistant.

Drafts a personalised, compliant retention email for a high-risk customer.
The LLM call is optional: without an API key the module falls back to a
deterministic, rule-based template so the prototype always works offline.

Environment (.env):
    LLM_API_KEY   - secret key (OpenAI-compatible endpoint). Never hardcoded.
    LLM_BASE_URL  - default https://api.openai.com/v1
    LLM_MODEL     - default gpt-4o-mini
"""
from __future__ import annotations

import json
import os
from typing import Dict

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional; raw env vars still work
    pass

from src.predict import risk_level

# Guardrail: the LLM may only offer what the bank has actually approved.
APPROVED_OFFERS = [
    "a free one-on-one account health review with a banker",
    "a no-fee account tier review that may lower monthly fees",
    "priority support routing for the next 90 days",
    "a free financial wellness check-in session",
]

# Guardrail: words that imply guarantees we cannot make (hallucination risk).
PROMISE_BLOCKLIST = [
    "guarantee", "guaranteed", "promise", "promised", "approved already",
    "loan approval", "credit approved", "interest-free", "risk-free",
    "you qualify", "you will qualify", "100%",
]


def infer_churn_reasons(profile: Dict[str, float]) -> list[str]:
    """Infer behavioural churn drivers from the profile, most severe first."""
    reasons = []
    if profile.get("Support_Tickets_30D", 0) >= 3:
        reasons.append("repeated support friction - "
                       f"{int(profile['Support_Tickets_30D'])} tickets in the last 30 days")
    if profile.get("Is_Active_Member", 1) == 0:
        reasons.append("low engagement - marked inactive and few logins")
    if profile.get("Transaction_Count_30D", 99) <= 3:
        reasons.append("dropping usage - very few transactions in the last 30 days")
    if profile.get("Tenure_Months", 99) <= 6:
        reasons.append("short tenure - relationship is still young")
    if profile.get("Num_Products", 2) == 1:
        reasons.append("single-product relationship with little switching cost")
    return reasons or ["mixed signals - no single dominant driver"]


def check_compliance(text: str) -> tuple[bool, list[str]]:
    """Return (is_compliant, violations) for a retention draft.

    Blocks language that implies financial guarantees the bank has not made.
    """
    lower = text.lower()
    hits = [w for w in PROMISE_BLOCKLIST if w in lower]
    return (len(hits) == 0, hits)


def _build_prompt(profile: Dict[str, float], churn_prob: float,
                  reasons: list[str]) -> list[dict]:
    """Assemble the system + user messages for the retention email."""
    system = (
        "You are a retention copywriter for a South African retail bank. "
        "Write a short, warm, personalised retention email (max 180 words). "
        "HARD RULES: 1) Only reference offers from the approved list. "
        "2) Never guarantee loan approvals, credit, returns or interest rates. "
        "3) No fabricated numbers, discounts or claims. "
        "4) Empathetic and respectful; acknowledge the customer's frustration. "
        "5) Include a clear single call to action. "
        "6) Plain language, no jargon. Output only the email body text."
    )
    user = json.dumps({
        "customer_segment": {
            "age": profile.get("Age"),
            "tenure_months": profile.get("Tenure_Months"),
            "num_products": profile.get("Num_Products"),
            "has_credit_card": bool(profile.get("Has_Credit_Card")),
            "is_active_member": bool(profile.get("Is_Active_Member")),
            "transactions_last_30d": profile.get("Transaction_Count_30D"),
            "support_tickets_last_30d": profile.get("Support_Tickets_30D"),
        },
        "churn_risk_pct": round(churn_prob * 100, 1),
        "inferred_churn_drivers": reasons,
        "approved_offers": APPROVED_OFFERS,
        "note": "PII minimisation: no names or account numbers are provided. "
                "Address the customer as 'Valued customer'.",
    }, indent=2)
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def _llm_draft(messages: list[dict]) -> str:
    """Call an OpenAI-compatible chat endpoint. Returns the email body."""
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        raise RuntimeError("LLM_API_KEY not set")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": messages, "temperature": 0.7,
              "max_tokens": 400},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _template_draft(profile: Dict[str, float], reasons: list[str]) -> str:
    """Deterministic fallback draft used when no LLM key is configured."""
    tickets = int(profile.get("Support_Tickets_30D", 0))
    if tickets >= 3:
        empathy = ("We have noticed that reaching us has been harder than it "
                   "should have been recently, and we are sorry about that.")
        offer = APPROVED_OFFERS[2]                      # priority support routing
    elif not profile.get("Is_Active_Member", 1) or profile.get("Transaction_Count_30D", 0) <= 3:
        empathy = ("We have noticed your account has been quieter lately, and "
                   "we would hate to see you go without a proper conversation.")
        offer = APPROVED_OFFERS[0]                      # account health review
    else:
        empathy = "We value the time you have banked with us and want to keep it that way."
        offer = APPROVED_OFFERS[3]                      # wellness check-in
    drivers = "; ".join(reasons)
    return (
        "Subject: Let us make things right, Valued customer\n\n"
        f"{empathy}\n\n"
        f"Based on your recent account activity, our team would like to offer "
        f"{offer}. This is our way of understanding what is not working for "
        f"you and fixing it directly.\n\n"
        f"Would you be open to a 15-minute call this week? Reply to this email "
        f"or book any slot in the app under 'Book a banker'.\n\n"
        "Warm regards,\nYour Relationship Team\n\n"
        "(Internal note - inferred drivers: " + drivers + ")"
    )


def draft_retention_email(profile: Dict[str, float], churn_prob: float) -> dict:
    """Generate a personalised retention draft with compliance guardrails.

    Args:
        profile: The customer's behavioural feature dict.
        churn_prob: Model churn probability (0-1).

    Returns:
        {"email": str, "source": "llm" | "template_fallback",
         "compliant": bool, "violations": list, "reasons": list,
         "risk_level": str}
    """
    reasons = infer_churn_reasons(profile)
    result = {"reasons": reasons, "risk_level": risk_level(churn_prob)}
    source = "template_fallback"
    try:
        body = _llm_draft(_build_prompt(profile, churn_prob, reasons))
        source = "llm"
    except Exception:
        body = _template_draft(profile, reasons)

    compliant, violations = check_compliance(body)
    result.update({
        "email": body, "source": source,
        "compliant": compliant, "violations": violations,
    })
    return result
