"""
Phase 3 Tools — 5 investigation tools + deterministic scoring.

Carries 3 tools from Phase 2 (same simulated data):
- check_customer_history
- verify_shipping_address
- check_payment_pattern

Adds 2 new tools:
- search_fraud_database — cross-reference against known fraud records
- calculate_risk_score — LLM-callable tool that triggers deterministic scoring
"""

from langchain_core.tools import tool


@tool
def check_customer_history(email: str) -> dict:
    """Look up a customer's order history, account age, and prior fraud flags by email."""

    data = {
        "alice.johnson@gmail.com": {
            "account_age_days": 1825,
            "prior_orders": 50,
            "fraud_flags": 0,
            "risk_level": "low",
        },
        "newbuyer_2026@outlook.com": {
            "account_age_days": 12,
            "prior_orders": 0,
            "fraud_flags": 0,
            "risk_level": "medium",
            "note": "New account, no purchase history",
        },
        "xuser8819@tempmail.ninja": {
            "account_age_days": 2,
            "prior_orders": 0,
            "fraud_flags": 0,
            "risk_level": "high",
            "note": "Disposable email provider detected",
        },
        "robert.chen@company.com": {
            "account_age_days": 730,
            "prior_orders": 30,
            "fraud_flags": 0,
            "risk_level": "low",
        },
        "pat.williams@yahoo.com": {
            "account_age_days": 365,
            "prior_orders": 8,
            "fraud_flags": 1,
            "last_flag_days_ago": 180,
            "risk_level": "medium",
            "note": "Previous fraud flag 6 months ago",
        },
        "sam.taylor@gmail.com": {
            "account_age_days": 200,
            "prior_orders": 5,
            "fraud_flags": 0,
            "risk_level": "low",
        },
    }

    if email in data:
        return data[email]

    return {"account_age_days": "unknown", "risk_level": "unknown"}


@tool
def verify_shipping_address(address: str) -> dict:
    """Verify a shipping address — checks if residential, commercial, warehouse, or freight forwarder. Also checks if the address is shared by other accounts."""

    addr_lower = address.lower()

    if "evergreen" in addr_lower:
        return {"type": "residential", "geo_risk": "low"}

    if "lake shore" in addr_lower:
        return {"type": "apartment", "geo_risk": "low"}

    if "industrial" in addr_lower:
        return {
            "type": "warehouse/freight_forwarder",
            "geo_risk": "high",
            "note": "Known freight forwarding facility",
        }

    if "commerce way" in addr_lower:
        return {
            "type": "warehouse",
            "geo_risk": "high",
            "shared_by_accounts": 3,
            "note": "Address shared by 3 accounts this week",
        }

    if "baker" in addr_lower:
        return {"type": "residential", "geo_risk": "low"}

    if "oak avenue" in addr_lower:
        return {"type": "residential", "geo_risk": "low"}

    return {"type": "unknown", "geo_risk": "medium"}


@tool
def check_payment_pattern(email: str, amount: float) -> dict:
    """Check if the order amount is typical for this customer. Analyzes spending history and flags anomalies."""

    if email.startswith("alice.johnson@"):
        return {
            "typical_range_min": 20,
            "typical_range_max": 80,
            "current_amount": amount,
            "anomaly": False,
            "velocity": "normal",
        }

    if email.startswith("newbuyer_2026@"):
        return {
            "typical_range": "no history",
            "current_amount": amount,
            "anomaly": "unknown",
            "velocity": "first_order",
            "note": "No purchase history to compare",
        }

    if email.startswith("xuser8819@"):
        return {
            "typical_range": "no history",
            "current_amount": amount,
            "anomaly": "unknown",
            "velocity": "first_order",
        }

    if email.startswith("robert.chen@"):
        return {
            "typical_range_min": 200,
            "typical_range_max": 800,
            "current_amount": amount,
            "anomaly": True,
            "velocity": "normal",
            "note": "Amount is 1.9x above typical maximum",
        }

    if email.startswith("pat.williams@"):
        return {
            "typical_range_min": 30,
            "typical_range_max": 100,
            "current_amount": amount,
            "anomaly": False,
            "velocity": "normal",
        }

    if email.startswith("sam.taylor@"):
        return {
            "error": "Payment processing service unavailable",
            "risk_level": "unknown",
        }

    return {
        "typical_range": "no history",
        "current_amount": amount,
        "anomaly": "unknown",
        "velocity": "unknown",
    }


@tool
def search_fraud_database(indicator: str) -> dict:
    """Search the fraud database for a given indicator (email, address pattern, or other identifier). Returns matching fraud records if any exist."""

    indicator_lower = indicator.lower()

    if "tempmail.ninja" in indicator_lower:
        return {
            "found": True,
            "matches": 14,
            "indicator_type": "email_domain",
            "details": "Disposable email provider frequently used in fraud",
            "fraud_rate": 0.89,
            "risk_level": "high",
        }

    if "industrial" in indicator_lower and "hayward" in indicator_lower:
        return {
            "found": True,
            "matches": 6,
            "indicator_type": "address",
            "details": "Known freight forwarder/reshipping facility",
            "fraud_rate": 0.72,
            "risk_level": "high",
        }

    if "commerce way" in indicator_lower and "reno" in indicator_lower:
        return {
            "found": True,
            "matches": 2,
            "indicator_type": "address",
            "details": "Shared warehouse address",
            "fraud_rate": 0.35,
            "risk_level": "medium",
        }

    if "pat.williams@yahoo.com" in indicator_lower:
        return {
            "found": True,
            "matches": 1,
            "indicator_type": "email",
            "details": "Prior flag resolved as false_positive",
            "fraud_rate": 0.0,
            "risk_level": "low",
        }

    return {
        "found": False,
        "matches": 0,
        "details": "No fraud records found",
    }


@tool
def calculate_risk_score(evidence_summary: str) -> str:
    """Finalize the investigation by calculating the deterministic risk score based on all gathered evidence. Call this when you have collected enough evidence. The evidence_summary parameter is a brief summary of findings (the actual scoring uses the structured evidence records)."""
    # This is a placeholder — execute_tools intercepts this and calls
    # _calculate_risk_score_impl with the real evidence from state.
    return "Risk score calculated. Investigation complete."


def _calculate_risk_score_impl(evidence: list[dict]) -> dict:
    """Deterministic scoring based on accumulated evidence.

    Returns dict with risk_score, decision, and investigation_complete.
    """
    weights = {
        "high_risk": 30,
        "medium_risk": 15,
        "low_risk": -10,
        "neutral": 0,
        "error": 5,
    }

    base_score = 20

    score = base_score
    has_high_confidence_high_risk = False

    for e in evidence:
        signal = e.get("risk_signal", "neutral")
        confidence = e.get("confidence", 0.5)
        weight = weights.get(signal, 0)
        score += weight * confidence

        if signal == "high_risk" and confidence > 0.8:
            has_high_confidence_high_risk = True

    # Bonus for high-confidence high-risk signals
    if has_high_confidence_high_risk:
        score += 10

    # Clamp 0-100
    score = max(0, min(100, int(round(score))))

    # Decision thresholds
    if score >= 80:
        decision = "reject"
    elif score >= 50:
        decision = "review"
    else:
        decision = "approve"

    return {
        "risk_score": score,
        "decision": decision,
        "investigation_complete": True,
    }


# All tools the LLM can call
all_tools = [
    check_customer_history,
    verify_shipping_address,
    check_payment_pattern,
    search_fraud_database,
    calculate_risk_score,
]
