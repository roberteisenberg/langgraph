"""
Phase 2 Tools — Simulated fraud investigation tools.

Three tools the LLM can call to investigate orders:
- check_customer_history: account age, prior orders, fraud flags
- verify_shipping_address: residential vs warehouse, geo risk
- check_payment_pattern: spending anomalies, velocity
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

    # Match on address substrings (case-insensitive)
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

    # Match on email prefix (allows partial matching)
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


# Export all tools as a list for easy binding
all_tools = [check_customer_history, verify_shipping_address, check_payment_pattern]
