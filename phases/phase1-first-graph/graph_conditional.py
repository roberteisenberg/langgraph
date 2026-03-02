"""
Phase 1B: Conditional Edges — Branching Fraud Scorer

After scoring, route to different nodes based on the score:
  START → parse_order → score_fraud → [route_decision]
                                        ├── score >= 80 → flag_order → END
                                        ├── score >= 50 → review_order → END
                                        └── score < 50  → approve_order → END

First branching logic. The LLM output drives the next step.
"""

import os
import sys
from typing import Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.state import FraudState

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)


# --- Nodes ---

def parse_order(state: FraudState) -> dict:
    """Validate and normalize the order."""
    order = state["order"]

    required = ["id", "customer_email", "amount", "items", "shipping_address"]
    missing = [f for f in required if f not in order]
    if missing:
        raise ValueError(f"Order missing required fields: {missing}")

    print(f"[parse_order] Order {order['id']}: ${order['amount']:.2f}, "
          f"{len(order['items'])} items")

    return {}


def score_fraud(state: FraudState) -> dict:
    """Call LLM to score fraud risk 0-100."""
    order = state["order"]

    prompt = f"""Score the fraud risk of this order from 0 (no risk) to 100 (certain fraud).

Order details:
- Order ID: {order['id']}
- Customer: {order.get('customer_name', 'Unknown')} ({order['customer_email']})
- Amount: ${order['amount']:.2f}
- Items: {', '.join(item['name'] for item in order['items'])}
- Shipping to: {order['shipping_address']}
- Account age: {order.get('metadata', {}).get('account_age_days', 'unknown')} days
- Prior orders: {order.get('metadata', {}).get('prior_orders', 'unknown')}
- Prior fraud flags: {order.get('metadata', {}).get('prior_fraud_flags', 'unknown')}

Respond with ONLY a JSON object, no other text:
{{"score": <0-100>, "reasoning": "<one sentence>"}}"""

    import time
    for attempt in range(3):
        try:
            response = llm.invoke(prompt)
            break
        except Exception as e:
            if attempt < 2 and "500" in str(e):
                print(f"[score_fraud] API error, retrying ({attempt + 1}/3)...")
                time.sleep(2)
            else:
                raise

    content = response.content.strip()

    import json
    try:
        result = json.loads(content)
        score = int(result["score"])
        reasoning = result["reasoning"]
    except (json.JSONDecodeError, KeyError, ValueError):
        import re
        numbers = re.findall(r'\d+', content)
        score = int(numbers[0]) if numbers else 50
        reasoning = content[:100]

    score = max(0, min(100, score))

    print(f"[score_fraud] Score: {score}/100 — {reasoning}")

    return {"risk_score": score}


def flag_order(state: FraudState) -> dict:
    """Handle high-risk orders (score >= 80)."""
    order = state["order"]
    print(f"[flag_order] 🚨 Order {order['id']} FLAGGED — "
          f"score {state['risk_score']}, blocking shipment")
    return {"decision": "reject"}


def review_order(state: FraudState) -> dict:
    """Handle medium-risk orders (50 <= score < 80)."""
    order = state["order"]
    print(f"[review_order] ⚠️  Order {order['id']} sent for REVIEW — "
          f"score {state['risk_score']}, holding for analyst")
    return {"decision": "review"}


def approve_order(state: FraudState) -> dict:
    """Handle low-risk orders (score < 50)."""
    order = state["order"]
    print(f"[approve_order] ✅ Order {order['id']} APPROVED — "
          f"score {state['risk_score']}")
    return {"decision": "approve"}


# --- Routing ---

def route_decision(state: FraudState) -> Literal["flag_order", "review_order", "approve_order"]:
    """Route based on risk score. LLM output drives the next step."""
    score = state["risk_score"]
    if score >= 80:
        return "flag_order"
    elif score >= 50:
        return "review_order"
    else:
        return "approve_order"


# --- Build Graph ---

def build_graph():
    """Build the Phase 1B graph with conditional edges."""
    graph = StateGraph(FraudState)

    # Add nodes
    graph.add_node("parse_order", parse_order)
    graph.add_node("score_fraud", score_fraud)
    graph.add_node("flag_order", flag_order)
    graph.add_node("review_order", review_order)
    graph.add_node("approve_order", approve_order)

    # Edges
    graph.add_edge(START, "parse_order")
    graph.add_edge("parse_order", "score_fraud")

    # Conditional edge — route based on score
    graph.add_conditional_edges(
        "score_fraud",
        route_decision,
        {
            "flag_order": "flag_order",
            "review_order": "review_order",
            "approve_order": "approve_order",
        },
    )

    # All decision nodes → END
    graph.add_edge("flag_order", END)
    graph.add_edge("review_order", END)
    graph.add_edge("approve_order", END)

    return graph.compile()


# --- Run ---

if __name__ == "__main__":
    from shared.test_cases import ALL_CASES

    app = build_graph()

    print("=" * 60)
    print("Phase 1B: Conditional Edges — Branching Fraud Scorer")
    print("=" * 60)

    for name, order in ALL_CASES:
        print(f"\n{'─' * 50}")
        print(f"  {name}")
        print(f"{'─' * 50}")

        result = app.invoke({"order": order, "risk_score": 0, "decision": ""})

        print(f"  → Routed to: {result['decision'].upper()} "
              f"(score: {result['risk_score']})")

    print(f"\n{'=' * 60}")
    print("Done. All 6 cases processed with conditional routing.")
    print("=" * 60)
