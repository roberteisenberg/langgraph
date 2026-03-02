"""
Phase 1: Your First Graph — Simple Fraud Scorer

A linear graph: START → parse_order → score_fraud → format_result → END

The score_fraud node calls an LLM to score risk 0-100.
This is deliberately simple — you don't need LangGraph for this.
But the graph structure pays off in Phase 3 when we add loops.
"""

import os
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END

# Add project root to path for shared imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.state import FraudState

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# LLM — used by score_fraud node
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)


# --- Nodes ---

def parse_order(state: FraudState) -> dict:
    """Validate and normalize the order. No LLM needed."""
    order = state["order"]

    # Basic validation
    required = ["id", "customer_email", "amount", "items", "shipping_address"]
    missing = [f for f in required if f not in order]
    if missing:
        raise ValueError(f"Order missing required fields: {missing}")

    print(f"[parse_order] Order {order['id']}: ${order['amount']:.2f}, "
          f"{len(order['items'])} items, to {order['shipping_address'][:40]}...")

    return {}  # no state changes — just validation


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

    # Retry on transient API errors
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

    # Parse the JSON response
    import json
    try:
        result = json.loads(content)
        score = int(result["score"])
        reasoning = result["reasoning"]
    except (json.JSONDecodeError, KeyError, ValueError):
        # If LLM doesn't follow format, extract number
        import re
        numbers = re.findall(r'\d+', content)
        score = int(numbers[0]) if numbers else 50
        reasoning = content[:100]

    score = max(0, min(100, score))  # clamp

    print(f"[score_fraud] Score: {score}/100 — {reasoning}")

    return {"risk_score": score}


def format_result(state: FraudState) -> dict:
    """Format the final decision based on score."""
    score = state["risk_score"]

    if score >= 80:
        decision = "reject"
    elif score >= 50:
        decision = "review"
    else:
        decision = "approve"

    order = state["order"]
    print(f"[format_result] Order {order['id']}: {decision.upper()} (score: {score})")

    return {"decision": decision}


# --- Build Graph ---

def build_graph():
    """Build the Phase 1 linear graph."""
    graph = StateGraph(FraudState)

    # Add nodes
    graph.add_node("parse_order", parse_order)
    graph.add_node("score_fraud", score_fraud)
    graph.add_node("format_result", format_result)

    # Linear edges: START → parse_order → score_fraud → format_result → END
    graph.add_edge(START, "parse_order")
    graph.add_edge("parse_order", "score_fraud")
    graph.add_edge("score_fraud", "format_result")
    graph.add_edge("format_result", END)

    return graph.compile()


# --- Run ---

if __name__ == "__main__":
    from shared.test_cases import ALL_CASES

    app = build_graph()

    print("=" * 60)
    print("Phase 1: Simple Fraud Scorer — Linear Graph")
    print("=" * 60)

    for name, order in ALL_CASES:
        print(f"\n{'─' * 50}")
        print(f"  {name}")
        print(f"{'─' * 50}")

        result = app.invoke({"order": order, "risk_score": 0, "decision": ""})

        print(f"  → Final: {result['decision'].upper()} "
              f"(score: {result['risk_score']})")

    print(f"\n{'=' * 60}")
    print("Done. All 6 cases processed.")
    print("=" * 60)
