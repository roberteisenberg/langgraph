"""
Phase 0: Baseline — A Python Function Calls an LLM

No LangGraph. No framework. A function takes an order, calls the LLM,
parses the response, and returns a score and decision.

This is the starting point. Everything that follows in Phases 1-6
builds on this foundation — but this is all you need for a single
LLM call.

Run: python3 phases/phase0-baseline/score_fraud.py
"""

import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# Add project root to path for shared imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.test_cases import ALL_CASES


# --- The LLM ---

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)


# --- The Function ---

def score_fraud(order: dict) -> dict:
    """Score an order's fraud risk. One function, one LLM call.

    Takes an order dict, builds a prompt, calls the LLM, parses the
    response, and returns a score (0-100) with a decision
    (approve/review/reject).
    """
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

    # Call the LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content

    # Parse the response
    try:
        json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', content)
        if json_match:
            result = json.loads(json_match.group())
            score = int(result["score"])
            reasoning = result.get("reasoning", "")
        else:
            raise ValueError("No JSON found")
    except (json.JSONDecodeError, KeyError, ValueError):
        numbers = re.findall(r'\d+', content)
        score = int(numbers[0]) if numbers else 50
        reasoning = content[:100]

    score = max(0, min(100, score))

    # Decision thresholds
    if score >= 80:
        decision = "reject"
    elif score >= 50:
        decision = "review"
    else:
        decision = "approve"

    return {
        "score": score,
        "decision": decision,
        "reasoning": reasoning,
    }


# --- Run ---

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 0: Baseline — A Python Function Calls an LLM")
    print("=" * 60)

    results = []
    for name, order in ALL_CASES:
        print(f"\n{'─' * 50}")
        print(f"  {name}")
        print(f"{'─' * 50}")

        result = score_fraud(order)
        results.append((name, result))

        print(f"  Score:     {result['score']}/100")
        print(f"  Decision:  {result['decision'].upper()}")
        print(f"  Reasoning: {result['reasoning']}")

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"  Summary")
    print(f"{'=' * 60}")
    print(f"  {'Case':<30} {'Score':>6} {'Decision':>10}")
    print(f"  {'-'*30} {'-'*6} {'-'*10}")
    for name, result in results:
        print(f"  {name:<30} {result['score']:>6} {result['decision'].upper():>10}")
    print(f"{'=' * 60}")
    print()
    print("  This is all you need for a single LLM call.")
    print("  Phase 1 wraps this in LangGraph to learn the framework.")
    print("  Phase 2 adds tools and the loop — that's when LangGraph earns its keep.")
    print()
