"""
Phase 2: The Loop — Tools + First Agent

Graph with loop: START → call_llm → [tools_condition]
                                      ├── has tool calls → tools → call_llm (loop back)
                                      └── no tool calls → format_result → END

The LLM now has tools to investigate orders before scoring.
This is where LangGraph starts earning its keep — the loop.
"""

import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Add project root to path for shared imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from tools import all_tools


# --- State ---

class FraudStateV2(TypedDict):
    """Phase 2 state — adds messages for LLM ↔ tool conversation."""
    order: dict
    risk_score: int
    decision: str
    messages: Annotated[list, add_messages]


# --- LLM ---

SYSTEM_PROMPT = """You are a fraud investigation assistant. Analyze the following order for fraud risk.

You have tools to check customer history, verify shipping addresses, and check payment patterns.
Use the tools you think are relevant — not every order needs every tool.

After investigating, provide your final assessment as a JSON object:
{"score": <0-100>, "reasoning": "<brief explanation>", "decision": "<approve|review|reject>"}"""

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0).bind_tools(all_tools)


# --- Nodes ---

def call_llm(state: FraudStateV2) -> dict:
    """Call the LLM with system prompt + accumulated messages.

    First call: injects order details as a HumanMessage.
    Subsequent calls: LLM sees prior reasoning + tool results.
    """
    messages = state["messages"]

    # First call — no messages yet, inject order details
    if not messages:
        order = state["order"]
        order_text = f"""Investigate this order for fraud:

- Order ID: {order['id']}
- Customer: {order.get('customer_name', 'Unknown')} ({order['customer_email']})
- Amount: ${order['amount']:.2f}
- Items: {', '.join(item['name'] for item in order['items'])}
- Shipping to: {order['shipping_address']}
- Account metadata: {json.dumps(order.get('metadata', {}), indent=2)}"""

        messages = [HumanMessage(content=order_text)]

    # Retry on transient API errors
    for attempt in range(3):
        try:
            response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
            break
        except Exception as e:
            if attempt < 2 and "500" in str(e):
                print(f"  [call_llm] API error, retrying ({attempt + 1}/3)...")
                time.sleep(2)
            else:
                raise

    return {"messages": [response]}


def format_result(state: FraudStateV2) -> dict:
    """Parse the LLM's final message for score and decision."""
    # The last message is the LLM's final assessment (no tool calls)
    last_message = state["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    # Try to extract JSON from the response
    try:
        # Find JSON in the response (may be wrapped in text)
        json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', content)
        if json_match:
            result = json.loads(json_match.group())
            score = int(result["score"])
            decision = result.get("decision", "")
        else:
            raise ValueError("No JSON found")
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: extract first number as score
        numbers = re.findall(r'\d+', content)
        score = int(numbers[0]) if numbers else 50
        decision = ""

    score = max(0, min(100, score))  # clamp

    # Determine decision from score if not provided or invalid
    if decision not in ("approve", "review", "reject"):
        if score >= 80:
            decision = "reject"
        elif score >= 50:
            decision = "review"
        else:
            decision = "approve"

    return {"risk_score": score, "decision": decision}


# --- Build Graph ---

def build_graph():
    """Build the Phase 2 graph with tool loop."""
    graph = StateGraph(FraudStateV2)

    # Nodes
    graph.add_node("call_llm", call_llm)
    graph.add_node("tools", ToolNode(all_tools))
    graph.add_node("format_result", format_result)

    # Edges
    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges(
        "call_llm",
        tools_condition,
        {"tools": "tools", "__end__": "format_result"},
    )
    graph.add_edge("tools", "call_llm")
    graph.add_edge("format_result", END)

    return graph.compile()


# --- Run ---

def extract_tool_calls(messages) -> list[str]:
    """Extract tool names from the message history."""
    tool_names = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_names.append(tc["name"])
    return tool_names


if __name__ == "__main__":
    from shared.test_cases import ALL_CASES

    app = build_graph()

    print("=" * 60)
    print("Phase 2: The Loop — Tools + First Agent")
    print("=" * 60)

    for name, order in ALL_CASES:
        print(f"\n{'─' * 50}")
        print(f"  {name}")
        print(f"{'─' * 50}")

        result = app.invoke({
            "order": order,
            "risk_score": 0,
            "decision": "",
            "messages": [],
        })

        tools_used = extract_tool_calls(result["messages"])
        print(f"  Tools called: {tools_used if tools_used else '(none)'}")
        print(f"  → Final: {result['decision'].upper()} "
              f"(score: {result['risk_score']})")

    print(f"\n{'=' * 60}")
    print("Done. All 6 cases processed.")
    print("=" * 60)
