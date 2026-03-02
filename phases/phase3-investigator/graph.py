"""
Phase 3: The Investigator — Full ReAct Agent

The hinge phase: LLM investigates selectively, Python scores deterministically,
LangGraph manages the loop with structured evidence accumulation.

Graph: START → parse_order → call_llm → [should_continue]
                                          ├── has tool calls → execute_tools → call_llm (loop)
                                          └── investigation_complete or no tool calls → assess_risk → format_report → END

Key improvements over Phase 2:
- LLM picks tools selectively (not all 3 every time)
- Deterministic scoring via _calculate_risk_score_impl
- Structured Evidence trail for every tool call
- Custom execute_tools node (replaces prebuilt ToolNode)
- calculate_risk_score sets investigation_complete for clean termination
"""

import json
import operator
import os
import sys
import time
from datetime import datetime, timezone
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Add project root to path for shared imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from tools import (
    all_tools,
    check_customer_history,
    check_payment_pattern,
    verify_shipping_address,
    search_fraud_database,
    calculate_risk_score,
    _calculate_risk_score_impl,
)


# --- State ---

class Evidence(TypedDict):
    tool: str              # "check_customer_history", etc.
    finding: str           # "Address matches known freight forwarder warehouse"
    risk_signal: str       # "low_risk" | "medium_risk" | "high_risk" | "neutral" | "error"
    confidence: float      # 0.0-1.0
    raw_data: dict         # full tool output for debugging
    timestamp: str         # ISO 8601


class FraudStateV3(TypedDict):
    order: dict
    messages: Annotated[list, add_messages]
    evidence: Annotated[list[Evidence], operator.add]
    risk_score: int
    decision: str
    investigation_complete: bool


# --- LLM ---

SYSTEM_PROMPT = """You are a fraud investigation agent. Your job is to investigate orders for potential fraud by gathering evidence using your tools.

## Your Role

You INVESTIGATE. You do NOT score. You do NOT make final decisions.

When you have gathered enough evidence, call the `calculate_risk_score` tool. That tool computes the score and finalizes the investigation. Do not attempt to assign risk scores yourself.

## How to Investigate

1. Read the order details carefully: customer email, amount, items, shipping address.

2. Decide which checks are relevant. Not every order needs every tool:
   - New customer with no history? Check `check_customer_history` and `check_payment_pattern`.
   - Unusual shipping address? Check `verify_shipping_address`.
   - High amount from known customer? Check `check_payment_pattern`.
   - Any suspicious signal? Cross-reference with `search_fraud_database`.

3. After each tool result, reason about what you learned:
   - Does this signal increase or decrease suspicion?
   - Does this new information make another check more relevant?
   - Do you have conflicting signals that need resolution?

4. When you have enough evidence to form a complete picture, call `calculate_risk_score` with a brief summary of your findings.

## Critical Rules

- When calling tools, you MUST use the EXACT values from the order. Pass the exact shipping address string, the exact email, and the exact amount. Do not fabricate, modify, or round values.
- Only state what tools have actually returned. If you haven't checked something, leave it out of your reasoning.
- Every order needs at least one confirming check, even low-risk ones.
- Conflicting signals (e.g., long customer history but suspicious address) require additional cross-referencing before concluding.
- If a tool returns an error, record it and continue with other tools.
- Do not call the same tool twice with the same inputs.

## When to Stop Investigating

Call `calculate_risk_score` when:
- You have checked all signals you consider relevant
- Conflicting signals have been investigated with at least one additional cross-reference
- You have at least 2 pieces of evidence (no single-check investigations except for obviously routine orders from established customers)

Do not keep investigating after calling `calculate_risk_score`. That tool finalizes the investigation."""

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0).bind_tools(all_tools)

# Tool dispatch map
TOOL_MAP = {
    "check_customer_history": check_customer_history,
    "verify_shipping_address": verify_shipping_address,
    "check_payment_pattern": check_payment_pattern,
    "search_fraud_database": search_fraud_database,
    "calculate_risk_score": calculate_risk_score,
}


# --- Evidence Extraction ---

def _create_evidence(tool_name: str, tool_result: dict) -> Evidence:
    """Deterministic mapping from tool output to an Evidence record."""
    timestamp = datetime.now(timezone.utc).isoformat()

    # Handle error results from any tool
    if "error" in tool_result:
        return Evidence(
            tool=tool_name,
            finding=f"Tool error: {tool_result['error']}",
            risk_signal="error",
            confidence=0.5,
            raw_data=tool_result,
            timestamp=timestamp,
        )

    if tool_name == "check_customer_history":
        risk_level = tool_result.get("risk_level", "unknown")
        signal_map = {"low": "low_risk", "medium": "medium_risk", "high": "high_risk"}
        risk_signal = signal_map.get(risk_level, "medium_risk")

        prior_orders = tool_result.get("prior_orders", 0)
        if isinstance(prior_orders, int) and prior_orders > 10:
            confidence = 0.95
        elif isinstance(prior_orders, int) and prior_orders > 0:
            confidence = 0.8
        else:
            confidence = 0.7

        note = tool_result.get("note", "")
        finding = f"Customer history: {prior_orders} prior orders, risk_level={risk_level}"
        if note:
            finding += f". {note}"

        return Evidence(
            tool=tool_name,
            finding=finding,
            risk_signal=risk_signal,
            confidence=confidence,
            raw_data=tool_result,
            timestamp=timestamp,
        )

    if tool_name == "verify_shipping_address":
        geo_risk = tool_result.get("geo_risk", "medium")
        addr_type = tool_result.get("type", "unknown")

        signal_map = {"low": "low_risk", "medium": "medium_risk", "high": "high_risk"}
        risk_signal = signal_map.get(geo_risk, "medium_risk")

        if addr_type == "residential":
            confidence = 0.9
        elif addr_type in ("warehouse", "warehouse/freight_forwarder"):
            confidence = 0.95
        else:
            confidence = 0.5

        note = tool_result.get("note", "")
        finding = f"Address type: {addr_type}, geo_risk={geo_risk}"
        if note:
            finding += f". {note}"

        return Evidence(
            tool=tool_name,
            finding=finding,
            risk_signal=risk_signal,
            confidence=confidence,
            raw_data=tool_result,
            timestamp=timestamp,
        )

    if tool_name == "check_payment_pattern":
        anomaly = tool_result.get("anomaly")
        velocity = tool_result.get("velocity", "unknown")

        if anomaly is True:
            risk_signal = "medium_risk"
            confidence = 0.8
        elif anomaly is False:
            risk_signal = "low_risk"
            confidence = 0.85
        else:
            # anomaly is "unknown" or None
            risk_signal = "medium_risk"
            if velocity == "first_order":
                confidence = 0.5
            else:
                confidence = 0.5

        note = tool_result.get("note", "")
        finding = f"Payment pattern: anomaly={anomaly}, velocity={velocity}"
        if note:
            finding += f". {note}"

        return Evidence(
            tool=tool_name,
            finding=finding,
            risk_signal=risk_signal,
            confidence=confidence,
            raw_data=tool_result,
            timestamp=timestamp,
        )

    if tool_name == "search_fraud_database":
        found = tool_result.get("found", False)
        if not found:
            return Evidence(
                tool=tool_name,
                finding="No fraud records found for this indicator",
                risk_signal="neutral",
                confidence=0.7,
                raw_data=tool_result,
                timestamp=timestamp,
            )

        # Found records — use max risk level
        risk_level = tool_result.get("risk_level", "medium")
        signal_map = {"low": "low_risk", "medium": "medium_risk", "high": "high_risk"}
        risk_signal = signal_map.get(risk_level, "medium_risk")

        fraud_rate = tool_result.get("fraud_rate", 0.5)
        confidence = min(1.0, 0.5 + fraud_rate * 0.4)

        details = tool_result.get("details", "")
        matches = tool_result.get("matches", 0)
        finding = f"Fraud DB: {matches} matches found. {details}"

        return Evidence(
            tool=tool_name,
            finding=finding,
            risk_signal=risk_signal,
            confidence=confidence,
            raw_data=tool_result,
            timestamp=timestamp,
        )

    # Fallback for unknown tools
    return Evidence(
        tool=tool_name,
        finding=f"Unknown tool result: {json.dumps(tool_result)[:100]}",
        risk_signal="neutral",
        confidence=0.5,
        raw_data=tool_result,
        timestamp=timestamp,
    )


# --- Nodes ---

def parse_order(state: FraudStateV3) -> dict:
    """Validate required fields in the order."""
    order = state["order"]

    required = ["id", "customer_email", "amount", "items", "shipping_address"]
    missing = [f for f in required if f not in order]
    if missing:
        raise ValueError(f"Order missing required fields: {missing}")

    print(f"  [parse_order] Order {order['id']}: ${order['amount']:.2f}, "
          f"{len(order['items'])} items, to {order['shipping_address'][:40]}...")

    return {}


def call_llm(state: FraudStateV3) -> dict:
    """Call the LLM with system prompt + accumulated messages.

    First call: injects order details as HumanMessage.
    Subsequent calls: LLM sees prior reasoning + tool results.
    """
    messages = state["messages"]

    # First call — no messages yet, inject order details
    if not messages:
        order = state["order"]
        order_text = f"""Investigate this order for fraud.

IMPORTANT: When calling tools, use these EXACT values — do not modify or fabricate any data.

Order ID: {order['id']}
Customer Name: {order.get('customer_name', 'Unknown')}
Customer Email: {order['customer_email']}
Order Amount: {order['amount']}
Items: {', '.join(item['name'] for item in order['items'])}
Shipping Address: {order['shipping_address']}
Account Metadata: {json.dumps(order.get('metadata', {}), indent=2)}

Use the exact email "{order['customer_email']}" for customer lookups.
Use the exact address "{order['shipping_address']}" for address verification.
Use the exact amount {order['amount']} for payment checks."""

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


def execute_tools(state: FraudStateV3) -> dict:
    """Custom tool execution node.

    For each tool call in the last message:
    1. Dispatch to the actual tool function
    2. Create a structured Evidence record
    3. Return ToolMessage (for message history) + Evidence (for state)

    Special case: calculate_risk_score is intercepted and routed to
    _calculate_risk_score_impl with the real evidence.
    """
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    tool_messages = []
    new_evidence = []
    score_result = None

    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_call_id = tc["id"]

        # Validate tool args against actual order data — prevent LLM hallucination
        order = state["order"]
        if tool_name == "verify_shipping_address":
            tool_args = {"address": order["shipping_address"]}
        elif tool_name == "check_payment_pattern":
            tool_args = {"email": order["customer_email"], "amount": order["amount"]}
        elif tool_name == "check_customer_history":
            tool_args = {"email": order["customer_email"]}

        if tool_name == "calculate_risk_score":
            # Intercept: score using real evidence, not LLM's summary
            all_evidence = list(state.get("evidence", [])) + new_evidence
            score_result = _calculate_risk_score_impl(all_evidence)

            result_text = (
                f"Risk score calculated: {score_result['risk_score']}/100. "
                f"Decision: {score_result['decision']}. "
                f"Investigation complete."
            )
            tool_messages.append(
                ToolMessage(content=result_text, tool_call_id=tool_call_id)
            )
            continue

        # Normal tool execution
        tool_fn = TOOL_MAP.get(tool_name)
        if not tool_fn:
            error_msg = f"Unknown tool: {tool_name}"
            tool_messages.append(
                ToolMessage(content=error_msg, tool_call_id=tool_call_id)
            )
            new_evidence.append(Evidence(
                tool=tool_name,
                finding=error_msg,
                risk_signal="error",
                confidence=0.5,
                raw_data={"error": error_msg},
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
            continue

        try:
            result = tool_fn.invoke(tool_args)
            # tool.invoke returns the result directly (already deserialized)
            if isinstance(result, str):
                try:
                    result_data = json.loads(result)
                except json.JSONDecodeError:
                    result_data = {"raw": result}
            elif isinstance(result, dict):
                result_data = result
            else:
                result_data = {"raw": str(result)}

            # Create evidence record
            evidence = _create_evidence(tool_name, result_data)
            new_evidence.append(evidence)

            # Create tool message for LLM history
            result_text = json.dumps(result_data, indent=2)
            tool_messages.append(
                ToolMessage(content=result_text, tool_call_id=tool_call_id)
            )

        except Exception as e:
            error_data = {"error": str(e)}
            evidence = _create_evidence(tool_name, error_data)
            new_evidence.append(evidence)

            tool_messages.append(
                ToolMessage(content=f"Error: {e}", tool_call_id=tool_call_id)
            )

    # Build return state
    result = {
        "messages": tool_messages,
        "evidence": new_evidence,
    }

    if score_result:
        result["risk_score"] = score_result["risk_score"]
        result["decision"] = score_result["decision"]
        result["investigation_complete"] = score_result["investigation_complete"]

    return result


def assess_risk(state: FraudStateV3) -> dict:
    """Safety net: ensure scoring happened.

    If the LLM exited without calling calculate_risk_score,
    run scoring on whatever evidence exists.
    """
    if state.get("investigation_complete"):
        # Score already set by execute_tools
        return {}

    # Fallback: score with whatever evidence we have
    evidence = state.get("evidence", [])
    print(f"  [assess_risk] Fallback scoring on {len(evidence)} evidence items")
    result = _calculate_risk_score_impl(evidence)

    return {
        "risk_score": result["risk_score"],
        "decision": result["decision"],
        "investigation_complete": True,
    }


def format_report(state: FraudStateV3) -> dict:
    """Print a structured investigation report."""
    order = state["order"]
    evidence = state.get("evidence", [])

    print(f"\n  {'=' * 46}")
    print(f"  INVESTIGATION REPORT — Order {order['id']}")
    print(f"  {'=' * 46}")
    print(f"  Customer: {order.get('customer_name', 'Unknown')} ({order['customer_email']})")
    print(f"  Amount:   ${order['amount']:.2f}")
    print(f"  Decision: {state['decision'].upper()}")
    print(f"  Score:    {state['risk_score']}/100")
    print(f"  Evidence: {len(evidence)} items collected")

    for i, e in enumerate(evidence, 1):
        signal_icon = {
            "high_risk": "!!",
            "medium_risk": "! ",
            "low_risk": "ok",
            "neutral": "- ",
            "error": "??",
        }.get(e["risk_signal"], "  ")
        print(f"    {i}. [{signal_icon}] {e['tool']}: {e['finding']}")
        print(f"       Signal: {e['risk_signal']} | Confidence: {e['confidence']:.2f}")

    print(f"  {'=' * 46}")

    return {}


# --- Routing ---

def should_continue(state: FraudStateV3) -> Literal["execute_tools", "assess_risk"]:
    """Route after call_llm: continue investigating or wrap up."""
    if state.get("investigation_complete"):
        return "assess_risk"

    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tools"

    return "assess_risk"  # fallback: LLM stopped without calling calculate_risk_score


# --- Build Graph ---

def build_graph():
    """Build the Phase 3 graph with custom tool execution and evidence tracking."""
    graph = StateGraph(FraudStateV3)

    # Nodes
    graph.add_node("parse_order", parse_order)
    graph.add_node("call_llm", call_llm)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("assess_risk", assess_risk)
    graph.add_node("format_report", format_report)

    # Edges
    graph.add_edge(START, "parse_order")
    graph.add_edge("parse_order", "call_llm")
    graph.add_conditional_edges("call_llm", should_continue)
    graph.add_edge("execute_tools", "call_llm")
    graph.add_edge("assess_risk", "format_report")
    graph.add_edge("format_report", END)

    return graph.compile()


# --- Runner ---

def extract_tool_calls(messages) -> list[str]:
    """Extract tool names from the message history."""
    tool_names = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_names.append(tc["name"])
    return tool_names


def count_llm_calls(messages) -> int:
    """Count how many AI (assistant) messages are in the history."""
    count = 0
    for msg in messages:
        if hasattr(msg, "type") and msg.type == "ai":
            count += 1
    return count


if __name__ == "__main__":
    from shared.test_cases import ALL_CASES

    app = build_graph()

    print("=" * 60)
    print("Phase 3: The Investigator — Full ReAct Agent")
    print("=" * 60)

    for name, order in ALL_CASES:
        print(f"\n{'─' * 55}")
        print(f"  {name}")
        print(f"{'─' * 55}")

        result = app.invoke({
            "order": order,
            "risk_score": 0,
            "decision": "",
            "messages": [],
            "evidence": [],
            "investigation_complete": False,
        })

        tools_used = extract_tool_calls(result["messages"])
        llm_calls = count_llm_calls(result["messages"])
        evidence_count = len(result.get("evidence", []))

        print(f"\n  Summary:")
        print(f"    Tools called ({len(tools_used)}): {tools_used}")
        print(f"    LLM calls:    {llm_calls}")
        print(f"    Evidence:     {evidence_count} items")
        print(f"    → Final:      {result['decision'].upper()} "
              f"(score: {result['risk_score']})")

    print(f"\n{'=' * 60}")
    print("Done. All 6 cases processed.")
    print("=" * 60)
