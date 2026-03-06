"""
Phase 5: Real Workflows — Checkpointing + Human-in-the-Loop

Three capabilities built on one mechanism (checkpointed state):
1. Checkpointing  — save/inspect/resume graph state ("save game")
2. Human-in-the-Loop — pause for human review on ambiguous cases, resume with decision
3. State forking  — "what-if" analysis by forking from a checkpoint

Graph: START -> parse_order -> call_llm -> [should_continue]
                                            |-- has tool calls -> execute_tools -> call_llm (loop)
                                            +-- done -> assess_risk -> [route_after_assessment]
                                                                        |-- review -> human_review (INTERRUPT) -> format_report -> END
                                                                        +-- approve/reject -> format_report -> END

Only "review" decisions (score 50-79) trigger HITL. Among the 6 test cases,
only Case 4 (conflicting signals, score 61) pauses for human review.
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
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

# Add project root to path for shared imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# Add phase3 directory for tools import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase3-investigator"))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# LangSmith tracing: only enable if API key is actually set.
if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true" and not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ.pop("LANGCHAIN_TRACING_V2", None)

from tools import (
    all_tools,
    check_customer_history,
    check_payment_pattern,
    verify_shipping_address,
    search_fraud_database,
    calculate_risk_score,
    _calculate_risk_score_impl,
)


# --- Configuration ---

DEFAULT_CONFIG = {
    "max_loops": 10,
    "token_budget": 100_000,
    "recursion_limit": 25,
}


def make_config(thread_id: str) -> dict:
    """Create a run config with a thread_id for checkpointing."""
    return {
        "recursion_limit": DEFAULT_CONFIG["recursion_limit"],
        "configurable": {
            "thread_id": thread_id,
            "max_loops": DEFAULT_CONFIG["max_loops"],
            "token_budget": DEFAULT_CONFIG["token_budget"],
        },
    }


# --- State ---

class Evidence(TypedDict):
    tool: str              # "check_customer_history", etc.
    finding: str           # "Address matches known freight forwarder warehouse"
    risk_signal: str       # "low_risk" | "medium_risk" | "high_risk" | "neutral" | "error"
    confidence: float      # 0.0-1.0
    raw_data: dict         # full tool output for debugging
    timestamp: str         # ISO 8601


class FraudStateV5(TypedDict):
    # --- carried from Phase 4 ---
    order: dict
    messages: Annotated[list, add_messages]
    evidence: Annotated[list[Evidence], operator.add]
    risk_score: int
    decision: str
    investigation_complete: bool
    loop_count: int
    tokens_used: int
    guardrail_triggered: str
    # --- new in Phase 5 ---
    human_decision: str       # "" | "approved" | "rejected"
    human_notes: str          # analyst comments


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


# --- Evidence Extraction (same as Phase 3/4) ---

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

def parse_order(state: FraudStateV5) -> dict:
    """Validate required fields in the order."""
    order = state["order"]

    required = ["id", "customer_email", "amount", "items", "shipping_address"]
    missing = [f for f in required if f not in order]
    if missing:
        raise ValueError(f"Order missing required fields: {missing}")

    print(f"  [parse_order] Order {order['id']}: ${order['amount']:.2f}, "
          f"{len(order['items'])} items, to {order['shipping_address'][:40]}...")

    return {}


def call_llm(state: FraudStateV5, config: RunnableConfig) -> dict:
    """Call the LLM with loop counting and token budget enforcement."""
    configurable = config.get("configurable", {})
    token_budget = configurable.get("token_budget", DEFAULT_CONFIG["token_budget"])

    loop_count = state.get("loop_count", 0) + 1
    tokens_used = state.get("tokens_used", 0)

    # Token budget circuit breaker — check BEFORE calling the API
    if tokens_used >= token_budget:
        print(f"  [call_llm] Token budget exceeded ({tokens_used:,}/{token_budget:,}) "
              f"at loop {loop_count} — forcing exit")
        return {
            "loop_count": loop_count,
            "guardrail_triggered": "token_budget",
            "investigation_complete": True,
        }

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

    # Extract token usage from response metadata
    new_tokens = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        meta = response.usage_metadata
        new_tokens = meta.get("input_tokens", 0) + meta.get("output_tokens", 0)

    return {
        "messages": [response],
        "loop_count": loop_count,
        "tokens_used": tokens_used + new_tokens,
    }


def execute_tools(state: FraudStateV5) -> dict:
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


def assess_risk(state: FraudStateV5, config: RunnableConfig) -> dict:
    """Safety net scoring with guardrail detection.

    Three paths:
    1. Normal completion (investigation_complete=True, no guardrail) — score already set, return empty.
    2. Token budget fired (investigation_complete=True, guardrail="token_budget") — run scoring.
    3. Dead-end detected (loop_count >= max_loops) — set guardrail, run scoring.
    """
    configurable = config.get("configurable", {})
    max_loops = configurable.get("max_loops", DEFAULT_CONFIG["max_loops"])

    guardrail = state.get("guardrail_triggered", "")

    # Detect dead-end: loop limit hit without normal completion
    if not guardrail and state.get("loop_count", 0) >= max_loops:
        guardrail = "dead_end"

    # Normal completion — score already set by execute_tools
    if state.get("investigation_complete") and not guardrail:
        return {}

    # Score with whatever evidence exists
    evidence = state.get("evidence", [])
    label = f"Guardrail: {guardrail} — scoring" if guardrail else "Fallback scoring"
    print(f"  [assess_risk] {label} on {len(evidence)} evidence items")
    result = _calculate_risk_score_impl(evidence)

    # Annotate decision with guardrail reason
    decision = result["decision"]
    if guardrail:
        decision = f"{decision} ({guardrail})"

    update = {
        "risk_score": result["risk_score"],
        "decision": decision,
        "investigation_complete": True,
    }
    if guardrail:
        update["guardrail_triggered"] = guardrail

    return update


def human_review(state: FraudStateV5) -> dict:
    """Pause for human review on ambiguous cases.

    Critical: NO side effects before interrupt(). The entire node re-executes
    on resume. Any API calls before interrupt() would run twice.
    """
    # Prepare review package (idempotent — safe to re-execute)
    review_package = {
        "order_id": state["order"]["id"],
        "customer": state["order"].get("customer_name", "Unknown"),
        "amount": state["order"]["amount"],
        "risk_score": state["risk_score"],
        "decision": state["decision"],
        "evidence_summary": [
            f"[{e['risk_signal']}] {e['tool']}: {e['finding']}"
            for e in state.get("evidence", [])
        ],
        "action_required": "Review this investigation and provide your decision.",
        "options": ["approved", "rejected"],
    }

    # INTERRUPT — graph pauses here, review_package returned to caller
    human_input = interrupt(review_package)

    # --- Everything below runs AFTER resume ---
    human_decision = human_input.get("decision", "approved")
    human_notes = human_input.get("notes", "")

    result = {"human_decision": human_decision, "human_notes": human_notes}

    if human_decision == "rejected":
        result["decision"] = "reject"
    # "approved" = original "review" decision stands, human confirmed it

    return result


def format_report(state: FraudStateV5) -> dict:
    """Print a structured investigation report with HITL section."""
    order = state["order"]
    evidence = state.get("evidence", [])
    guardrail = state.get("guardrail_triggered", "")
    human_decision = state.get("human_decision", "")
    human_notes = state.get("human_notes", "")

    print(f"\n  {'=' * 50}")
    print(f"  INVESTIGATION REPORT — Order {order['id']}")
    print(f"  {'=' * 50}")
    print(f"  Customer: {order.get('customer_name', 'Unknown')} ({order['customer_email']})")
    print(f"  Amount:   ${order['amount']:.2f}")
    print(f"  Decision: {state['decision'].upper()}")
    print(f"  Score:    {state['risk_score']}/100")
    print(f"  Evidence: {len(evidence)} items collected")

    if guardrail:
        print(f"  Guardrail: {guardrail}")

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

    # Human review section
    if human_decision:
        print(f"  --- Human Review ---")
        print(f"  Human Decision: {human_decision}")
        if human_notes:
            print(f"  Notes: {human_notes}")

    # Infrastructure stats
    print(f"  --- Infrastructure ---")
    print(f"  Loops:  {state.get('loop_count', 0)}")
    print(f"  Tokens: {state.get('tokens_used', 0):,}")
    print(f"  {'=' * 50}")

    return {}


# --- Routing ---

def should_continue(state: FraudStateV5, config: RunnableConfig) -> Literal["execute_tools", "assess_risk"]:
    """Route after call_llm: continue investigating, hit dead-end, or wrap up."""
    if state.get("investigation_complete"):
        return "assess_risk"

    # Dead-end prevention
    configurable = config.get("configurable", {})
    max_loops = configurable.get("max_loops", DEFAULT_CONFIG["max_loops"])
    if state.get("loop_count", 0) >= max_loops:
        print(f"  [should_continue] Dead-end prevention: "
              f"{state['loop_count']}/{max_loops} loops — forcing exit")
        return "assess_risk"

    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tools"

    return "assess_risk"  # fallback: LLM stopped without calling calculate_risk_score


def route_after_assessment(state: FraudStateV5) -> Literal["human_review", "format_report"]:
    """Route after assess_risk: review decisions go to HITL, others straight to report."""
    if state.get("decision", "").startswith("review"):
        return "human_review"
    return "format_report"


# --- Build Graph ---

def build_graph(checkpointer=None, interrupt_before=None):
    """Build the Phase 5 graph with optional checkpointing and interrupts.

    Args:
        checkpointer: A LangGraph checkpointer (e.g. MemorySaver). Required for
                      interrupt() to work — stores state between pause and resume.
        interrupt_before: List of node names to always pause before (compile-time breakpoints).
    """
    graph = StateGraph(FraudStateV5)

    # Nodes
    graph.add_node("parse_order", parse_order)
    graph.add_node("call_llm", call_llm)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("assess_risk", assess_risk)
    graph.add_node("human_review", human_review)
    graph.add_node("format_report", format_report)

    # Edges
    graph.add_edge(START, "parse_order")
    graph.add_edge("parse_order", "call_llm")
    graph.add_conditional_edges("call_llm", should_continue)
    graph.add_edge("execute_tools", "call_llm")
    graph.add_conditional_edges("assess_risk", route_after_assessment)
    graph.add_edge("human_review", "format_report")
    graph.add_edge("format_report", END)

    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before
    return graph.compile(**compile_kwargs)


# --- Helpers ---

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


def make_initial_state(order: dict) -> dict:
    """Create a fresh initial state for a given order."""
    return {
        "order": order,
        "messages": [],
        "evidence": [],
        "risk_score": 0,
        "decision": "",
        "investigation_complete": False,
        "loop_count": 0,
        "tokens_used": 0,
        "guardrail_triggered": "",
        "human_decision": "",
        "human_notes": "",
    }


# --- Runner ---

if __name__ == "__main__":
    from shared.test_cases import ALL_CASES, CASE_4_CONFLICTING_SIGNALS, CASE_1_OBVIOUSLY_LEGIT

    print("=" * 70)
    print("Phase 5: Real Workflows — Checkpointing + Human-in-the-Loop")
    print("  Three capabilities, one mechanism: checkpointed state")
    print("=" * 70)

    # LangSmith tracing status
    tracing_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    langsmith_key = os.environ.get("LANGCHAIN_API_KEY", "")
    langsmith_project = os.environ.get("LANGCHAIN_PROJECT", "default")
    if tracing_enabled and langsmith_key:
        print(f"\n  LangSmith tracing: ON (project: {langsmith_project})")
    elif tracing_enabled and not langsmith_key:
        print(f"\n  LangSmith tracing: CONFIGURED but LANGCHAIN_API_KEY is empty")
    else:
        print(f"\n  LangSmith tracing: OFF")

    # ── Graph Visualization ─────────────────────────────────
    checkpointer = MemorySaver()
    app = build_graph(checkpointer=checkpointer)

    print(f"\n{'─' * 60}")
    print("  Graph Visualization (new human_review node)")
    print(f"{'─' * 60}")

    mermaid_text = app.get_graph().draw_mermaid()
    print("\n  Mermaid diagram:\n")
    for line in mermaid_text.strip().split("\n"):
        print(f"    {line}")

    # Try PNG export
    try:
        png_data = app.get_graph().draw_mermaid_png()
        png_path = os.path.join(os.path.dirname(__file__), "graph.png")
        with open(png_path, "wb") as f:
            f.write(png_data)
        print(f"\n  Graph image saved to: graph.png")
    except Exception as e:
        print(f"\n  (PNG export not available: {type(e).__name__}: {e})")

    # ══════════════════════════════════════════════════════════
    # Demo 1: Checkpointing Basics
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print("  Demo 1: Checkpointing Basics")
    print(f"  \"Save game\" — every node creates a checkpoint.")
    print(f"  MemorySaver stores them in memory (dev only).")
    print(f"  PostgresSaver for production.")
    print(f"{'═' * 60}")

    config_demo1 = make_config("demo1-checkpoint-basics")
    case1_order = CASE_1_OBVIOUSLY_LEGIT

    print(f"\n  Running Case 1 (Obviously Legit) with checkpointing...")
    result1 = app.invoke(make_initial_state(case1_order), config=config_demo1)

    print(f"\n  Result: {result1['decision'].upper()} (score: {result1['risk_score']})")

    # Inspect final state via get_state()
    final_state = app.get_state(config_demo1)
    print(f"\n  get_state() — final checkpoint:")
    print(f"    values.decision:    {final_state.values.get('decision', '')}")
    print(f"    values.risk_score:  {final_state.values.get('risk_score', 0)}")
    print(f"    values.evidence:    {len(final_state.values.get('evidence', []))} items")
    print(f"    values.loop_count:  {final_state.values.get('loop_count', 0)}")
    print(f"    values.tokens_used: {final_state.values.get('tokens_used', 0):,}")
    print(f"    next nodes:         {final_state.next}")

    # Show checkpoint history
    history = list(app.get_state_history(config_demo1))
    print(f"\n  get_state_history() — {len(history)} checkpoints:")
    for i, snapshot in enumerate(history):
        step = snapshot.metadata.get("step", "?")
        node = snapshot.metadata.get("source", "?")
        writes = snapshot.metadata.get("writes", {})
        write_nodes = list(writes.keys()) if writes else ["(start)"]
        print(f"    [{i}] step={step}, source={node}, wrote={write_nodes}, "
              f"next={snapshot.next}")

    # ══════════════════════════════════════════════════════════
    # Demo 2: HITL — Human Approves
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print("  Demo 2: Human-in-the-Loop — Human Approves")
    print(f"  Case 4 (conflicting signals, score 61) triggers interrupt.")
    print(f"  Human reviews and approves.")
    print(f"{'═' * 60}")

    config_demo2 = make_config("demo2-hitl-approve")
    case4_order = CASE_4_CONFLICTING_SIGNALS

    print(f"\n  Running Case 4... (will pause at human_review)")

    # Run until interrupt
    for event in app.stream(
        make_initial_state(case4_order), config=config_demo2, stream_mode="values"
    ):
        pass  # consume stream until pause

    # Check state — should be paused at human_review
    paused_state = app.get_state(config_demo2)
    print(f"\n  Graph paused. Next nodes: {paused_state.next}")

    if paused_state.tasks and paused_state.tasks[0].interrupts:
        review_package = paused_state.tasks[0].interrupts[0].value
        print(f"\n  Review package sent to human:")
        print(f"    Order:   {review_package['order_id']}")
        print(f"    Customer: {review_package['customer']}")
        print(f"    Amount:  ${review_package['amount']:.2f}")
        print(f"    Score:   {review_package['risk_score']}/100")
        print(f"    Decision: {review_package['decision']}")
        print(f"    Evidence:")
        for line in review_package['evidence_summary']:
            print(f"      {line}")
        print(f"    Options: {review_package['options']}")
    else:
        print(f"\n  (No interrupt found — graph may have completed without pausing)")

    # Resume with human approval
    print(f"\n  Human decides: APPROVED")
    print(f"  Notes: \"Verified: warehouse is the customer's actual office.\"")

    for event in app.stream(
        Command(resume={"decision": "approved", "notes": "Verified: warehouse is the customer's actual office."}),
        config=config_demo2,
        stream_mode="values",
    ):
        pass  # consume stream until completion

    # Show final result
    final2 = app.get_state(config_demo2)
    print(f"\n  Final result after human approval:")
    print(f"    Decision:       {final2.values.get('decision', '').upper()}")
    print(f"    Score:          {final2.values.get('risk_score', 0)}/100")
    print(f"    Human Decision: {final2.values.get('human_decision', '')}")
    print(f"    Human Notes:    {final2.values.get('human_notes', '')}")

    # ══════════════════════════════════════════════════════════
    # Demo 3: HITL — Human Rejects
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print("  Demo 3: Human-in-the-Loop — Human Rejects")
    print(f"  Same Case 4, but human overrides to reject.")
    print(f"{'═' * 60}")

    config_demo3 = make_config("demo3-hitl-reject")

    print(f"\n  Running Case 4... (will pause at human_review)")

    for event in app.stream(
        make_initial_state(case4_order), config=config_demo3, stream_mode="values"
    ):
        pass

    paused3 = app.get_state(config_demo3)
    print(f"  Graph paused. Next nodes: {paused3.next}")

    # Resume with rejection
    print(f"\n  Human decides: REJECTED")
    print(f"  Notes: \"Internal check confirms account takeover.\"")

    for event in app.stream(
        Command(resume={"decision": "rejected", "notes": "Internal check confirms account takeover."}),
        config=config_demo3,
        stream_mode="values",
    ):
        pass

    final3 = app.get_state(config_demo3)
    print(f"\n  Final result after human rejection:")
    print(f"    Decision:       {final3.values.get('decision', '').upper()}")
    print(f"    Score:          {final3.values.get('risk_score', 0)}/100")
    print(f"    Human Decision: {final3.values.get('human_decision', '')}")
    print(f"    Human Notes:    {final3.values.get('human_notes', '')}")
    print(f"\n  Decision changed from 'review' to 'reject' — human override applied.")

    # ══════════════════════════════════════════════════════════
    # Demo 4: State Forking / What-If Analysis
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print("  Demo 4: State Forking / What-If Analysis")
    print(f"  Fork from Demo 2's checkpoint, inject different score,")
    print(f"  see different path through the graph.")
    print(f"{'═' * 60}")

    # Find the checkpoint just before human_review from Demo 2's history
    demo2_history = list(app.get_state_history(make_config("demo2-hitl-approve")))

    # Walk history to find state right after assess_risk (before human_review)
    pre_review_checkpoint = None
    for snapshot in demo2_history:
        # The checkpoint where next=[human_review] is the one after assess_risk
        if "human_review" in (snapshot.next or []):
            pre_review_checkpoint = snapshot
            break

    if pre_review_checkpoint:
        print(f"\n  Found pre-review checkpoint:")
        print(f"    Score:    {pre_review_checkpoint.values.get('risk_score', 0)}/100")
        print(f"    Decision: {pre_review_checkpoint.values.get('decision', '')}")
        print(f"    Next:     {pre_review_checkpoint.next}")

        # What-if: filter out high-risk evidence, recalculate score offline
        original_evidence = pre_review_checkpoint.values.get("evidence", [])
        filtered_evidence = [e for e in original_evidence if e["risk_signal"] != "high_risk"]

        whatif_result = _calculate_risk_score_impl(filtered_evidence)
        print(f"\n  What-if: Remove high-risk evidence, recalculate:")
        print(f"    Original evidence: {len(original_evidence)} items")
        print(f"    Filtered evidence: {len(filtered_evidence)} items (high_risk removed)")
        print(f"    New score:    {whatif_result['risk_score']}/100")
        print(f"    New decision: {whatif_result['decision']}")

        # Fork: copy full state from pre-review checkpoint, override score+decision
        config_fork = make_config("demo4-fork-whatif")

        # Build full forked state — must include all fields since this is a new thread.
        # evidence uses operator.add reducer: adding to empty = the list itself.
        # messages uses add_messages reducer: adding to empty = the messages.
        fork_values = dict(pre_review_checkpoint.values)
        fork_values["risk_score"] = whatif_result["risk_score"]
        fork_values["decision"] = whatif_result["decision"]

        # as_node="assess_risk" tells LangGraph this checkpoint was produced by assess_risk,
        # so the conditional edge route_after_assessment runs next to determine the path.
        app.update_state(
            config_fork,
            values=fork_values,
            as_node="assess_risk",
        )

        forked_state = app.get_state(config_fork)
        print(f"\n  Forked state injected:")
        print(f"    Score:    {forked_state.values.get('risk_score', 0)}/100")
        print(f"    Decision: {forked_state.values.get('decision', '')}")
        print(f"    Next:     {forked_state.next}")

        if "format_report" in (forked_state.next or []):
            print(f"\n  Decision is '{whatif_result['decision']}' — skips human_review entirely!")

            # Resume from fork
            for event in app.stream(None, config=config_fork, stream_mode="values"):
                pass

            final_fork = app.get_state(config_fork)
            print(f"\n  Forked path completed:")
            print(f"    Decision:       {final_fork.values.get('decision', '').upper()}")
            print(f"    Human Decision: {final_fork.values.get('human_decision', '') or '(none — skipped)'}")
        else:
            # Forked decision is still "review" — still routes to human_review
            print(f"\n  Forked decision is still 'review' — would still route to human_review.")
            print(f"  (The what-if didn't change the outcome enough to skip HITL.)")

        print(f"\n  Comparison:")
        print(f"    Original: score={pre_review_checkpoint.values.get('risk_score', 0)}, "
              f"decision={pre_review_checkpoint.values.get('decision', '')}, path=HITL")
        fork_decision = whatif_result['decision']
        fork_path = "no HITL" if fork_decision == "approve" else "HITL"
        print(f"    Forked:   score={whatif_result['risk_score']}, "
              f"decision={fork_decision}, path={fork_path}")
    else:
        print(f"\n  Could not find pre-review checkpoint in Demo 2 history.")
        print(f"  (This can happen if Case 4 didn't trigger HITL.)")

    # ══════════════════════════════════════════════════════════
    # Demo 5: Compile-Time Breakpoints
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print("  Demo 5: Compile-Time Breakpoints")
    print(f"  interrupt_before=['format_report'] pauses EVERY run before formatting.")
    print(f"  Different from interrupt() which pauses CONDITIONALLY.")
    print(f"{'═' * 60}")

    checkpointer_demo5 = MemorySaver()
    app_breakpoint = build_graph(
        checkpointer=checkpointer_demo5,
        interrupt_before=["format_report"],
    )

    config_demo5 = make_config("demo5-breakpoint")

    print(f"\n  Running Case 1 (Obviously Legit) with breakpoint before format_report...")

    for event in app_breakpoint.stream(
        make_initial_state(case1_order), config=config_demo5, stream_mode="values"
    ):
        pass

    paused5 = app_breakpoint.get_state(config_demo5)
    print(f"\n  Graph paused. Next nodes: {paused5.next}")
    print(f"    Score:    {paused5.values.get('risk_score', 0)}/100")
    print(f"    Decision: {paused5.values.get('decision', '')}")
    print(f"\n  This is Case 1 (score 0, approve) — NOT a review case.")
    print(f"  But it paused anyway because interrupt_before is unconditional.")

    # Resume — for interrupt_before breakpoints, just pass None as input
    # (no Command needed since there's no interrupt() call to send data to)
    print(f"\n  Resuming (no input needed for compile-time breakpoints)...")
    for event in app_breakpoint.stream(
        None, config=config_demo5, stream_mode="values"
    ):
        pass

    final5 = app_breakpoint.get_state(config_demo5)
    print(f"\n  Completed. Decision: {final5.values.get('decision', '').upper()}")
    print(f"\n  Teaching: interrupt_before is for ALWAYS pausing (debugging, auditing).")
    print(f"  interrupt() inside a node is for CONDITIONAL pauses (review cases only).")

    # ══════════════════════════════════════════════════════════
    # Demo 6: Full Run — All 6 Cases
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print("  Demo 6: Full Run — All 6 Cases with HITL-Enabled Graph")
    print(f"{'═' * 60}")

    full_results = []
    for i, (name, order) in enumerate(ALL_CASES, 1):
        thread_id = f"demo6-case-{i}"
        config_case = make_config(thread_id)

        print(f"\n  >> {name}")

        # Run until completion or interrupt
        for event in app.stream(
            make_initial_state(order), config=config_case, stream_mode="values"
        ):
            pass

        state = app.get_state(config_case)
        hit_interrupt = bool(state.next)

        if hit_interrupt:
            # Auto-approve for the full run demo
            print(f"     INTERRUPTED at {state.next} — auto-approving for demo...")
            for event in app.stream(
                Command(resume={"decision": "approved", "notes": "Auto-approved in demo run."}),
                config=config_case,
                stream_mode="values",
            ):
                pass
            state = app.get_state(config_case)

        decision = state.values.get("decision", "?")
        score = state.values.get("risk_score", 0)
        human = state.values.get("human_decision", "")
        loops = state.values.get("loop_count", 0)
        tokens = state.values.get("tokens_used", 0)
        evidence_count = len(state.values.get("evidence", []))

        full_results.append({
            "name": name,
            "decision": decision,
            "score": score,
            "evidence": evidence_count,
            "loops": loops,
            "tokens": tokens,
            "hitl": hit_interrupt,
            "human_decision": human,
        })

        hitl_str = f" + human={human}" if human else ""
        print(f"     -> {decision.upper()} (score: {score}) | "
              f"{loops} loops, {tokens:,} tokens{hitl_str}")

    # ── Summary Table ────────────────────────────────────────
    print(f"\n{'═' * 80}")
    print("  Phase 5 Results Summary")
    print(f"{'═' * 80}")
    print(f"  {'Case':<30} {'Decision':<12} {'Score':>5} {'HITL?':>6} {'Human':>10} {'Tokens':>8}")
    print(f"  {'─' * 79}")
    for r in full_results:
        hitl_str = "Yes" if r["hitl"] else "No"
        human_str = r["human_decision"] or "-"
        print(f"  {r['name']:<30} {r['decision'].upper():<12} {r['score']:>5} "
              f"{hitl_str:>6} {human_str:>10} {r['tokens']:>8,}")

    # Phase 4 baseline comparison
    print(f"\n  Phase 4 baseline (for comparison):")
    print(f"  {'Case':<30} {'Decision':<12} {'Score':>5} {'HITL?':>6}")
    print(f"  {'─' * 55}")
    p4 = [
        ("1: Obviously Legit", "APPROVE", 0),
        ("2: Mildly Suspicious", "APPROVE", 33),
        ("3: High Risk", "REJECT", 100),
        ("4: Conflicting Signals", "REVIEW", 61),
        ("5: Historical Fraud", "APPROVE", 10),
        ("6: Tool Error", "APPROVE", 6),
    ]
    for name, decision, score in p4:
        print(f"  {name:<30} {decision:<12} {score:>5} {'No':>6}")

    # ── Context Injection Narrative ──────────────────────────
    print(f"\n{'═' * 60}")
    print("  Context Injection — The Pattern (no runnable demo)")
    print(f"{'═' * 60}")
    print("""
  The honest memory framing:

  Before invoke, load history. After invoke, save findings.
  LangGraph standardizes the pattern. The persistence is your database.

  Pattern:
    # Before investigation
    customer_context = load_from_db(order["customer_email"])
    state["messages"].insert(0, SystemMessage(content=customer_context))

    # Run investigation
    result = app.invoke(state, config)

    # After investigation
    save_to_db(order["customer_email"], result["evidence"])

  This is NOT a LangGraph feature — it's the same pattern you'd use with
  any framework. LangGraph's contribution is checkpointing (pause/resume)
  and state management (reducers, typed state). The persistence layer
  (PostgresSaver, Redis, your own DB) is pluggable.
""")

    # ── Broader Context: Workflow Orchestration ──────────────
    print(f"{'═' * 70}")
    print("  Broader Context: Where LangGraph Fits in Workflow Orchestration")
    print(f"{'═' * 70}")
    print("""
  If you've used workflow orchestrators before (Dapr, Temporal, Airflow),
  checkpointing and HITL will feel familiar — LangGraph applies those same
  durable workflow patterns to LLM agent loops.

  The question is which tool owns the workflow. If the business process
  drives things (order fulfillment, inventory, payments) and AI is one
  step, a traditional orchestrator should own it — call LangGraph or a
  raw LLM when you need reasoning. If the entire workflow IS the AI
  reasoning (research, investigation, content generation), LangGraph
  owns it. For simple one-shot scoring ("rate this order 0-100"), you
  may not need an agent framework at all — just call the LLM directly.

  For concrete examples of a traditional workflow calling an LLM as one
  step in a larger business process, see:
  - github.com/reisenberg/dapr_demo Phase 5: .NET Dapr Workflow (saga
    pattern with compensation)
  - github.com/reisenberg/dapr_demo Phase 7-A: AI fraud scoring as a
    workflow activity (CheckFraudActivity calls Claude via the Dapr
    Conversation API — no agent loop, just a single LLM call inside
    a durable saga)
""")

    print(f"{'═' * 70}")
    print("Done. All 6 demos complete.")
    print(f"{'═' * 70}")
