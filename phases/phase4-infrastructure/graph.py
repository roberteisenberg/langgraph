"""
Phase 4: The Infrastructure — What LangGraph Gives You Beyond a While Loop

Six infrastructure features that justify the framework:

Already built into LangGraph (just use them):
1. Graph visualization  — app.get_graph().draw_mermaid()
2. Streaming            — app.stream(state, stream_mode="updates")
3. Recursion limits     — config={"recursion_limit": 25}

Need state/node changes (build into the graph):
4. Error handling       — Phase 3 already handles errors; Phase 4 adds structured summary
5. Dead-end prevention  — loop_count in state, force-exit after N iterations
6. Token budget breaker — tokens_used in state, kill if budget exceeded

Graph: START -> parse_order -> call_llm -> [should_continue]
                                            |-- has tool calls -> execute_tools -> call_llm (loop)
                                            +-- done -> assess_risk -> format_report -> END

Same shape as Phase 3 — infrastructure lives inside existing nodes and the runner.
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

# Add project root to path for shared imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# Add phase3 directory for tools import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase3-investigator"))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# LangSmith tracing: only enable if API key is actually set.
# Without this guard, an empty LANGCHAIN_API_KEY + LANGCHAIN_TRACING_V2=true
# causes hundreds of 401 errors to stderr.
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


# --- State ---

class Evidence(TypedDict):
    tool: str              # "check_customer_history", etc.
    finding: str           # "Address matches known freight forwarder warehouse"
    risk_signal: str       # "low_risk" | "medium_risk" | "high_risk" | "neutral" | "error"
    confidence: float      # 0.0-1.0
    raw_data: dict         # full tool output for debugging
    timestamp: str         # ISO 8601


class FraudStateV4(TypedDict):
    # --- carried from Phase 3 ---
    order: dict
    messages: Annotated[list, add_messages]
    evidence: Annotated[list[Evidence], operator.add]
    risk_score: int
    decision: str
    investigation_complete: bool
    # --- new in Phase 4 ---
    loop_count: int                # incremented each call_llm pass
    tokens_used: int               # cumulative token count
    guardrail_triggered: str       # "" | "recursion_limit" | "dead_end" | "token_budget"


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


# --- Evidence Extraction (same as Phase 3) ---

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

def parse_order(state: FraudStateV4) -> dict:
    """Validate required fields in the order."""
    order = state["order"]

    required = ["id", "customer_email", "amount", "items", "shipping_address"]
    missing = [f for f in required if f not in order]
    if missing:
        raise ValueError(f"Order missing required fields: {missing}")

    print(f"  [parse_order] Order {order['id']}: ${order['amount']:.2f}, "
          f"{len(order['items'])} items, to {order['shipping_address'][:40]}...")

    return {}


def call_llm(state: FraudStateV4, config: RunnableConfig) -> dict:
    """Call the LLM with loop counting and token budget enforcement.

    Phase 4 additions over Phase 3:
    - Increments loop_count on every call
    - Checks token budget before calling the API — skips if exceeded
    - Extracts token usage from response.usage_metadata after each call
    """
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


def execute_tools(state: FraudStateV4) -> dict:
    """Custom tool execution node — same logic as Phase 3.

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


def assess_risk(state: FraudStateV4, config: RunnableConfig) -> dict:
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


def format_report(state: FraudStateV4) -> dict:
    """Print a structured investigation report with infrastructure stats."""
    order = state["order"]
    evidence = state.get("evidence", [])
    guardrail = state.get("guardrail_triggered", "")

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

    # Infrastructure stats
    print(f"  --- Infrastructure ---")
    print(f"  Loops:  {state.get('loop_count', 0)}")
    print(f"  Tokens: {state.get('tokens_used', 0):,}")
    print(f"  {'=' * 50}")

    return {}


# --- Routing ---

def should_continue(state: FraudStateV4, config: RunnableConfig) -> Literal["execute_tools", "assess_risk"]:
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


# --- Build Graph ---

def build_graph():
    """Build the Phase 4 graph — same shape as Phase 3, with infrastructure in nodes."""
    graph = StateGraph(FraudStateV4)

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
    }


# --- Runner ---

if __name__ == "__main__":
    from langgraph.errors import GraphRecursionError
    from shared.test_cases import ALL_CASES

    print("=" * 60)
    print("Phase 4: The Infrastructure")
    print("  What LangGraph gives you beyond a while loop")
    print("=" * 60)

    # LangSmith tracing status
    tracing_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    langsmith_key = os.environ.get("LANGCHAIN_API_KEY", "")
    langsmith_project = os.environ.get("LANGCHAIN_PROJECT", "default")
    if tracing_enabled and langsmith_key:
        print(f"\n  LangSmith tracing: ON (project: {langsmith_project})")
        print(f"  View traces at: https://smith.langchain.com")
    elif tracing_enabled and not langsmith_key:
        print(f"\n  LangSmith tracing: CONFIGURED but LANGCHAIN_API_KEY is empty")
        print(f"  Set your key in .env to enable — get one at https://smith.langchain.com")
    else:
        print(f"\n  LangSmith tracing: OFF (set LANGCHAIN_TRACING_V2=true in .env to enable)")

    app = build_graph()

    # ── Demo 1: Graph Visualization ──────────────────────────
    print(f"\n{'─' * 55}")
    print("  Demo 1: Graph Visualization")
    print(f"{'─' * 55}")

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

    # ── Demo 2: Streaming — All 6 Cases ─────────────────────
    # app.stream() IS the execution — it runs the graph and yields
    # events as each node completes. This is not a separate observation
    # layer; it's how you run the graph with real-time visibility.
    #
    # For visual graph traces (nodes lighting up, state inspection),
    # see LangGraph Studio (desktop app) or LangSmith (cloud tracing).
    # Those tools consume the same stream events and render them graphically.
    print(f"\n{'─' * 55}")
    print("  Demo 2: Streaming — All 6 Cases")
    print(f"  (app.stream() IS the execution, with real-time node events)")
    print(f"{'─' * 55}")

    case1_order = ALL_CASES[0][1]
    config_normal = {
        "recursion_limit": DEFAULT_CONFIG["recursion_limit"],
        "configurable": {
            "max_loops": DEFAULT_CONFIG["max_loops"],
            "token_budget": DEFAULT_CONFIG["token_budget"],
        },
    }

    streamed_results = []
    for name, order in ALL_CASES:
        print(f"\n  >> {name}")

        # Stream the execution — each event is a node completing
        trace = []
        final_state = {}
        for event in app.stream(
            make_initial_state(order), config=config_normal, stream_mode="updates"
        ):
            node_name = list(event.keys())[0]
            trace.append(node_name)
            updates = event[node_name]
            if updates:
                keys = list(updates.keys())
                print(f"     [{node_name}] {keys}")
                final_state.update(updates)
            else:
                print(f"     [{node_name}] (no state changes)")

        # Compact trace: collapse repeated call_llm→execute_tools into loop counts
        compact = []
        loop_runs = 0
        i = 0
        while i < len(trace):
            if trace[i] == "call_llm" and i + 1 < len(trace) and trace[i + 1] == "execute_tools":
                loop_runs += 1
                i += 2
            else:
                if loop_runs > 0:
                    compact.append(f"[call_llm <-> execute_tools] x{loop_runs}")
                    loop_runs = 0
                compact.append(trace[i])
                i += 1
        if loop_runs > 0:
            compact.append(f"[call_llm <-> execute_tools] x{loop_runs}")

        tools_used = extract_tool_calls(final_state.get("messages", []))
        tokens = final_state.get("tokens_used", 0)
        loops = final_state.get("loop_count", 0)
        decision = final_state.get("decision", "?")
        score = final_state.get("risk_score", 0)
        evidence_count = len(final_state.get("evidence", []))

        streamed_results.append({
            "name": name,
            "decision": decision,
            "score": score,
            "evidence": evidence_count,
            "tools": len(tools_used),
            "loops": loops,
            "tokens": tokens,
            "trace": compact,
            "steps": len(trace),
        })

        print(f"     Trace: {' -> '.join(compact)}")
        print(f"     -> {decision.upper()} (score: {score}) | "
              f"{len(trace)} steps, {loops} loops, {tokens:,} tokens")

    # ── Demo 3: Recursion Limit ──────────────────────────────
    print(f"\n{'─' * 55}")
    print("  Demo 3: Recursion Limit (provoke GraphRecursionError)")
    print(f"{'─' * 55}")

    case3_order = ALL_CASES[2][1]  # High Risk — will loop
    config_tight_recursion = {
        "recursion_limit": 3,  # Very tight — will hit limit fast
        "configurable": {
            "max_loops": DEFAULT_CONFIG["max_loops"],
            "token_budget": DEFAULT_CONFIG["token_budget"],
        },
    }

    try:
        app.invoke(make_initial_state(case3_order), config=config_tight_recursion)
        print("\n  (Recursion limit was NOT hit — increase test case complexity)")
    except GraphRecursionError as e:
        print(f"\n  Caught GraphRecursionError (as expected):")
        print(f"    {e}")

    # ── Demo 4: Error Handling Summary ───────────────────────
    print(f"\n{'─' * 55}")
    print("  Demo 4: Error Handling (Case 6 — Tool Error)")
    print(f"{'─' * 55}")

    case6_order = ALL_CASES[5][1]  # Tool Error case
    result4 = app.invoke(make_initial_state(case6_order), config=config_normal)

    print(f"\n  Evidence trail showing error handling:")
    for i, e in enumerate(result4.get("evidence", []), 1):
        marker = " *** ERROR SIGNAL ***" if e["risk_signal"] == "error" else ""
        print(f"    {i}. {e['tool']}: {e['risk_signal']} (conf={e['confidence']:.2f}){marker}")
        if e["risk_signal"] == "error":
            print(f"       Finding: {e['finding']}")
            print(f"       The system continued investigating despite this error.")
    print(f"\n  Final decision: {result4['decision'].upper()} (score: {result4['risk_score']})")
    print(f"  The error was recorded as evidence but did not block the investigation.")

    # ── Demo 5: Dead-End Prevention ──────────────────────────
    print(f"\n{'─' * 55}")
    print("  Demo 5: Dead-End Prevention (Case 3 with max_loops=2)")
    print(f"{'─' * 55}")

    config_tight_loops = {
        "recursion_limit": DEFAULT_CONFIG["recursion_limit"],
        "configurable": {
            "max_loops": 2,
            "token_budget": DEFAULT_CONFIG["token_budget"],
        },
    }

    result5 = app.invoke(make_initial_state(case3_order), config=config_tight_loops)

    print(f"\n  Forced exit after {result5.get('loop_count', '?')} loops")
    print(f"  Evidence collected: {len(result5.get('evidence', []))} items")
    print(f"  Guardrail:  {result5.get('guardrail_triggered', 'none')}")
    print(f"  Decision:   {result5['decision'].upper()}")
    print(f"  Score:      {result5['risk_score']}/100")
    print(f"  The system scored with partial evidence rather than looping forever.")

    # ── Demo 6: Token Budget Circuit Breaker ─────────────────
    print(f"\n{'─' * 55}")
    print("  Demo 6: Token Budget Circuit Breaker (Case 1 with budget=1000)")
    print(f"{'─' * 55}")

    config_tight_budget = {
        "recursion_limit": DEFAULT_CONFIG["recursion_limit"],
        "configurable": {
            "max_loops": DEFAULT_CONFIG["max_loops"],
            "token_budget": 1000,  # Very low — will trigger after 1-2 calls
        },
    }

    result6 = app.invoke(make_initial_state(case1_order), config=config_tight_budget)

    print(f"\n  Budget exhausted after {result6.get('loop_count', '?')} loops, "
          f"{result6.get('tokens_used', '?'):,} tokens")
    print(f"  Guardrail:  {result6.get('guardrail_triggered', 'none')}")
    print(f"  Decision:   {result6['decision'].upper()}")
    print(f"  Score:      {result6['risk_score']}/100")
    print(f"  The system terminated early and scored with available evidence.")

    # ── Execution Trace Comparison (from Demo 2 streaming) ──
    print(f"\n{'=' * 75}")
    print("  Execution Traces — How Different Cases Navigate the Same Graph")
    print(f"{'=' * 75}")
    for r in streamed_results:
        print(f"\n  {r['name']}")
        print(f"    {' -> '.join(r['trace'])}")
        print(f"    {r['decision'].upper()} ({r['score']}) | "
              f"{r['steps']} steps, {r['loops']} loops, {r['tokens']:,} tokens")

    # ── Summary Table ────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("  Phase 4 Results Summary")
    print(f"{'=' * 75}")
    print(f"  {'Case':<30} {'Decision':<12} {'Score':>5} {'Steps':>5} {'Loops':>5} {'Tokens':>8}")
    print(f"  {'─' * 74}")
    for r in streamed_results:
        print(f"  {r['name']:<30} {r['decision'].upper():<12} {r['score']:>5} "
              f"{r['steps']:>5} {r['loops']:>5} {r['tokens']:>8,}")

    print(f"\n  Phase 3 baseline (for comparison):")
    print(f"  {'Case':<30} {'Decision':<12} {'Score':>5}")
    print(f"  {'─' * 50}")
    p3 = [
        ("1: Obviously Legit", "APPROVE", 0),
        ("2: Mildly Suspicious", "APPROVE", 33),
        ("3: High Risk", "REJECT", 100),
        ("4: Conflicting Signals", "REVIEW", 61),
        ("5: Historical Fraud", "APPROVE", 10),
        ("6: Tool Error", "APPROVE", 6),
    ]
    for name, decision, score in p3:
        print(f"  {name:<30} {decision:<12} {score:>5}")

    print(f"\n  Note: For visual graph traces (nodes lighting up, state inspection),")
    print(f"  see LangGraph Studio (desktop) or LangSmith (cloud). They consume")
    print(f"  the same stream events and render them as interactive diagrams.")

    print(f"\n{'=' * 75}")
    print("Done. All demos and cases complete.")
    print(f"{'=' * 75}")
