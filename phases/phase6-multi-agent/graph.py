"""
Phase 6: Multi-Agent — When One Agent Isn't Enough

Supervisor (Sonnet) + Specialists (Haiku) architecture. The supervisor decides
which specialist to consult next. Specialists make one LLM call each, execute
their domain tools, and return evidence. The supervisor reviews findings and
routes to the next specialist or finalizes.

Graph: START -> parse_order -> supervisor -> [Command routing]
                                   ^         |-- customer_analyst -> supervisor
                                   |         |-- address_analyst -> supervisor
                                   |         |-- payment_analyst -> supervisor
                                   |         +-- assess_risk -> [route_after_assessment]
                                   |                              |-- review -> human_review -> format_report -> END
                                   +------------------------------+-- approve/reject -> format_report -> END

Honest framing: With 5 tools, a single agent handles everything. Multi-agent
adds coordination overhead. The real payoff is at 200+ tools (specialists reduce
tool selection noise) and cost optimization (Haiku specialists at ~$0.25/MTok
vs Sonnet at ~$3/MTok).
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
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
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
    check_customer_history,
    check_payment_pattern,
    verify_shipping_address,
    search_fraud_database,
    _calculate_risk_score_impl,
)


# --- Configuration ---

DEFAULT_CONFIG = {
    "max_loops": 10,
    "token_budget": 100_000,
    "recursion_limit": 50,  # Higher than Phase 5 — supervisor loops add steps
}

SUPERVISOR_MODEL = "claude-sonnet-4-20250514"
SPECIALIST_MODEL = "claude-haiku-4-5-20251001"


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
    tool: str
    finding: str
    risk_signal: str
    confidence: float
    raw_data: dict
    timestamp: str


class FraudStateV6(TypedDict):
    # --- carried from Phase 5 ---
    order: dict
    messages: Annotated[list, add_messages]
    evidence: Annotated[list[Evidence], operator.add]
    risk_score: int
    decision: str
    investigation_complete: bool
    loop_count: int
    tokens_used: int
    guardrail_triggered: str
    human_decision: str
    human_notes: str
    # --- new in Phase 6 ---
    specialist_log: Annotated[list[str], operator.add]


# --- Evidence Extraction (same as Phase 4/5 — defined here since tools.py doesn't export it) ---

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


# --- Routing Tools (for supervisor only) ---

@tool
def consult_customer_analyst(reason: str) -> str:
    """Route to the customer analyst to check customer history and fraud records."""
    return "Routing to customer analyst"


@tool
def consult_address_analyst(reason: str) -> str:
    """Route to the address analyst to verify shipping address and geo-risk."""
    return "Routing to address analyst"


@tool
def consult_payment_analyst(reason: str) -> str:
    """Route to the payment analyst to check payment patterns and anomalies."""
    return "Routing to payment analyst"


@tool
def finalize_investigation(summary: str) -> str:
    """All relevant specialists have been consulted. Finalize the investigation."""
    return "Finalizing"


ROUTING_TOOLS = [
    consult_customer_analyst,
    consult_address_analyst,
    consult_payment_analyst,
    finalize_investigation,
]

TOOL_TO_TARGET = {
    "consult_customer_analyst": "customer_analyst",
    "consult_address_analyst": "address_analyst",
    "consult_payment_analyst": "payment_analyst",
    "finalize_investigation": "assess_risk",
}


# --- Supervisor ---

SUPERVISOR_PROMPT = """You are a fraud investigation supervisor coordinating specialist analysts.

Review the order details and evidence collected so far. Decide which specialist
to consult next. Each specialist checks different aspects:

- customer_analyst: Customer history, account age, prior fraud flags
- address_analyst: Shipping address type, geo-risk, shared addresses
- payment_analyst: Payment patterns, spending anomalies, velocity

Strategy:
1. Start with the most relevant check for this order
2. After each specialist reports, review findings and decide if more checks needed
3. Conflicting signals require consulting additional specialists
4. When you have enough evidence, call finalize_investigation

Do NOT consult the same specialist twice. Do NOT skip specialists when signals conflict.

When calling a routing tool, provide a brief reason explaining why you are routing to that specialist."""

supervisor_llm = ChatAnthropic(
    model=SUPERVISOR_MODEL, temperature=0
).bind_tools(ROUTING_TOOLS)


def _extract_tokens(response) -> int:
    """Extract total tokens from an LLM response."""
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        meta = response.usage_metadata
        return meta.get("input_tokens", 0) + meta.get("output_tokens", 0)
    return 0


def supervisor(state: FraudStateV6, config: RunnableConfig) -> Command[Literal[
    "customer_analyst", "address_analyst", "payment_analyst", "assess_risk"
]]:
    """Supervisor node — routes to specialists via Command().

    Reads order + evidence + specialist_log, decides which specialist to
    consult next (or finalize). Returns Command(goto=target) for routing.
    """
    configurable = config.get("configurable", {})
    max_loops = configurable.get("max_loops", DEFAULT_CONFIG["max_loops"])

    loop_count = state.get("loop_count", 0) + 1

    # Guardrail: max supervisor loops
    if loop_count > max_loops:
        print(f"  [supervisor] Loop limit hit ({loop_count}/{max_loops}) — forcing assessment")
        return Command(goto="assess_risk", update={
            "loop_count": loop_count,
            "guardrail_triggered": "dead_end",
        })

    # Build context for supervisor
    order = state["order"]
    evidence = state.get("evidence", [])
    specialist_log = state.get("specialist_log", [])

    context_parts = [
        f"Order ID: {order['id']}",
        f"Customer: {order.get('customer_name', 'Unknown')} ({order['customer_email']})",
        f"Amount: ${order['amount']:.2f}",
        f"Items: {', '.join(item['name'] for item in order['items'])}",
        f"Shipping Address: {order['shipping_address']}",
        f"Metadata: {json.dumps(order.get('metadata', {}), indent=2)}",
    ]

    if specialist_log:
        context_parts.append(f"\nSpecialists already consulted: {', '.join(specialist_log)}")

    if evidence:
        context_parts.append("\nEvidence collected so far:")
        for i, e in enumerate(evidence, 1):
            context_parts.append(
                f"  {i}. [{e['risk_signal']}] {e['tool']}: {e['finding']} "
                f"(confidence: {e['confidence']:.2f})"
            )

    context = "\n".join(context_parts)

    # Call supervisor LLM
    messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=f"Investigate this order:\n\n{context}"),
    ]

    for attempt in range(3):
        try:
            response = supervisor_llm.invoke(messages)
            break
        except Exception as e:
            if attempt < 2 and "500" in str(e):
                print(f"  [supervisor] API error, retrying ({attempt + 1}/3)...")
                time.sleep(2)
            else:
                raise

    new_tokens = _extract_tokens(response)

    # Parse routing tool call
    if not response.tool_calls:
        # No tool call — supervisor didn't route, force finalize
        print(f"  [supervisor] No routing tool called — forcing assessment")
        return Command(goto="assess_risk", update={
            "messages": [response],
            "loop_count": loop_count,
            "tokens_used": state.get("tokens_used", 0) + new_tokens,
        })

    tool_call = response.tool_calls[0]
    tool_name = tool_call["name"]
    tool_reason = tool_call["args"].get("reason", "") or tool_call["args"].get("summary", "")
    target = TOOL_TO_TARGET.get(tool_name)

    if not target:
        print(f"  [supervisor] Unknown routing tool: {tool_name} — forcing assessment")
        target = "assess_risk"

    print(f"  [supervisor] -> {target} (reason: {tool_reason[:80]})")

    return Command(
        goto=target,
        update={
            "messages": [response],
            "loop_count": loop_count,
            "tokens_used": state.get("tokens_used", 0) + new_tokens,
        },
    )


# --- Specialist Factory ---

def make_specialist(name: str, system_prompt: str, tools: list, model: str):
    """Create a specialist node — one LLM call, execute tools, return evidence."""
    specialist_llm = ChatAnthropic(model=model, temperature=0).bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    def specialist_node(state: FraudStateV6) -> dict:
        order = state["order"]

        # Build focused prompt with order details
        order_text = (
            f"Order ID: {order['id']}\n"
            f"Customer: {order.get('customer_name', 'Unknown')} ({order['customer_email']})\n"
            f"Amount: ${order['amount']:.2f}\n"
            f"Items: {', '.join(item['name'] for item in order['items'])}\n"
            f"Shipping Address: {order['shipping_address']}\n"
            f"Metadata: {json.dumps(order.get('metadata', {}), indent=2)}\n\n"
            f"IMPORTANT: Use these EXACT values when calling tools:\n"
            f"  Email: {order['customer_email']}\n"
            f"  Address: {order['shipping_address']}\n"
            f"  Amount: {order['amount']}"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Investigate this order:\n\n{order_text}"),
        ]

        # ONE LLM call — Haiku calls domain tools
        for attempt in range(3):
            try:
                response = specialist_llm.invoke(messages)
                break
            except Exception as e:
                if attempt < 2 and "500" in str(e):
                    print(f"  [{name}] API error, retrying ({attempt + 1}/3)...")
                    time.sleep(2)
                else:
                    raise

        new_tokens = _extract_tokens(response)

        # Execute tool calls, create evidence
        new_evidence = []
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]

            # Validate tool args — prevent LLM hallucination (same as Phase 4/5)
            if tool_name == "verify_shipping_address":
                tool_args = {"address": order["shipping_address"]}
            elif tool_name == "check_payment_pattern":
                tool_args = {"email": order["customer_email"], "amount": order["amount"]}
            elif tool_name == "check_customer_history":
                tool_args = {"email": order["customer_email"]}

            tool_fn = tool_map.get(tool_name)
            if not tool_fn:
                new_evidence.append(Evidence(
                    tool=tool_name,
                    finding=f"Unknown tool: {tool_name}",
                    risk_signal="error",
                    confidence=0.5,
                    raw_data={"error": f"Unknown tool: {tool_name}"},
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

                evidence = _create_evidence(tool_name, result_data)
                new_evidence.append(evidence)

                print(f"  [{name}] {tool_name}: {evidence['risk_signal']} "
                      f"(confidence: {evidence['confidence']:.2f})")

            except Exception as e:
                error_data = {"error": str(e)}
                evidence = _create_evidence(tool_name, error_data)
                new_evidence.append(evidence)
                print(f"  [{name}] {tool_name}: ERROR — {e}")

        if not response.tool_calls:
            print(f"  [{name}] No tools called (model returned text only)")

        return {
            "evidence": new_evidence,
            "tokens_used": state.get("tokens_used", 0) + new_tokens,
            "specialist_log": [name],
        }

    specialist_node.__name__ = name
    return specialist_node


# --- Create Specialists ---

customer_analyst_node = make_specialist(
    name="customer_analyst",
    system_prompt=(
        "You investigate customer backgrounds for fraud investigations. "
        "Check customer history and cross-reference the customer's EMAIL with "
        "fraud databases. Report your findings factually — do not make final "
        "risk decisions.\n\n"
        "Use the exact email address provided for all lookups. "
        "When calling search_fraud_database, use the customer's EMAIL as the indicator. "
        "Do NOT search for the shipping address — that is the address analyst's job."
    ),
    tools=[check_customer_history, search_fraud_database],
    model=SPECIALIST_MODEL,
)

address_analyst_node = make_specialist(
    name="address_analyst",
    system_prompt=(
        "You investigate shipping addresses for fraud investigations. "
        "Verify address type and geo-risk. Report your findings factually — "
        "do not make final risk decisions.\n\n"
        "Use the exact shipping address provided for verification."
    ),
    tools=[verify_shipping_address],
    model=SPECIALIST_MODEL,
)

payment_analyst_node = make_specialist(
    name="payment_analyst",
    system_prompt=(
        "You investigate payment patterns for fraud investigations. "
        "Check if the order amount is typical for this customer and flag "
        "anomalies. Report your findings factually — do not make final "
        "risk decisions.\n\n"
        "Use the exact email and amount provided for lookups. "
        "Call all available tools to build a complete picture."
    ),
    tools=[check_payment_pattern],
    model=SPECIALIST_MODEL,
)


# --- Shared Nodes (adapted from Phase 5) ---

def parse_order(state: FraudStateV6) -> dict:
    """Validate required fields in the order."""
    order = state["order"]

    required = ["id", "customer_email", "amount", "items", "shipping_address"]
    missing = [f for f in required if f not in order]
    if missing:
        raise ValueError(f"Order missing required fields: {missing}")

    print(f"  [parse_order] Order {order['id']}: ${order['amount']:.2f}, "
          f"{len(order['items'])} items, to {order['shipping_address'][:40]}...")

    return {}


def assess_risk(state: FraudStateV6, config: RunnableConfig) -> dict:
    """Deterministic scoring on accumulated evidence.

    In multi-agent mode, no specialist calls calculate_risk_score — the
    supervisor routes here directly after finalize_investigation.
    Always runs _calculate_risk_score_impl on whatever evidence exists.
    """
    guardrail = state.get("guardrail_triggered", "")

    evidence = state.get("evidence", [])
    label = f"Guardrail: {guardrail} — scoring" if guardrail else "Scoring"
    print(f"  [assess_risk] {label} on {len(evidence)} evidence items")
    result = _calculate_risk_score_impl(evidence)

    decision = result["decision"]
    if guardrail:
        decision = f"{decision} ({guardrail})"

    return {
        "risk_score": result["risk_score"],
        "decision": decision,
        "investigation_complete": True,
    }


def human_review(state: FraudStateV6) -> dict:
    """Pause for human review on ambiguous cases. Same as Phase 5."""
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
        "specialists_consulted": state.get("specialist_log", []),
        "action_required": "Review this investigation and provide your decision.",
        "options": ["approved", "rejected"],
    }

    human_input = interrupt(review_package)

    human_decision = human_input.get("decision", "approved")
    human_notes = human_input.get("notes", "")

    result = {"human_decision": human_decision, "human_notes": human_notes}

    if human_decision == "rejected":
        result["decision"] = "reject"

    return result


def format_report(state: FraudStateV6) -> dict:
    """Print a structured investigation report with multi-agent details."""
    order = state["order"]
    evidence = state.get("evidence", [])
    guardrail = state.get("guardrail_triggered", "")
    human_decision = state.get("human_decision", "")
    human_notes = state.get("human_notes", "")
    specialist_log = state.get("specialist_log", [])

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

    # Multi-agent section
    print(f"  --- Multi-Agent ---")
    print(f"  Specialists: {', '.join(specialist_log) if specialist_log else '(none)'}")
    print(f"  Supervisor loops: {state.get('loop_count', 0)}")
    print(f"  Tokens: {state.get('tokens_used', 0):,}")
    print(f"  {'=' * 50}")

    return {}


# --- Routing ---

def route_after_assessment(state: FraudStateV6) -> Literal["human_review", "format_report"]:
    """Route after assess_risk: review decisions go to HITL."""
    if state.get("decision", "").startswith("review"):
        return "human_review"
    return "format_report"


# --- Build Graph ---

def build_graph(checkpointer=None):
    """Build the Phase 6 multi-agent graph."""
    graph = StateGraph(FraudStateV6)

    # Nodes
    graph.add_node("parse_order", parse_order)
    graph.add_node("supervisor", supervisor)
    graph.add_node("customer_analyst", customer_analyst_node)
    graph.add_node("address_analyst", address_analyst_node)
    graph.add_node("payment_analyst", payment_analyst_node)
    graph.add_node("assess_risk", assess_risk)
    graph.add_node("human_review", human_review)
    graph.add_node("format_report", format_report)

    # Edges
    graph.add_edge(START, "parse_order")
    graph.add_edge("parse_order", "supervisor")
    # supervisor routes via Command() — no conditional edges needed
    graph.add_edge("customer_analyst", "supervisor")
    graph.add_edge("address_analyst", "supervisor")
    graph.add_edge("payment_analyst", "supervisor")
    graph.add_conditional_edges("assess_risk", route_after_assessment)
    graph.add_edge("human_review", "format_report")
    graph.add_edge("format_report", END)

    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    return graph.compile(**compile_kwargs)


# --- Helpers ---

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
        "specialist_log": [],
    }


# --- Runner ---

if __name__ == "__main__":
    from shared.test_cases import (
        ALL_CASES, CASE_4_CONFLICTING_SIGNALS, CASE_1_OBVIOUSLY_LEGIT,
    )

    print("=" * 70)
    print("Phase 6: Multi-Agent — When One Agent Isn't Enough")
    print("  Supervisor (Sonnet) + Specialists (Haiku)")
    print("  Honest framing: orchestration overhead vs cost optimization")
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

    # ══════════════════════════════════════════════════════════
    # Demo 1: Graph Visualization
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  Demo 1: Graph Visualization")
    print(f"  Supervisor hub with specialist spokes — star topology")
    print(f"{'=' * 60}")

    checkpointer = MemorySaver()
    app = build_graph(checkpointer=checkpointer)

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

    print(f"\n  Compare to Phase 5's linear graph — this is a star topology.")
    print(f"  The supervisor is the hub. Specialists are spokes. Evidence flows inward.")

    # ══════════════════════════════════════════════════════════
    # Demo 2: Single Case Walkthrough (Case 4 — Conflicting Signals)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  Demo 2: Case 4 Walkthrough (Conflicting Signals)")
    print(f"  Supervisor sees conflicting signals, consults all three specialists.")
    print(f"  Shows routing decisions, specialist findings, and HITL interrupt.")
    print(f"{'=' * 60}")

    config_demo2 = make_config("demo2-case4-walkthrough")
    case4_order = CASE_4_CONFLICTING_SIGNALS

    print(f"\n  Running Case 4... (will pause at human_review if score is 50-79)")
    print(f"  Watch supervisor routing decisions:\n")

    # Stream to show routing decisions
    for event in app.stream(
        make_initial_state(case4_order), config=config_demo2, stream_mode="updates"
    ):
        node_name = list(event.keys())[0]
        updates = event[node_name]
        if node_name == "supervisor":
            pass  # Routing printed inside supervisor node
        elif node_name in ("customer_analyst", "address_analyst", "payment_analyst"):
            new_evidence = updates.get("evidence", [])
            if new_evidence:
                for e in new_evidence:
                    pass  # Findings printed inside specialist node

    # Check if paused at human_review
    state_demo2 = app.get_state(config_demo2)
    hit_interrupt = bool(state_demo2.next)

    if hit_interrupt:
        print(f"\n  Graph paused at: {state_demo2.next}")

        if state_demo2.tasks and state_demo2.tasks[0].interrupts:
            review_package = state_demo2.tasks[0].interrupts[0].value
            print(f"\n  Review package sent to human:")
            print(f"    Order:    {review_package['order_id']}")
            print(f"    Customer: {review_package['customer']}")
            print(f"    Amount:   ${review_package['amount']:.2f}")
            print(f"    Score:    {review_package['risk_score']}/100")
            print(f"    Decision: {review_package['decision']}")
            print(f"    Specialists: {review_package.get('specialists_consulted', [])}")
            print(f"    Evidence:")
            for line in review_package['evidence_summary']:
                print(f"      {line}")

        # Resume with human approval
        print(f"\n  Human decides: APPROVED")
        print(f"  Notes: \"Multi-agent investigation confirmed — warehouse is customer's office.\"")

        for event in app.stream(
            Command(resume={
                "decision": "approved",
                "notes": "Multi-agent investigation confirmed — warehouse is customer's office.",
            }),
            config=config_demo2,
            stream_mode="values",
        ):
            pass

        final_demo2 = app.get_state(config_demo2)
        print(f"\n  Final result:")
        print(f"    Decision:       {final_demo2.values.get('decision', '').upper()}")
        print(f"    Score:          {final_demo2.values.get('risk_score', 0)}/100")
        print(f"    Human Decision: {final_demo2.values.get('human_decision', '')}")
        print(f"    Specialists:    {final_demo2.values.get('specialist_log', [])}")
    else:
        final_demo2 = app.get_state(config_demo2)
        print(f"\n  Completed without HITL interrupt.")
        print(f"    Decision:    {final_demo2.values.get('decision', '').upper()}")
        print(f"    Score:       {final_demo2.values.get('risk_score', 0)}/100")
        print(f"    Specialists: {final_demo2.values.get('specialist_log', [])}")

    # ══════════════════════════════════════════════════════════
    # Demo 3: All 6 Cases
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  Demo 3: All 6 Cases — Multi-Agent Graph")
    print(f"{'=' * 60}")

    multi_results = []
    for i, (name, order) in enumerate(ALL_CASES, 1):
        thread_id = f"demo3-case-{i}"
        config_case = make_config(thread_id)

        print(f"\n  >> {name}")

        for event in app.stream(
            make_initial_state(order), config=config_case, stream_mode="values"
        ):
            pass

        state = app.get_state(config_case)
        hit_interrupt = bool(state.next)

        if hit_interrupt:
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
        specialists = state.values.get("specialist_log", [])

        multi_results.append({
            "name": name,
            "decision": decision,
            "score": score,
            "evidence": evidence_count,
            "loops": loops,
            "tokens": tokens,
            "hitl": hit_interrupt,
            "human_decision": human,
            "specialists": specialists,
        })

        hitl_str = f" + human={human}" if human else ""
        spec_str = ", ".join(specialists)
        print(f"     -> {decision.upper()} (score: {score}) | "
              f"{loops} loops, {tokens:,} tokens | specialists: [{spec_str}]{hitl_str}")

    # Summary table
    print(f"\n{'=' * 90}")
    print("  Phase 6 Results Summary")
    print(f"{'=' * 90}")
    print(f"  {'Case':<30} {'Decision':<12} {'Score':>5} {'HITL?':>6} "
          f"{'Specialists':<35} {'Tokens':>8}")
    print(f"  {'─' * 89}")
    for r in multi_results:
        hitl_str = "Yes" if r["hitl"] else "No"
        spec_str = ", ".join(r["specialists"])
        print(f"  {r['name']:<30} {r['decision'].upper():<12} {r['score']:>5} "
              f"{hitl_str:>6} {spec_str:<35} {r['tokens']:>8,}")

    # ══════════════════════════════════════════════════════════
    # Demo 4: Unit Economics — Single-Agent vs Multi-Agent
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  Demo 4: Unit Economics — Single-Agent vs Multi-Agent")
    print(f"  Import Phase 5's graph, run all 6 cases, compare.")
    print(f"{'=' * 60}")

    # Import Phase 5's build_graph using importlib to avoid module name collision
    import importlib.util
    phase5_path = os.path.join(os.path.dirname(__file__), "..", "phase5-workflows", "graph.py")
    phase5_path = os.path.abspath(phase5_path)
    spec = importlib.util.spec_from_file_location("phase5_graph", phase5_path)
    phase5_graph = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase5_graph)
    build_graph_v5 = phase5_graph.build_graph
    make_initial_state_v5 = phase5_graph.make_initial_state

    checkpointer_v5 = MemorySaver()
    app_v5 = build_graph_v5(checkpointer=checkpointer_v5)

    single_results = []
    print(f"\n  Running all 6 cases with Phase 5 (single-agent)...")
    for i, (name, order) in enumerate(ALL_CASES, 1):
        thread_id = f"demo4-single-case-{i}"
        config_case = {
            "recursion_limit": 25,
            "configurable": {
                "thread_id": thread_id,
                "max_loops": 10,
                "token_budget": 100_000,
            },
        }

        for event in app_v5.stream(
            make_initial_state_v5(order), config=config_case, stream_mode="values"
        ):
            pass

        state = app_v5.get_state(config_case)
        hit_interrupt = bool(state.next)

        if hit_interrupt:
            for event in app_v5.stream(
                Command(resume={"decision": "approved", "notes": "Auto-approved."}),
                config=config_case,
                stream_mode="values",
            ):
                pass
            state = app_v5.get_state(config_case)

        single_results.append({
            "name": name,
            "decision": state.values.get("decision", "?"),
            "score": state.values.get("risk_score", 0),
            "tokens": state.values.get("tokens_used", 0),
            "loops": state.values.get("loop_count", 0),
        })

    # Side-by-side comparison
    print(f"\n{'─' * 95}")
    print(f"  {'':30} {'Single-Agent (Phase 5)':<30} {'Multi-Agent (Phase 6)':<30}")
    print(f"  {'Case':<30} {'Decision':<10} {'Score':>5} {'Tokens':>8}   "
          f"{'Decision':<10} {'Score':>5} {'Tokens':>8}")
    print(f"  {'─' * 94}")

    total_single_tokens = 0
    total_multi_tokens = 0
    for s, m in zip(single_results, multi_results):
        total_single_tokens += s["tokens"]
        total_multi_tokens += m["tokens"]
        print(f"  {s['name']:<30} {s['decision'].upper():<10} {s['score']:>5} "
              f"{s['tokens']:>8,}   {m['decision'].upper():<10} {m['score']:>5} "
              f"{m['tokens']:>8,}")

    print(f"  {'─' * 94}")
    print(f"  {'TOTAL':<30} {'':10} {'':>5} {total_single_tokens:>8,}   "
          f"{'':10} {'':>5} {total_multi_tokens:>8,}")

    # Cost estimate
    # Sonnet: ~$3/MTok input, ~$15/MTok output (blended ~$6/MTok)
    # Haiku: ~$0.80/MTok input, ~$4/MTok output (blended ~$1.50/MTok)
    # Rough: single-agent is all Sonnet, multi-agent splits Sonnet (supervisor) + Haiku (specialists)
    sonnet_cost_per_mtok = 6.0  # blended
    haiku_cost_per_mtok = 1.5   # blended

    single_cost = total_single_tokens / 1_000_000 * sonnet_cost_per_mtok
    # For multi-agent, supervisor tokens are Sonnet, specialist tokens are Haiku
    # We don't track separately, so estimate: supervisor ~40% of tokens, specialists ~60%
    multi_supervisor_tokens = total_multi_tokens * 0.4
    multi_specialist_tokens = total_multi_tokens * 0.6
    multi_cost = (
        multi_supervisor_tokens / 1_000_000 * sonnet_cost_per_mtok
        + multi_specialist_tokens / 1_000_000 * haiku_cost_per_mtok
    )

    print(f"\n  Estimated cost (6 cases):")
    print(f"    Single-agent (all Sonnet):           ~${single_cost:.4f}")
    print(f"    Multi-agent (Sonnet+Haiku, ~40/60):  ~${multi_cost:.4f}")

    # ══════════════════════════════════════════════════════════
    # Demo 5: When Multi-Agent Earns Its Keep (Narrative)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  Demo 5: When Multi-Agent Earns Its Keep")
    print(f"{'=' * 70}")

    print("""
  The honest assessment:

  WITH 5 TOOLS (our demo):
  - One agent handles everything. The supervisor adds coordination overhead
    without improving investigation quality.
  - Every supervisor call is a Sonnet call just to decide "which of 3
    specialists to call next" — work the single agent does implicitly.
  - Token count may be HIGHER because the supervisor repeats context each loop.

  WITH 200 TOOLS (production scale):
  - A single agent seeing 200 tool descriptions has degraded tool selection.
    It picks wrong tools, hallucinates arguments, and wastes tokens.
  - Specialists with 10-15 domain tools each make better, faster decisions.
  - The supervisor only needs to know "which domain" — not "which specific tool."

  THE REAL COST OPTIMIZATION:
  - Haiku specialists at ~$0.80/MTok input vs Sonnet at ~$3/MTok input.
  - Push commodity work (run these 2 tools, report findings) to cheap models.
  - Reserve expensive reasoning for the supervisor's synthesis and routing.
  - At scale, this 4-10x cost reduction per specialist call adds up.

  IF ALL AGENTS USE THE SAME MODEL:
  - You've built orchestration theatre. The overhead of coordination
    (extra LLM calls for routing) outweighs any benefit.
  - Heterogeneous models (different capabilities, different costs) are
    the only honest justification for multi-agent at small scale.

  QUICK PROTOTYPING:
  - The `langgraph-supervisor` package exists for quick multi-agent setups.
  - Hand-built (like this phase) gives more control over routing logic,
    state management, and specialist prompts.
  - Start with hand-built to understand the pattern, then evaluate whether
    the package's abstractions help or hide important details.
""")

    # ── Final Summary ────────────────────────────────────────
    print(f"{'=' * 70}")
    print("Done. All 5 demos complete.")
    print(f"{'=' * 70}")
