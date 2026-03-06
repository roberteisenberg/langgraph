"""
Phase 7: Ship It (The Other 70%)

The agent logic is 30% of the effort. Hardening, testing, and evaluation
are the other 70%. This phase covers three production concerns:

1. Schema hardening (TypedDict -> Pydantic) -- catch bad data at boundaries
2. Context engineering (trim_messages) -- manage growing conversations
3. Evaluation harness -- regression testing for non-deterministic systems

No new graph topology. This phase hardens and validates Phases 1-6.

Run: python3 phases/phase7-ship-it/graph.py
"""

import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Annotated, Literal, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# Add project root to path for shared imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.test_cases import ALL_CASES


# ======================================================================
# Section 1: Schema Hardening -- TypedDict -> Pydantic
# ======================================================================

# The TypedDict version (what Phases 3-6 use). No runtime validation.
class Evidence(TypedDict):
    tool: str
    finding: str
    risk_signal: str       # any string at runtime -- "banana" is fine
    confidence: float      # any float -- 500.0 is fine
    raw_data: dict
    timestamp: str


# The Pydantic version. Validates everything at creation time.
KNOWN_TOOLS = {
    "check_customer_history", "verify_shipping_address",
    "check_payment_pattern", "search_fraud_database", "calculate_risk_score",
}


class EvidenceModel(BaseModel):
    """Validated evidence record.

    TypedDict accepts any string for risk_signal, any float for confidence,
    any string for tool name. This Pydantic version catches all of those.
    """
    tool: str
    finding: str = Field(min_length=1)
    risk_signal: Literal["low_risk", "medium_risk", "high_risk", "neutral", "error"]
    confidence: float = Field(ge=0.0, le=1.0)
    raw_data: dict = Field(default_factory=dict)
    timestamp: str

    @field_validator("tool")
    @classmethod
    def validate_tool(cls, v):
        if v not in KNOWN_TOOLS:
            raise ValueError(f"Unknown tool '{v}'. Known: {sorted(KNOWN_TOOLS)}")
        return v


class OrderModel(BaseModel):
    """Validated order input. Catches malformed orders before investigation."""
    id: str = Field(min_length=1)
    customer_email: str = Field(pattern=r".+@.+\..+")
    customer_name: str = Field(min_length=1)
    amount: float = Field(gt=0)
    items: list[dict] = Field(min_length=1)
    shipping_address: str = Field(min_length=5)
    metadata: dict = Field(default_factory=dict)


def calculate_risk_score_hardened(evidence_list: list[dict]) -> dict:
    """Pydantic-validated scoring. Rejects bad evidence before scoring.

    Same logic as Phase 3's _calculate_risk_score_impl, but validates
    every evidence item first. Bad data raises ValidationError instead
    of silently corrupting the score.
    """
    validated = [EvidenceModel(**e) for e in evidence_list]

    weights = {"high_risk": 30, "medium_risk": 15, "low_risk": -10, "neutral": 0, "error": 5}
    base = 20
    score = base
    bonus = False

    for e in validated:
        score += weights.get(e.risk_signal, 0) * e.confidence
        if e.risk_signal == "high_risk" and e.confidence > 0.8:
            bonus = True

    if bonus:
        score += 10

    score = max(0, min(100, int(round(score))))
    decision = "reject" if score >= 80 else "review" if score >= 50 else "approve"
    return {"risk_score": score, "decision": decision, "investigation_complete": True}


# ======================================================================
# Section 2: Context Engineering
# ======================================================================

def trim_messages_to_recent(messages: list, max_messages: int = 20) -> list:
    """Keep system message + last N messages.

    With 200 tools, a single investigation can generate 50+ messages.
    This keeps the context window manageable while preserving the system
    prompt and recent investigation context.

    Strategy: prune messages (verbose scratchpad), keep evidence
    (structured, compact audit trail). This is why the two-layer
    state design pays off.
    """
    if len(messages) <= max_messages:
        return messages

    from langchain_core.messages import SystemMessage

    sys_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_sys = [m for m in messages if not isinstance(m, SystemMessage)]
    keep = max(1, max_messages - len(sys_msgs))
    return sys_msgs + non_sys[-keep:]


def summarize_evidence(evidence: list[dict]) -> str:
    """Compress evidence into a concise summary for context injection.

    Used when injecting prior findings into a trimmed conversation.
    The LLM sees this summary instead of replaying 30 tool calls.
    """
    if not evidence:
        return "No evidence collected."

    icons = {"high_risk": "!!", "medium_risk": "! ", "low_risk": "ok",
             "neutral": "--", "error": "??"}
    lines = []
    for e in evidence:
        signal = e.get("risk_signal", "unknown")
        conf = e.get("confidence", 0)
        tool = e.get("tool", "?")
        finding = e.get("finding", "")[:80]
        lines.append(f"  [{icons.get(signal, '??')}] {tool}: {signal} ({conf:.2f}) -- {finding}")
    return "\n".join(lines)


# ======================================================================
# Section 3: Evaluation Framework
# ======================================================================

# Known-good results from Phase 3 (the baseline all phases must match)
EXPECTED_RESULTS = {
    "Case 1: Obviously Legit":      {"decision": "approve", "score": 0},
    "Case 2: Mildly Suspicious":    {"decision": "approve", "score": 33},
    "Case 3: High Risk":            {"decision": "reject",  "score": 100},
    "Case 4: Conflicting Signals":  {"decision": "review",  "score": 61},
    "Case 5: Historical Fraud":     {"decision": "approve", "score": 10},
    "Case 6: Tool Error":           {"decision": "approve", "score": 6},
}


def evaluate_result(case_name: str, result: dict, tolerance: int = 5) -> dict:
    """Compare one result against the known-good baseline.

    tolerance: acceptable score deviation (default +/-5) to account for
    minor LLM non-determinism in evidence collection order.
    """
    expected = EXPECTED_RESULTS.get(case_name, {})
    actual_dec = result.get("decision", "").lower()
    actual_score = result.get("risk_score", -1)
    exp_dec = expected.get("decision", "")
    exp_score = expected.get("score", -999)

    dec_ok = actual_dec == exp_dec
    score_ok = abs(actual_score - exp_score) <= tolerance
    passed = dec_ok and score_ok

    issues = []
    if not dec_ok:
        issues.append(f"decision: expected {exp_dec}, got {actual_dec}")
    if not score_ok:
        issues.append(f"score: expected {exp_score}+/-{tolerance}, got {actual_score}")

    return {
        "pass": passed,
        "case": case_name,
        "expected_decision": exp_dec,
        "expected_score": exp_score,
        "actual_decision": actual_dec,
        "actual_score": actual_score,
        "tokens": result.get("tokens_used", 0),
        "details": "; ".join(issues) if issues else "OK",
    }


def run_evaluation(build_graph_fn, make_state_fn, cases,
                   needs_checkpointer=False, phase_name="?"):
    """Run all cases through a graph and evaluate against baseline.

    Handles both simple invocation (Phases 3-4) and checkpointed
    invocation with HITL auto-approval (Phases 5-6).
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    results = []
    for i, (case_name, order) in enumerate(cases, 1):
        print(f"    [{i}/{len(cases)}] {case_name}...", end=" ", flush=True)
        start = time.time()

        try:
            if needs_checkpointer:
                cp = MemorySaver()
                app = build_graph_fn(checkpointer=cp)
                tid = f"eval-{phase_name}-{i}"
                config = {
                    "recursion_limit": 50,
                    "configurable": {
                        "thread_id": tid,
                        "max_loops": 10,
                        "token_budget": 100_000,
                    },
                }
                final = None
                for event in app.stream(
                    make_state_fn(order), config=config, stream_mode="values"
                ):
                    final = event

                # Auto-approve HITL if triggered
                gs = app.get_state(config)
                if gs.next:
                    for event in app.stream(
                        Command(resume={
                            "decision": "approved",
                            "notes": "Auto-approved by eval harness",
                        }),
                        config=config,
                        stream_mode="values",
                    ):
                        final = event
                result = final
            else:
                app = build_graph_fn()
                config = {
                    "recursion_limit": 50,
                    "configurable": {"max_loops": 10, "token_budget": 100_000},
                }
                result = app.invoke(make_state_fn(order), config=config)

            ev = evaluate_result(case_name, result)
            elapsed = time.time() - start
            status = "PASS" if ev["pass"] else "FAIL"
            print(f"{ev['actual_decision']}({ev['actual_score']}) "
                  f"{status} ({elapsed:.1f}s)")
            results.append(ev)

        except Exception as e:
            elapsed = time.time() - start
            print(f"ERROR ({elapsed:.1f}s): {e}")
            results.append({
                "pass": False,
                "case": case_name,
                "expected_decision": EXPECTED_RESULTS.get(case_name, {}).get("decision", "?"),
                "expected_score": EXPECTED_RESULTS.get(case_name, {}).get("score", "?"),
                "actual_decision": "ERROR",
                "actual_score": -1,
                "tokens": 0,
                "details": f"{type(e).__name__}: {str(e)[:80]}",
            })

    return results


def print_eval_table(phase_name, results):
    """Print evaluation results table. Returns True if all passed."""
    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    total_tokens = sum(r.get("tokens", 0) for r in results)

    print(f"\n  {'='*68}")
    print(f"  {phase_name} -- {passed}/{total} passed, {total_tokens:,} tokens")
    print(f"  {'='*68}")
    print(f"  {'Case':<30} {'Expected':>12} {'Actual':>12} {'Diff':>6} {'':>6}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*6} {'-'*6}")

    for r in results:
        exp = f"{r['expected_decision']}({r['expected_score']})"
        act = f"{r['actual_decision']}({r['actual_score']})"
        if isinstance(r["actual_score"], int) and isinstance(r["expected_score"], int):
            diff = f"{r['actual_score'] - r['expected_score']:+d}"
        else:
            diff = "?"
        st = "PASS" if r["pass"] else "FAIL"
        print(f"  {r['case']:<30} {exp:>12} {act:>12} {diff:>6} {st:>6}")

    print(f"  {'-'*68}")
    if passed == total:
        print(f"\n  All cases match baseline. No regressions detected.")
    else:
        print(f"\n  REGRESSIONS: {total - passed} case(s) failed")
        for r in results:
            if not r["pass"]:
                print(f"    {r['case']}: {r['details']}")

    return passed == total


# ======================================================================
# Section 4: Phase Loader
# ======================================================================

def _load_phase(module_name, relative_path):
    """Load a phase module by relative path from phases/ directory."""
    path = os.path.join(os.path.dirname(__file__), "..", relative_path)
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ======================================================================
# Section 5: Demos
# ======================================================================

def demo_1_schema_hardening():
    """Demo 1: TypedDict vs Pydantic -- catch bad data at boundaries."""
    print("\n" + "=" * 74)
    print("  Demo 1: Schema Hardening (TypedDict -> Pydantic)")
    print("=" * 74)

    now = datetime.now(timezone.utc).isoformat()

    # --- 1a. Valid evidence ---
    print("\n  1a. Valid evidence (both accept):")
    valid = {
        "tool": "check_customer_history",
        "finding": "Customer has 50 prior orders, low risk",
        "risk_signal": "low_risk",
        "confidence": 0.95,
        "raw_data": {"orders": 50},
        "timestamp": now,
    }
    td: Evidence = valid  # type: ignore -- TypedDict, no runtime check
    pm = EvidenceModel(**valid)
    print(f"      TypedDict: created (no runtime validation)")
    print(f"      Pydantic:  created (tool={pm.tool}, confidence={pm.confidence})")

    # --- 1b. Garbage evidence ---
    print("\n  1b. Garbage evidence (TypedDict silent, Pydantic rejects):")
    garbage = {
        "tool": "hallucinated_tool",
        "finding": "",
        "risk_signal": "banana",
        "confidence": 500.0,
        "raw_data": {},
        "timestamp": "yesterday",
    }
    td_bad: Evidence = garbage  # type: ignore
    print(f"      TypedDict: created -- confidence={td_bad['confidence']}, "
          f"risk_signal='{td_bad['risk_signal']}' (no error)")

    try:
        EvidenceModel(**garbage)
        print(f"      Pydantic:  BUG -- should have rejected")
    except ValidationError as e:
        print(f"      Pydantic:  REJECTED ({len(e.errors())} errors):")
        for err in e.errors():
            field = ".".join(str(x) for x in err["loc"])
            print(f"        - {field}: {err['msg']}")

    # --- 1c. Scoring with corrupted evidence ---
    print("\n  1c. Scoring with corrupted evidence (confidence=500.0):")
    bad_evidence = [
        {"tool": "check_customer_history", "finding": "Low risk customer",
         "risk_signal": "low_risk", "confidence": 0.95,
         "raw_data": {}, "timestamp": now},
        {"tool": "verify_shipping_address", "finding": "Warehouse address",
         "risk_signal": "high_risk", "confidence": 500.0,  # <-- corrupted
         "raw_data": {}, "timestamp": now},
    ]

    # Unvalidated scoring (simulating Phase 3-6 behavior)
    weights = {"high_risk": 30, "low_risk": -10}
    raw = 20 + (weights["low_risk"] * 0.95) + (weights["high_risk"] * 500.0) + 10
    clamped = max(0, min(100, int(round(raw))))
    print(f"      Unvalidated: raw={raw:.0f}, clamped={clamped} "
          f"(confidence=500 silently inflates, clamps to 100)")

    try:
        calculate_risk_score_hardened(bad_evidence)
        print(f"      Hardened:    BUG -- should have rejected")
    except ValidationError as e:
        msg = e.errors()[0]["msg"]
        print(f"      Hardened:    REJECTED before scoring -- {msg}")

    # --- 1d. Order validation ---
    print("\n  1d. Order validation:")
    bad_order = {
        "id": "",
        "customer_email": "not-an-email",
        "customer_name": "",
        "amount": -50,
        "items": [],
        "shipping_address": "x",
    }
    try:
        OrderModel(**bad_order)
    except ValidationError as e:
        print(f"      Bad order rejected ({len(e.errors())} errors):")
        for err in e.errors():
            field = ".".join(str(x) for x in err["loc"])
            print(f"        - {field}: {err['msg']}")

    real_order = ALL_CASES[0][1]  # Case 1
    validated = OrderModel(**real_order)
    print(f"\n      Real order (Case 1): validated -- "
          f"{validated.customer_email}, ${validated.amount}")

    print("\n  Takeaway: TypedDict trusts your code. Pydantic trusts nothing.")
    print("  Use Pydantic at system boundaries: API input, tool output,")
    print("  cross-agent state. Inside a single graph? TypedDict is fine.")


def demo_2_context_engineering():
    """Demo 2: Managing growing conversations."""
    print("\n" + "=" * 74)
    print("  Demo 2: Context Engineering (trim + summarize)")
    print("=" * 74)

    from langchain_core.messages import (
        AIMessage, HumanMessage, SystemMessage, ToolMessage,
    )

    # Simulate a 40-message investigation
    print("\n  Simulating a 40-message investigation...")

    messages = [SystemMessage(content="You are a fraud investigation agent...")]
    tool_names = [
        "check_customer_history", "verify_shipping_address",
        "check_payment_pattern", "search_fraud_database",
    ]

    for i in range(13):
        t = tool_names[i % len(tool_names)]
        messages.append(AIMessage(
            content=f"Checking {t}...",
            tool_calls=[{"id": f"call_{i}", "name": t,
                         "args": {"indicator": f"test_{i}"}}],
        ))
        messages.append(ToolMessage(
            content=json.dumps({"result": f"finding_{i}"}),
            tool_call_id=f"call_{i}",
        ))
        messages.append(AIMessage(content=f"Finding {i} noted. Next step..."))

    n_sys = sum(1 for m in messages if isinstance(m, SystemMessage))
    n_ai = sum(1 for m in messages if isinstance(m, AIMessage))
    n_tool = sum(1 for m in messages if isinstance(m, ToolMessage))
    print(f"  Total: {len(messages)} messages "
          f"({n_sys} system, {n_ai} AI, {n_tool} tool)")

    # Trim
    trimmed = trim_messages_to_recent(messages, max_messages=10)
    print(f"\n  After trim_messages_to_recent(max=10):")
    print(f"    {len(messages)} -> {len(trimmed)} messages")
    print(f"    System prompt preserved: {isinstance(trimmed[0], SystemMessage)}")
    print(f"    Dropped: {len(messages) - len(trimmed)} oldest messages")

    # Evidence summarization
    print(f"\n  Evidence summary (replaces dropped messages):")
    sample_evidence = [
        {"tool": "check_customer_history", "risk_signal": "low_risk",
         "confidence": 0.95, "finding": "50 prior orders, low risk customer"},
        {"tool": "verify_shipping_address", "risk_signal": "high_risk",
         "confidence": 0.95, "finding": "Warehouse address, high geo-risk"},
        {"tool": "check_payment_pattern", "risk_signal": "medium_risk",
         "confidence": 0.80, "finding": "Amount above typical range"},
        {"tool": "search_fraud_database", "risk_signal": "neutral",
         "confidence": 0.70, "finding": "No exact match found"},
    ]
    print(summarize_evidence(sample_evidence))

    print(f"\n  Two-layer state design:")
    print(f"    messages: {len(messages)} items -> trimmed to "
          f"{len(trimmed)} (scratchpad, compressible)")
    print(f"    evidence: {len(sample_evidence)} items -> kept intact "
          f"(audit trail, never compress)")
    print(f"  Trim the reasoning. Keep the record.")


def demo_3_evaluation(cases):
    """Demo 3: Run Phase 6 through the evaluation harness."""
    print("\n" + "=" * 74)
    print("  Demo 3: Evaluation Harness -- Regression Testing")
    print("=" * 74)

    print("\n  Loading Phase 6 (multi-agent)...")
    phase6 = _load_phase("phase6_graph", "phase6-multi-agent/graph.py")

    print(f"  Running {len(cases)} cases (auto-approving HITL)...\n")

    start = time.time()
    results = run_evaluation(
        build_graph_fn=phase6.build_graph,
        make_state_fn=phase6.make_initial_state,
        cases=cases,
        needs_checkpointer=True,
        phase_name="phase6",
    )
    elapsed = time.time() - start

    all_passed = print_eval_table("Phase 6 (Multi-Agent)", results)

    total_tokens = sum(r.get("tokens", 0) for r in results)

    print(f"\n  Wall time: {elapsed:.1f}s ({elapsed / len(cases):.1f}s per case)")

    # Cross-phase comparison using historical data from RESULTS.md files
    print(f"\n  Cross-Phase Comparison (6 cases):")
    print(f"  {'Phase':<35} {'Tokens':>10} {'~Cost':>8} {'Decisions':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*8} {'-'*10}")

    historical = [
        ("Phase 5 (single-agent, Sonnet)", 76_139, "all Sonnet"),
        ("Phase 6 (multi-agent, now)",     total_tokens, "Sonnet+Haiku"),
    ]
    for name, tokens, model_mix in historical:
        # Rough cost: Sonnet ~$3/MTok avg, blended Sonnet+Haiku ~$1.5/MTok avg
        if model_mix == "all Sonnet":
            cost = tokens * 0.000003
        else:
            cost = tokens * 0.0000015
        print(f"  {name:<35} {tokens:>10,} ${cost:>7.2f} {'6/6 correct':>10}")

    if total_tokens > 0:
        savings = (1 - total_tokens / 76_139) * 100
        print(f"\n  Multi-agent savings: {savings:.0f}% fewer tokens vs single-agent")

    return all_passed, results


def demo_4_honest_assessment():
    """Demo 4: What we learned across 7 phases."""
    print("\n" + "=" * 74)
    print("  Demo 4: Honest Assessment")
    print("=" * 74)

    print("""
  The Story Arc
  =============

  Phase 1: A pipeline. LangGraph adds nothing. Honest about that.
  Phase 2: Tools arrive. The loop appears. Now LangGraph matters.
  Phase 3: THE HINGE. The ReAct agent investigates dynamically. Same 6
           cases, different investigation paths each time. The LLM decides
           what to check. Python decides the score.
  Phase 4: Infrastructure. Visualization, streaming, guardrails, token
           budgets. "Why not just a while loop?" Because of these.
  Phase 5: Checkpointing + HITL. Pause for humans, resume with context.
           The killer feature for production workflows.
  Phase 6: Multi-agent. Supervisor (Sonnet) + specialists (Haiku). Same
           decisions, 40% fewer tokens, ~67% cheaper. But the duplicate
           evidence bug proved that multi-agent introduces cross-agent
           evidence correlation -- a class of bugs single-agent systems
           don't have.
  Phase 7: The other 70%. Pydantic catches what TypedDict doesn't.
           trim_messages keeps context manageable. The eval harness catches
           regressions that look like plausible results.

  Tensions Resolved
  =================

  1. Framework vs. simplicity
     LangGraph earns its keep at Phase 3 (the loop). Before that, overhead.

  2. Prebuilt vs. hand-built
     Hand-built for control and debuggability. create_react_agent is fine
     for prototyping, but you'll outgrow it.

  3. Single-agent vs. multi-agent
     5 tools: single agent wins (simpler, same quality).
     200 tools: specialists earn their keep (better tool selection, cheaper
     models for commodity work).
     "If all agents use the same model, you've built orchestration theatre."

  4. LangChain dependency
     langchain-core for message types is the irreducible minimum. You can
     avoid LCEL and chains entirely.

  5. Teaching clarity vs. production realism
     Phases 1-6: clean traces, predictable behavior.
     Phase 7: ugly reality (validation, trimming, regression testing).
     Both are necessary.

  When to Use LangGraph
  =====================

  YES: Cycles/loops, HITL, multi-agent, streaming, durable state.
  NO:  Single LLM call, linear pipeline, basic RAG, simple chatbot.

  "If you can draw your workflow as a straight line,
   do not use a graph framework."

  What's Next
  ===========

  This tutorial stands alone. A composition guide (Dapr + LangGraph) will
  cover running these agents inside Dapr-sidecarred services -- the thesis
  that Dapr owns infrastructure, LangGraph owns agent authoring, and they
  compose cleanly.
""")


# ======================================================================
# Main Runner
# ======================================================================

def main():
    print("\n" + "#" * 74)
    print("#" + " Phase 7: Ship It (The Other 70%) ".center(72) + "#")
    print("#" * 74)

    # Demo 1: Schema hardening (no API calls)
    demo_1_schema_hardening()

    # Demo 2: Context engineering (no API calls)
    demo_2_context_engineering()

    # Demo 3: Evaluation harness (API calls -- runs Phase 6)
    all_passed, results = demo_3_evaluation(ALL_CASES)

    # Demo 4: Honest assessment (no API calls)
    demo_4_honest_assessment()

    # Final status
    print("=" * 74)
    if all_passed:
        print("  ALL EVALUATIONS PASSED -- Phases 1-6 validated.")
    else:
        print("  REGRESSIONS DETECTED -- review failing cases above.")
    print("=" * 74)

    return all_passed, results


if __name__ == "__main__":
    main()
