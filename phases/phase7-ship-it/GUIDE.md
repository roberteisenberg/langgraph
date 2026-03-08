# Phase 7: Ship It — The Other 70%

## What This Phase Does

This phase covers making a LangGraph app ready for production across three concerns:

1. **Schema hardening** — TypedDict → Pydantic. Catch bad data at boundaries before it corrupts scores.
2. **Context engineering** — manage growing conversations so the LLM doesn't lose track of its investigation.
3. **Evaluation harness** — regression testing for non-deterministic systems.

No new graph topology. This phase hardens and validates everything built in Phases 1-6.

## Why This Phase Exists

Phases 1-6 run in a controlled tutorial environment with 6 test cases and clean inputs. Production is different: malformed orders arrive, conversations grow past context limits, and LLM non-determinism means the same input can produce slightly different investigations. Phase 7 addresses the gap between "it works in the demo" and "it works in production."

## Section 1: Schema Hardening — TypedDict → Pydantic

### The Problem

Phases 3-6 use `TypedDict` for state and evidence. TypedDict is a type hint — it documents structure but performs no runtime validation. At runtime, any value goes:

```python
class Evidence(TypedDict):
    tool: str
    finding: str
    risk_signal: str       # "banana" is fine at runtime
    confidence: float      # 500.0 is fine at runtime
    raw_data: dict
    timestamp: str
```

This means a hallucinated tool name, a nonsensical risk signal, or an out-of-range confidence value all pass silently into the scoring formula. The score clamps to 0-100, so corrupted inputs produce *plausible but wrong* results — the hardest kind of bug to catch.

### The Fix

Pydantic models validate at creation time:

```python
class EvidenceModel(BaseModel):
    tool: str                    # validated against KNOWN_TOOLS
    finding: str = Field(min_length=1)
    risk_signal: Literal["low_risk", "medium_risk", "high_risk", "neutral", "error"]
    confidence: float = Field(ge=0.0, le=1.0)
    raw_data: dict = Field(default_factory=dict)
    timestamp: str
```

A `@field_validator` on `tool` rejects unknown tool names. `risk_signal` is constrained to the five valid values. `confidence` must be between 0.0 and 1.0. Bad data raises `ValidationError` instead of silently corrupting the score.

`OrderModel` validates inputs at the other boundary — malformed orders (empty IDs, invalid emails, negative amounts, no items) are rejected before investigation begins.

### When to Use Which

- **TypedDict** — inside a single graph where you control all the data. Lightweight, no overhead.
- **Pydantic** — at system boundaries: API input, tool output, cross-agent state, anything from an external source. Trust nothing at the edges.

### What the Demo Shows

The demo runs four comparisons:
1. **Valid evidence** — both TypedDict and Pydantic accept it
2. **Garbage evidence** — TypedDict silently accepts `confidence=500.0` and `risk_signal="banana"`. Pydantic rejects with specific error messages.
3. **Scoring with corrupted evidence** — unvalidated scoring clamps the inflated result to 100. Hardened scoring rejects before calculating.
4. **Order validation** — a malformed order with 6 errors is caught at the boundary

## Section 2: Context Engineering

### The Problem

A single investigation generates 10-30 messages (system prompt, LLM reasoning, tool calls, tool results). Multi-agent investigations with three specialists can produce 50+. At 200 tools, conversations easily exceed context limits. The LLM starts losing track of early findings, repeating tool calls, or forgetting its investigation plan.

### The Fix: Two-Layer State Design

The state design from Phase 3 pays off here. State has two layers:

- **`messages`** — the LLM's scratchpad. Verbose, growing, compressible. This is the conversation history: reasoning, tool calls, tool results.
- **`evidence`** — the structured audit trail. Compact, append-only, never compressed. Each piece is an `Evidence` record with tool, finding, risk signal, confidence, and timestamp.

These two layers serve different purposes and have different lifecycles:

**`trim_messages_to_recent`** keeps the system prompt and the most recent N messages, dropping the oldest non-system messages. The LLM loses its early reasoning but retains the system prompt and recent context.

```python
def trim_messages_to_recent(messages, max_messages=20):
    # Keep system messages + last N non-system messages
```

**`summarize_evidence`** compresses the evidence list into a concise text summary that can be injected into a trimmed conversation. The LLM sees what was found without replaying 30 tool calls:

```
  [ok] check_customer_history: low_risk (0.95) -- 50 prior orders, low risk customer
  [!!] verify_shipping_address: high_risk (0.95) -- Warehouse address, high geo-risk
  [! ] check_payment_pattern: medium_risk (0.80) -- Amount above typical range
  [--] search_fraud_database: neutral (0.70) -- No exact match found
```

### The Key Insight

Trim the reasoning. Keep the record. Messages are the LLM's working memory — they can be summarized and compressed. Evidence is the audit trail — it must be preserved intact. The two-layer design means you never have to choose between context limits and auditability.

### What the Demo Shows

The demo simulates a 40-message investigation, trims it to 10, and shows the evidence summary that replaces the dropped messages.

## Section 3: Evaluation Harness

### The Problem

LLM-based systems are non-deterministic. The same input can produce slightly different investigation paths — tools called in different order, different wording in findings, minor confidence variations. How do you know a code change didn't break something when "correct" output varies between runs?

### The Approach

Phase 3 established the baseline: 6 cases, each with an expected decision and score. Every subsequent phase must match these results. The evaluation harness formalizes this:

```python
EXPECTED_RESULTS = {
    "Case 1: Obviously Legit":      {"decision": "approve", "score": 0},
    "Case 2: Mildly Suspicious":    {"decision": "approve", "score": 33},
    "Case 3: High Risk":            {"decision": "reject",  "score": 100},
    "Case 4: Conflicting Signals":  {"decision": "review",  "score": 61},
    "Case 5: Historical Fraud":     {"decision": "approve", "score": 10},
    "Case 6: Tool Error":           {"decision": "approve", "score": 6},
}
```

For each case, the harness checks:
- **Decision matches** — approve/review/reject must be identical
- **Score within tolerance** — default ±5 points to account for minor LLM non-determinism in evidence collection order

A case passes only if both checks succeed.

### What the Harness Catches

The evaluation harness catches exactly the kind of bug that Phase 6's duplicate evidence problem demonstrated: results that are *plausible but wrong*. Case 4 scoring 80 instead of 61 produces a defensible "reject" — you'd never catch it by eyeballing output. The harness compares against the baseline and flags it immediately.

### HITL Auto-Approval

The harness handles checkpointed graphs (Phases 5-6) by auto-approving HITL interrupts. When Case 4 pauses for human review, the harness resumes with `Command(resume={"decision": "approved", "notes": "Auto-approved by eval harness"})`. This lets the full pipeline run without manual intervention.

### Cross-Phase Comparison

After running Phase 6 through the harness, the demo prints a comparison table showing token usage and estimated cost between single-agent (Phase 5) and multi-agent (Phase 6), reinforcing the cost optimization story from Phase 6.

### What the Demo Shows

The harness loads Phase 6's graph, runs all 6 cases, auto-approves HITL, compares against baselines, and prints a pass/fail table with score diffs.

## The Moving Parts — What Phase 7 Adds

Phase 7 doesn't change the graph topology. It adds hardening *around* the existing graph:

| Concern | What It Does | Where It Applies |
|---------|-------------|-----------------|
| Schema hardening | Validates data at creation time | System boundaries — input orders, tool output, cross-agent state |
| Context trimming | Keeps conversations within context limits | Long investigations, multi-agent chains |
| Evidence summary | Compresses findings for trimmed contexts | Injected after trimming to preserve investigation context |
| Evaluation harness | Regression tests against Phase 3 baseline | Every phase, every code change |

## The Honest Assessment

Phase 7's demo ends with an honest look at the full tutorial arc:

- **Phase 1** is a pipeline. LangGraph adds nothing.
- **Phase 2** introduces tools. The loop appears. Now LangGraph matters.
- **Phase 3** is the hinge — the ReAct agent investigates dynamically.
- **Phase 4** adds infrastructure for LLM-driven loops.
- **Phase 5** adds checkpointing and HITL — the killer feature for production workflows.
- **Phase 6** adds multi-agent coordination with cost optimization.
- **Phase 7** adds the hardening that makes all of it shippable.

The tensions that run through the tutorial:

1. **Framework vs. simplicity** — LangGraph earns its keep at Phase 3 (the loop). Before that, it's overhead.
2. **Single-agent vs. multi-agent** — at 5 tools, single-agent wins on simplicity. At 200+ tools, specialists earn their keep.
3. **TypedDict vs. Pydantic** — TypedDict inside the graph, Pydantic at the boundaries.
4. **Demo vs. production** — Phases 1-6 are clean. Phase 7 is the ugly reality.

## How to Run

```bash
python3 phases/phase7-ship-it/graph.py
```

Runs 4 demos:
1. **Schema hardening** — TypedDict vs Pydantic comparison (no API calls)
2. **Context engineering** — message trimming and evidence summarization (no API calls)
3. **Evaluation harness** — Phase 6 regression test against baseline (API calls)
4. **Honest assessment** — what we learned across 7 phases (no API calls)
