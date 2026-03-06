# Phase 7: Ship It (The Other 70%) — Results

## What This Phase Does

No new graph topology. Phase 7 hardens and validates Phases 1-6 with three production concerns:

1. **Schema hardening** — TypedDict → Pydantic validation at system boundaries
2. **Context engineering** — trim_messages for growing conversations
3. **Evaluation harness** — regression testing for non-deterministic systems

## Demo 1: Schema Hardening

TypedDict (Phases 3-6) performs no runtime validation. Pydantic catches bad data at creation time.

| Test | TypedDict | Pydantic |
|------|-----------|----------|
| Valid evidence | Created (no validation) | Created (validated) |
| confidence=500.0 | Created silently | REJECTED: must be <= 1.0 |
| risk_signal="banana" | Created silently | REJECTED: must be low_risk/medium_risk/high_risk/neutral/error |
| tool="hallucinated_tool" | Created silently | REJECTED: unknown tool |
| finding="" | Created silently | REJECTED: min length 1 |

**Scoring impact:** An unvalidated confidence=500.0 on a high_risk signal produces a raw score of 15,020 (clamped to 100). The hardened scoring function rejects it before calculation.

**Order validation:** OrderModel catches 6 errors on a malformed order (empty id, invalid email, negative amount, empty items, short address, empty name).

**Takeaway:** TypedDict trusts your code. Pydantic trusts nothing. Use Pydantic at system boundaries (API input, tool output, cross-agent state). Inside a single graph, TypedDict is fine.

## Demo 2: Context Engineering

With 200 tools, a single investigation can generate 50+ messages. The two-layer state design handles this:

| Layer | Purpose | Compression |
|-------|---------|-------------|
| `messages` | LLM reasoning scratchpad | Trim to recent N messages |
| `evidence` | Structured audit trail | Never compress |

**trim_messages_to_recent(max=10):** 40 messages → 10, system prompt preserved, 30 oldest dropped.

**summarize_evidence:** Produces a compact summary that can replace dropped messages:
```
[ok] check_customer_history: low_risk (0.95) -- 50 prior orders, low risk customer
[!!] verify_shipping_address: high_risk (0.95) -- Warehouse address, high geo-risk
[! ] check_payment_pattern: medium_risk (0.80) -- Amount above typical range
[--] search_fraud_database: neutral (0.70) -- No exact match found
```

The evidence layer is the audit trail — structured, compact, never compressed. The message layer is the scratchpad — verbose, compressible, disposable.

## Demo 3: Evaluation Harness

Ran Phase 6 (multi-agent) against the Phase 3 baseline. All 6 cases match exactly.

| Case | Expected | Actual | Diff | Result |
|------|----------|--------|------|--------|
| 1: Obviously Legit | approve(0) | approve(0) | +0 | PASS |
| 2: Mildly Suspicious | approve(33) | approve(33) | +0 | PASS |
| 3: High Risk | reject(100) | reject(100) | +0 | PASS |
| 4: Conflicting Signals | review(61) | review(61) | +0 | PASS |
| 5: Historical Fraud | approve(10) | approve(10) | +0 | PASS |
| 6: Tool Error | approve(6) | approve(6) | +0 | PASS |

**6/6 passed. No regressions detected.**

### Cross-Phase Comparison

| Phase | Architecture | Tokens | ~Cost | Decisions |
|-------|-------------|--------|-------|-----------|
| Phase 5 | Single-agent (all Sonnet) | 76,139 | ~$0.23 | 6/6 correct |
| Phase 6 | Multi-agent (Sonnet + Haiku) | 45,221 | ~$0.07 | 6/6 correct |

Multi-agent saves 41% tokens and ~67% cost with identical decisions and scores.

Wall time: 151.8s (25.3s per case).

### What the Eval Harness Catches

The evaluation framework compares each result against a known-good baseline (Phase 3). It catches:

- **Decision flips** — approve/review/reject changes (e.g., the duplicate evidence bug in Phase 6 flipped Case 4 from REVIEW to REJECT)
- **Score drift** — scores outside ±5 tolerance (e.g., Case 4 scoring 71 instead of 61 before the tool-overlap fix)
- **Errors** — tool failures, graph crashes, timeout exceptions

Without this harness, the Phase 6 duplicate evidence bug would have shipped. The scores were plausible, the decisions were defensible, and the system ran without errors. The only signal was a 19-point score difference that required comparing against the baseline.

## The Full Story Arc

| Phase | What it teaches | Key addition |
|-------|----------------|--------------|
| 1 | StateGraph basics | Pipeline (LangGraph adds nothing) |
| 2 | Tools + the loop | Dynamic tool selection |
| 3 | Full ReAct agent | THE HINGE — LangGraph earns its keep |
| 4 | Infrastructure | Viz, streaming, guardrails, token budgets |
| 5 | Checkpointing + HITL | Pause for humans, resume with context |
| 6 | Multi-agent | Supervisor + specialists, cost optimization |
| 7 | The other 70% | Validation, context management, evaluation |

## Tensions Resolved

| Tension | Resolution |
|---------|------------|
| Framework vs. simplicity | LangGraph earns its keep at Phase 3 (the loop). Before that, overhead. |
| Prebuilt vs. hand-built | Hand-built for control. create_react_agent for prototyping. |
| Single vs. multi-agent | 5 tools: single agent. 200 tools: specialists. |
| LangChain dependency | langchain-core for message types is the irreducible minimum. |
| Teaching vs. production | Phases 1-6: clean demos. Phase 7: ugly reality. Both necessary. |

## Honest Assessment

With 5 tools and 6 test cases, the full 7-phase progression demonstrates that:

1. **LangGraph's value is the loop + infrastructure.** The ReAct cycle (Phase 3) is where it earns its keep. Viz, streaming, guardrails (Phase 4) are why you'd pick it over a plain `while` loop. HITL (Phase 5) is the production killer feature.

2. **Multi-agent is a cost optimization, not a capability upgrade.** At 5 tools, a single agent handles everything. At 200 tools, specialists with 10-15 domain tools each make better selection decisions and cost less (Haiku vs Sonnet). But multi-agent introduces cross-agent evidence correlation — a new bug class that requires baseline comparison to catch.

3. **The other 70% is real.** Pydantic validation catches silent corruption. Context trimming prevents context window overflow. Regression testing catches plausible-but-wrong results. None of this is glamorous. All of it is necessary.

"If you can draw your workflow as a straight line, do not use a graph framework."
