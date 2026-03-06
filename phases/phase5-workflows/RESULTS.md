# Phase 5: Real Workflows — Results

## Overview

Phase 5 adds three capabilities built on one mechanism (checkpointed state):
1. **Checkpointing** — save/inspect/resume graph state
2. **Human-in-the-Loop** — pause for human review on ambiguous cases, resume with decision
3. **State forking** — "what-if" analysis by forking from a checkpoint

## Graph Shape

```
START -> parse_order -> call_llm -> [should_continue]
                                     |-- has tool calls -> execute_tools -> call_llm (loop)
                                     +-- done -> assess_risk -> [route_after_assessment]
                                                                 |-- review -> human_review (INTERRUPT) -> format_report -> END
                                                                 +-- approve/reject -> format_report -> END
```

New conditional branch after `assess_risk`: "review" decisions (score 50-79) route to `human_review` (interrupt), others go straight to `format_report`.

## HITL Trigger Design

Only "review" decisions (score 50-79) trigger human review:
- **reject** (score >= 80): Evidence is overwhelming. Human review adds friction without value.
- **review** (score 50-79): Genuinely ambiguous. This is where humans add value.
- **approve** (score < 50): Low-risk, automatic.

Among the 6 test cases, only **Case 4** (conflicting signals, score 61) triggers HITL.

## Results Table

| Case | Decision | Score | HITL? | Human Decision | Tokens |
|------|----------|-------|-------|----------------|--------|
| 1: Obviously Legit | APPROVE | 0 | No | - | 11,075 |
| 2: Mildly Suspicious | APPROVE | 33 | No | - | 13,468 |
| 3: High Risk | REJECT | 100 | No | - | 13,751 |
| 4: Conflicting Signals | REVIEW | 61 | Yes | approved | 13,669 |
| 5: Historical Fraud | APPROVE | 10 | No | - | 13,416 |
| 6: Tool Error | APPROVE | 6 | No | - | 10,760 |

All scores match Phase 4 exactly. Case 4 is the only case that triggers HITL.

## Phase 4 Comparison

| Case | Phase 4 | Phase 5 | HITL? |
|------|---------|---------|-------|
| 1: Obviously Legit | APPROVE (0) | APPROVE (0) | No |
| 2: Mildly Suspicious | APPROVE (33) | APPROVE (33) | No |
| 3: High Risk | REJECT (100) | REJECT (100) | No |
| 4: Conflicting Signals | REVIEW (61) | REVIEW (61) + human | Yes |
| 5: Historical Fraud | APPROVE (10) | APPROVE (10) | No |
| 6: Tool Error | APPROVE (6) | APPROVE (6) | No |

## Demo Results

### Demo 1: Checkpointing Basics
- Case 1 ran with `MemorySaver` checkpointer
- `get_state()` shows final checkpoint with all state fields
- `get_state_history()` shows 16 checkpoints (one per node completion)
- Every node transition creates a checkpoint — "save game" after every move

### Demo 2: HITL — Human Approves
- Case 4 paused at `human_review` node
- Review package showed: order_id, customer, amount, score (61), evidence summary, options
- Resumed with `Command(resume={"decision": "approved", "notes": "..."})`
- Final report shows `human_decision = "approved"` with notes

### Demo 3: HITL — Human Rejects
- Case 4 paused at `human_review` again (new thread)
- Resumed with rejection: `{"decision": "rejected", "notes": "Internal check confirms account takeover."}`
- Decision changed from "review" to "reject" — human override applied

### Demo 4: State Forking / What-If
- Found pre-review checkpoint from Demo 2's history (score=61, decision=review)
- Filtered out high-risk evidence, recalculated: score=22, decision=approve
- Forked state injected via `update_state()` with `as_node="assess_risk"`
- Forked path routed to `format_report` directly — **skipped human_review entirely**
- Comparison: Original (61, review, HITL) vs Forked (22, approve, no HITL)

### Demo 5: Compile-Time Breakpoints
- Built graph with `interrupt_before=["format_report"]`
- Case 1 (approve, score 0) paused before format_report — unconditional
- Resumed with `None` — no human input needed
- Teaching: `interrupt_before` = ALWAYS pause. `interrupt()` = CONDITIONAL pause.

### Demo 6: Full Run — All 6 Cases
- Cases 1, 2, 3, 5, 6 completed without interruption
- Case 4 hit interrupt at `human_review`, auto-approved in demo
- All results match Phase 4 baseline

## Key Learnings

1. **Checkpointing is the foundation** — HITL and forking are both built on the same checkpoint mechanism
2. **No side effects before `interrupt()`** — the node re-executes on resume
3. **`interrupt_before` vs `interrupt()`** — compile-time (always) vs runtime (conditional)
4. **State forking** — `update_state()` + `as_node` creates a fork that respects conditional edges
5. **MemorySaver for dev, PostgresSaver for production** — same interface, different backends
6. **Evidence uses `operator.add` reducer** — `update_state` appends, so fork only injects scalar overrides

## Files

```
phases/phase5-workflows/
├── __init__.py          # empty
├── graph.py             # state, nodes, graph builder, runner with all 6 demos
├── graph.png            # Mermaid diagram export
└── RESULTS.md           # this file
```
