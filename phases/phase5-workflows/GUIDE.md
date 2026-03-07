# Phase 5: Real Workflows — Checkpointing + Human-in-the-Loop

## What This Phase Does

Phase 3 built a complete fraud investigator. Phase 4 added infrastructure (visualization, streaming, guardrails, token budgets) inside the same graph shape. But both phases run to completion without stopping — there's no way for a human to intervene.

Phase 5 adds **checkpointing** and builds two capabilities on top of it:
- **Human-in-the-loop (HITL)** — pause the investigation for human review on ambiguous cases, resume with the analyst's decision
- **State forking** — "what-if" analysis by forking from a checkpoint and injecting modified evidence

These are the production workflow patterns. Every workflow engine (Temporal, Dapr Workflow) has equivalents. LangGraph's version is well-integrated with its graph model, but architecturally, this is "save game" with extras.

## Architecture

```
START -> parse_order -> call_llm -> [should_continue]
                           ^         |-- has tool calls -> execute_tools -> call_llm (loop)
                           |         +-- done -> assess_risk -> [route_after_assessment]
                           |                                     |-- review -> human_review (INTERRUPT) -> format_report -> END
                           +---------+---------------------------+-- approve/reject -> format_report -> END
```

Same investigation loop as Phase 3. The new branch: after `assess_risk`, if the decision is "review" (score 50-79), the graph routes to `human_review` which calls `interrupt()` — pausing the entire graph until a human resumes it.

## What's New Since Phase 3

- **Checkpointing** — `MemorySaver` (dev) / `PostgresSaver` (production). Every node completion creates a checkpoint. Thread IDs identify conversation continuity.
- **`interrupt()`** — dynamic breakpoint inside a node. The graph pauses, persists its state, and waits. Hours or days later, a human resumes with `Command(resume={"decision": "approved", "notes": "..."})`.
- **`route_after_assessment`** — new conditional edge after `assess_risk`. Score 50-79 routes to `human_review`, everything else goes straight to `format_report`.
- **`human_review` node** — prepares a review package (order summary, evidence, score), calls `interrupt()`, processes the human's response on resume.
- **State forking** — `get_state_history()` to find a checkpoint, `update_state()` to inject modified evidence, re-run from that point. Original investigation preserved, corrected branch runs independently.

### HITL Trigger Design

Not every decision needs a human:
- **reject** (score >= 80): Evidence is overwhelming. Human review adds friction, not value.
- **review** (score 50-79): Genuinely ambiguous. This is where humans add judgment.
- **approve** (score < 50): Low-risk, automatic.

Of the 6 test cases, only **Case 4** (conflicting signals, score 61) triggers HITL.

### Critical Gotcha: No Side Effects Before `interrupt()`

Everything before `interrupt()` in a node **re-executes on resume**. If you make an API call, charge a credit card, or send an email before the interrupt, it happens again when the human resumes. Keep side effects after the interrupt, or make them idempotent.

## The Moving Parts — Progressive Growth

| # | Moving Part | Phase 0 | Phase 1 | Phase 2 | Phase 3 | Phase 5 |
|---|-------------|---------|---------|---------|---------|---------|
| 1 | **Nodes** | 0 — no graph | 3 (parse, score, format) | 3 (call_llm, tools, format) | 5 (parse, call_llm, execute_tools, assess_risk, format_report) | **6 (+human_review)** |
| 2 | **Edges** | 0 — no graph | Linear: A->B->C | Loop: call_llm<->tools | Same loop | Same + human_review->format_report |
| 3 | **Conditional edges** | 0 — no graph | 1 (node dependency — score threshold) | 1 (LLM-driven — has tool calls?) | 1 (LLM-driven — should_continue) | **2 (+route_after_assessment)** |
| 4 | **Cases** | 6 | 6 | 6 | 6 | 6 — Case 4 now triggers HITL |
| 5 | **Tools** | 0 | 0 | 3 | 5 | 5 |
| 6 | **LLM** | 1 (Sonnet) | 1 (Sonnet) | 1 (Sonnet) | 1 (Sonnet) | 1 (Sonnet) |
| 7 | **Prompts** | 1 (scorer) | 1 (scorer) | 1 (investigator) | 1 (investigator v2) | 1 (same investigator) |

**What changed:** One new node (row 1) and one new conditional edge (row 3). That's it. The investigation loop is identical to Phase 3. The complexity isn't in new moving parts — it's in the new *capability* that checkpointing enables: pause, persist, resume, fork.

## Results

| Case | Decision | Score | HITL? | Human Decision | Tokens |
|------|----------|-------|-------|----------------|--------|
| 1: Obviously Legit | APPROVE | 0 | No | — | 11,075 |
| 2: Mildly Suspicious | APPROVE | 33 | No | — | 13,468 |
| 3: High Risk | REJECT | 100 | No | — | 13,751 |
| 4: Conflicting Signals | REVIEW | 61 | **Yes** | approved | 13,669 |
| 5: Historical Fraud | APPROVE | 10 | No | — | 13,416 |
| 6: Tool Error | APPROVE | 6 | No | — | 10,760 |

All scores match Phase 3 exactly. The investigation produces the same evidence and the same scores — checkpointing doesn't change what the agent finds, only what happens after scoring.

## The Demos

Phase 5 has 6 demos that build on each other:

1. **Checkpointing basics** — run Case 1 with `MemorySaver`, inspect 16 checkpoints (one per node completion)
2. **HITL: human approves** — Case 4 pauses at score 61, analyst approves
3. **HITL: human rejects** — Case 4 pauses again, analyst overrides to reject (account takeover suspicion)
4. **State forking / what-if** — fork from pre-review checkpoint, remove high-risk evidence, re-score: 61 -> 22 (approve). Shows how the same investigation could have gone differently.
5. **Compile-time breakpoints** — `interrupt_before=["format_report"]` pauses *every* case unconditionally, vs `interrupt()` which pauses conditionally
6. **Full run** — all 6 cases, auto-approve Case 4's HITL

## What This Is vs What Production Looks Like

### Checkpointing: MemorySaver Is a Demo Trap

`MemorySaver` stores checkpoints in process memory. When the process dies, everything is gone. Multiple developers have been burned deploying prototypes that worked in dev and lost all state on first restart. In production, you'd use `PostgresSaver` — same interface, durable backend. Every workflow engine (Temporal, Dapr Workflow) has the same pattern: serialize state to a persistent store so it survives process restarts. LangGraph calls it checkpointing. Architecturally, it's "save game."

### HITL: Simulated Here, Queued in Production

In this tutorial, the human decision is a Python function call — `Command(resume={"decision": "approved", "notes": "..."})` runs in the same script, immediately after the interrupt. This simulates the workflow so you can see it end-to-end without standing up a queue.

In a real system, `interrupt()` sends a review package to a queue — email, Slack, a dashboard. The graph is serialized to the checkpoint store and the process moves on. Hours or days later, when an analyst makes a decision, a different process loads the checkpoint and resumes with `Command(resume=...)`. The code is identical. The difference is time, process boundaries, and a persistent backend instead of `MemorySaver`.

### What's First-Class and What Isn't

Workflows and HITL are first-class citizens in LangGraph — `interrupt()`, `Command(resume=...)`, conditional routing, state forking. These are built into the framework and work well.

Persistence is not. LangGraph provides the checkpointing interface and some implementations, but the infrastructure — the database, its lifecycle, cross-investigation memory — is yours.

## How to Run

```bash
python3 phases/phase5-workflows/graph.py
```
