# Phase 6: Multi-Agent — When One Agent Isn't Enough

## Why Multi-Agent

Phase 3 proved that a single ReAct agent with 5 tools handles all 6 fraud cases correctly. The single agent works. So why split it into multiple agents?

Because **multi-agent is a fundamental pattern worth understanding on its own.** A supervisor deciding which specialist to consult. Specialists scoped to specific domains. Evidence flowing through shared state. Routing decisions based on accumulated findings. These are the building blocks of any serious agent system, and seeing them work with 5 tools and 3 specialists makes them concrete in a way that documentation alone can't.

With 5 tools, you can hold all the moving parts in your head. You can trace the supervisor's routing decisions, watch evidence accumulate from each specialist, and understand exactly why the system produced the score it did. Even at this small scale, the duplicate evidence bug (covered below) emerged — proving that multi-agent complexity is real regardless of tool count. Small scale doesn't mean simple. It means debuggable.

This understanding also prepares you for production scale. At 200 tools — identity verification, graph analysis, OSINT, compliance checks, payment forensics — a single agent's tool selection degrades. It sees 200 tool descriptions in its context window and picks poorly. Specialists with 10-15 domain tools each make better decisions because they see less and focus more. But you don't want to be learning multi-agent coordination *and* debugging tool selection noise at the same time. Internalizing the pattern at 5 tools means you're ready when scale demands it.

The cost argument reinforces the pattern: push commodity work to cheap models. A specialist that calls 1-2 tools and reports findings doesn't need Sonnet's reasoning power. Haiku does it for ~$0.25/MTok vs Sonnet's ~$3/MTok. At scale, that's a 10x cost reduction on 80% of your LLM calls.

## Architecture

```
START -> parse_order -> supervisor -> [Command routing]
                            ^         |-- customer_analyst -> supervisor
                            |         |-- address_analyst  -> supervisor
                            |         |-- payment_analyst  -> supervisor
                            |         +-- assess_risk -> [route_after_assessment]
                            |                              |-- review -> human_review -> format_report -> END
                            +------------------------------+-- approve/reject -> format_report -> END
```

The supervisor is the hub. Specialists are spokes. Evidence flows inward.

| Agent | Model | Tools | Role |
|-------|-------|-------|------|
| Supervisor | Sonnet | 4 routing tools | Decides which specialist to consult next |
| Customer Analyst | Haiku | check_customer_history, search_fraud_database | Account history, fraud flags |
| Address Analyst | Haiku | verify_shipping_address | Shipping verification, geo-risk |
| Payment Analyst | Haiku | check_payment_pattern | Payment patterns, anomalies |

If all agents used the same model, you'd have orchestration theatre. The heterogeneous model mix — expensive reasoning for the supervisor, cheap execution for the specialists — is the real payoff.

## The Moving Parts — Progressive Growth

Each phase adds moving parts to the graph. Phases 4 and 7 are omitted because they add infrastructure and hardening inside existing nodes without changing the graph topology.

| # | Moving Part | Phase 0 | Phase 1 | Phase 2 | Phase 3 | Phase 5 | Phase 6 |
|---|-------------|---------|---------|---------|---------|---------|---------|
| 1 | **Nodes** | 0 — no graph | 3 (parse, score, format) | 3 (call_llm, tools, format) | 5 (+parse_order, execute_tools, assess_risk) | 6 (+human_review) | 8 (+supervisor, 3 specialists) |
| 2 | **Edges** | 0 — no graph | Linear: A->B->C | Loop: call_llm<->tools | Same loop | +human_review->format | Star: supervisor hub via `Command()` |
| 3 | **Conditional edges** | 0 — no graph | 1 (node dependency — score threshold) | 1 (LLM-driven — has tool calls?) | 1 (LLM-driven — should_continue) | 2 (+route_after_assessment) | 1 (route_after_assessment) |
| 4 | **Cases** | 6 | 6 | 6 | 6 | 6 | 6 — the constant across all phases |
| 5 | **Tools** | 0 | 0 | 3 (history, address, payment) | 5 (+fraud_db, scoring) | 5 | 5 investigation + 4 routing |
| 6 | **LLM** | 1 Sonnet | 1 Sonnet | 1 Sonnet | 1 Sonnet | 1 Sonnet | 1 Sonnet + 3 Haiku |
| 7 | **Prompts** | 1 (scorer) | 1 (scorer) | 1 (investigator) | 1 (investigator v2) | 1 | 4 (supervisor + 3 specialists) |
| 8 | **Model tiers** | — | — | — | — | — | Sonnet for reasoning, Haiku for execution |
| 9 | **Role-specific prompts** | — | — | — | — | — | Each specialist scoped to its domain |
| 10 | **Supervisor routing** | — | — | — | — | — | LLM picks next specialist via tool calls |
| 11 | **Tool routing** | — | — | — | — | — | `TOOL_TO_TARGET`: tool call -> graph node |

**How to read this table:**

- **Rows 1-7** grow gradually across phases. Each phase adds a concept or two. By Phase 3, the core agent pattern is complete.
- **Rows 8-11** all arrive in Phase 6. These are the multi-agent concepts — they don't exist in single-agent systems. This is the jump in complexity, and why understanding rows 1-7 first matters.
- **Row 4 never changes.** The same 6 fraud cases run through every phase. This is how you know each phase produces correct results — the cases are the constant, the architecture is the variable.

Items 1-7 exist in Phase 3. Items 8-11 are what multi-agent adds. Each new concept is small by itself, but they multiply together. The duplicate evidence bug (covered below) emerged from the interaction between #9 (specialist prompts), #5 (shared tools), and #10 (routing decisions). No single moving part was wrong.

## How the Pieces Work Together

### State: The Shared Blackboard

Every node in the graph reads from and writes to the same state object. This is how information flows between the supervisor and specialists without them calling each other directly.

```python
class FraudStateV6(TypedDict):
    order: dict                                          # The order being investigated
    messages: Annotated[list, add_messages]               # Supervisor's conversation history
    evidence: Annotated[list[Evidence], operator.add]     # Accumulated findings (append-only)
    risk_score: int                                       # Final score (set by assess_risk)
    decision: str                                         # approve / review / reject
    investigation_complete: bool                          # Termination flag
    loop_count: int                                       # Supervisor iteration counter
    tokens_used: int                                      # Cumulative token usage
    guardrail_triggered: str                              # Dead-end or budget exceeded
    human_decision: str                                   # HITL response
    human_notes: str                                      # HITL analyst notes
    specialist_log: Annotated[list[str], operator.add]    # Which specialists were consulted
```

Two fields use `operator.add` reducers — `evidence` and `specialist_log`. This means every node *appends* to these lists rather than overwriting them. The customer analyst adds its evidence, the address analyst adds its evidence, and the final list contains everything. No node needs to know what other nodes contributed.

The `specialist_log` serves a specific purpose: the supervisor reads it to avoid re-consulting the same specialist. After the customer analyst runs, `specialist_log` contains `["customer_analyst"]`. The supervisor sees this and routes to a different specialist next.

### Routing Tools: How the Supervisor Decides

The supervisor doesn't call investigation tools. It calls *routing tools* — lightweight functions that exist purely so the LLM can express a routing decision as a structured tool call.

```python
@tool
def consult_customer_analyst(reason: str) -> str:
    """Route to the customer analyst to check customer history and fraud records."""
    return "Routing to customer analyst"  # This return value is never used
```

The tool function itself does nothing meaningful. What matters is that the LLM chose to call it, and the `reason` parameter captures *why*. The supervisor node intercepts the tool call and maps it to a graph node:

```python
TOOL_TO_TARGET = {
    "consult_customer_analyst": "customer_analyst",
    "consult_address_analyst": "address_analyst",
    "consult_payment_analyst": "payment_analyst",
    "finalize_investigation": "assess_risk",
}
```

Then returns a `Command()` that routes the graph:

```python
return Command(
    goto=target,          # e.g., "customer_analyst"
    update={...},         # state updates (loop_count, tokens)
)
```

Why use tool calls instead of asking the LLM to output JSON like `{"next": "customer_analyst"}`? Because tool calls are a structured format the LLM is trained to produce reliably. No regex parsing, no JSON extraction, no format errors. The tool descriptions also give the LLM rich context about what each specialist does — the docstring `"Route to the customer analyst to check customer history and fraud records"` is how the supervisor knows what the customer analyst is for.

If you've used Windows Workflow Foundation, routing tools are the closest analog to WF states — passive named positions that exist so the engine knows where to transition next. The real work happens in the specialist nodes they route to.

### The Supervisor Loop

The supervisor runs repeatedly until it calls `finalize_investigation`. Each iteration:

1. **Reads the current state** — order details, evidence collected so far, which specialists have already been consulted
2. **Calls the LLM** with the supervisor prompt + accumulated context
3. **The LLM picks a routing tool** — "consult the address analyst because the shipping address is suspicious"
4. **Returns `Command(goto="address_analyst")`** — the graph routes there
5. **The specialist runs, adds evidence, returns to supervisor**
6. **Repeat** — the supervisor now sees the new evidence and decides whether to consult another specialist or finalize

The supervisor prompt steers this behavior:

```
Strategy:
1. Start with the most relevant check for this order
2. After each specialist reports, review findings and decide if more checks needed
3. Conflicting signals require consulting additional specialists
4. When you have enough evidence, call finalize_investigation
```

A guardrail prevents infinite loops: if `loop_count` exceeds `max_loops`, the supervisor is forced to `assess_risk` with whatever evidence exists.

### Specialists: One-Shot Workers

Each specialist follows the same pattern, created by a factory function:

```python
customer_analyst_node = make_specialist(
    name="customer_analyst",
    system_prompt="You investigate customer backgrounds...",
    tools=[check_customer_history, search_fraud_database],
    model=SPECIALIST_MODEL,   # Haiku
)
```

A specialist makes **one LLM call**. With 1-2 tools, the LLM calls everything in a single pass — no multi-turn loop needed. The specialist:

1. Receives the order details from state
2. Calls Haiku with its domain-specific prompt and tools
3. Haiku calls the tools (e.g., `check_customer_history` + `search_fraud_database`)
4. The specialist executes the tool calls, creates `Evidence` records
5. Returns `{"evidence": [...], "specialist_log": ["customer_analyst"], "tokens_used": N}`

The specialist prompt is tightly scoped:

> "You investigate customer backgrounds for fraud investigations. Check customer history and cross-reference the customer's EMAIL with fraud databases. Report your findings factually — do not make final risk decisions."

Two things to notice:

1. **"Report factually — do not make final risk decisions."** Specialists gather evidence. The scoring is deterministic (`_calculate_risk_score_impl`), not LLM-generated. This separation — LLM investigates, Python scores — is core to the system's auditability.

2. **"Use the customer's EMAIL as the indicator. Do NOT search for the shipping address."** This instruction exists because of the duplicate evidence bug. More on that next.

### Non-Overlapping Tool Assignment

Each specialist gets a non-overlapping set of investigation tools:

| Specialist | Tools | Searches for... |
|-----------|-------|----------------|
| Customer Analyst | check_customer_history, search_fraud_database | Customer email |
| Address Analyst | verify_shipping_address | Shipping address |
| Payment Analyst | check_payment_pattern | Payment patterns |

This wasn't the original design. The first version gave `search_fraud_database` to both the customer analyst and the address analyst. That caused the duplicate evidence bug — the most instructive failure in the entire tutorial.

## The Duplicate Evidence Bug

The first working version of Phase 6 produced wrong scores. Not crashes, not errors — *subtly* wrong numbers that only showed up when comparing against the Phase 3 baseline. This is the kind of bug that ships to production if you're not careful.

### What Happened

The initial design gave `search_fraud_database` to both the customer analyst (search by email) and the address analyst (search by address). Same tool, different inputs, different evidence. The reasoning seemed sound.

### The First Run (Case 4 — Robert Chen, conflicting signals)

Phase 3's single agent produced:

```
1. [ok] check_customer_history: low_risk (0.95)    -> -9.5
2. [--] search_fraud_database: neutral (0.70)       ->  0.0  (searched email)
3. [!!] verify_shipping_address: high_risk (0.95)   -> +28.5
4. [! ] check_payment_pattern: medium_risk (0.80)   -> +12.0
5. [--] search_fraud_database: neutral (0.70)       ->  0.0  (searched address)
                                        base: 20 + 31.0 + 10 bonus = 61 -> REVIEW
```

The multi-agent version produced:

```
1. [ok] check_customer_history: low_risk (0.95)     -> -9.5   (customer analyst)
2. [--] search_fraud_database: neutral (0.70)        ->  0.0   (customer analyst -- email)
3. [! ] search_fraud_database: medium_risk (0.64)    -> +9.6   (customer analyst -- address)
4. [!!] verify_shipping_address: high_risk (0.95)    -> +28.5  (address analyst)
5. [--] search_fraud_database: neutral (0.70)        ->  0.0   (address analyst -- email)
6. [! ] search_fraud_database: medium_risk (0.64)    -> +9.6   (address analyst -- address)
7. [! ] check_payment_pattern: medium_risk (0.80)    -> +12.0  (payment analyst)
                                        base: 20 + 50.2 + 10 bonus = 80 -> REJECT
```

Score: 61 -> 80. Decision: REVIEW -> REJECT. HITL didn't trigger.

### The Root Cause

Both specialists searched the fraud database for the Commerce Way address, and both found the same warehouse match (medium_risk, confidence 0.64). The same real-world signal — "this address appears in fraud records" — was counted twice, adding +19.2 points instead of +9.6.

This isn't a bug in any individual component. The tools worked correctly. The specialists worked correctly. The scoring formula worked correctly. The bug is *emergent* — it only exists because of how the agents were composed together.

### Why This Is Hard to Catch

- All 6 decisions were still *plausible*. Case 4 going from REVIEW to REJECT isn't obviously wrong.
- The score (80) was right at the threshold.
- Cases 1, 2, 3, 5, and 6 all produced correct results.
- Without the Phase 3 baseline to compare against, you'd never know the score was inflated.

### The Fix

Remove `search_fraud_database` from the address analyst. Each specialist gets non-overlapping tools so duplicates can't happen. The customer analyst owns all fraud DB lookups. The address analyst focuses purely on address verification.

### The Broader Lesson

Multi-agent systems introduce a class of bugs that single-agent systems don't have: **cross-agent evidence correlation**. When you split investigation across specialists:

1. **Tool overlap creates duplicate evidence.** If two agents can call the same tool, they might produce the same finding. The scoring system counts it twice.
2. **Non-overlapping domains are the cleanest fix.** Deduplication at scoring time is fragile. Non-overlapping tool sets prevent the problem entirely.
3. **Baseline comparison is essential.** Without comparing against Phase 3, this bug would have shipped. The scores were plausible, the decisions were defensible.
4. **Emergent behavior scales with agents.** With 3 specialists, we hit one overlap. With 20 specialists, the overlap surface area grows combinatorially.

## Results

All 6 cases produce identical decisions and scores to the Phase 3 single-agent baseline:

| Case | Decision | Score | HITL? | Specialists | Tokens |
|------|----------|-------|-------|-------------|--------|
| 1: Obviously Legit | APPROVE | 0 | No | customer, address, payment | 7,331 |
| 2: Mildly Suspicious | APPROVE | 33 | No | customer, address, payment | 7,481 |
| 3: High Risk | REJECT | 100 | No | customer, address, payment | 7,775 |
| 4: Conflicting Signals | REVIEW | 61 | Yes | customer, address, payment | 7,933 |
| 5: Historical Fraud | APPROVE | 10 | No | customer, address, payment | 7,457 |
| 6: Tool Error | APPROVE | 6 | No | payment, customer, address | 7,255 |

### Single-Agent vs Multi-Agent

| Metric | Phase 5 (Single-Agent) | Phase 6 (Multi-Agent) |
|--------|----------------------|----------------------|
| Model | All Sonnet | Sonnet supervisor + Haiku specialists |
| Total tokens (6 cases) | 76,139 | 45,232 |
| Estimated cost | ~$0.23 | ~$0.07 |
| Decisions correct | 6/6 | 6/6 |
| Scores matching baseline | 6/6 | 6/6 |

41% fewer tokens. ~67% cheaper. Same decisions, same scores.

## Honest Assessment

With 5 tools, the single-agent is simpler and produces equivalent results. The multi-agent pattern adds coordination overhead (supervisor routing calls) without improving investigation quality at this scale.

The multi-agent pattern earns its keep at scale (200+ tools) where:
- Specialists with 10-15 domain tools each make better tool selection decisions
- Haiku specialists at ~$0.25/MTok vs Sonnet at ~$3/MTok provides real cost savings
- The supervisor only needs "which domain" not "which specific tool"

However, learning the multi-agent pattern in a concrete example with 5 tools makes it easier to understand how and when to apply it in production scenarios that contain the scale and/or complexity to justify it.

## How to Run

```bash
python3 phases/phase6-multi-agent/graph.py
```

Runs 5 demos:
1. **Graph visualization** — Mermaid diagram showing the star topology
2. **Case 4 walkthrough** — Streaming supervisor routing decisions for the hardest case
3. **All 6 cases** — Full run with HITL auto-approval for Case 4
4. **Single-agent vs multi-agent comparison** — Side-by-side token and cost analysis
5. **When multi-agent earns its keep** — The honest framing
