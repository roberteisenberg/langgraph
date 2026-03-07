# Phase 2: The Loop — Tools + First Agent

## What This Phase Does

Phase 1 scored fraud from a single LLM call — no tools, no investigation. Case 4 scored 5/100 because the LLM couldn't verify the warehouse address or cross-reference the payment anomaly. It guessed from surface data and got it wrong.

Phase 2 adds three tools and the loop. The LLM can now call tools, observe results, and call more tools before producing a final assessment. This is where LangGraph starts earning its keep — **the loop**.

### What's a Tool?

A tool is a Python function that the LLM can choose to call. You decorate a function with `@tool`, give it a docstring, and bind it to the LLM. The LLM sees the function name, description, and parameters — it doesn't know or care what happens inside. The function could query a database, call an external API, hit an MCP server, or run local logic. In this tutorial, all tools are local functions returning simulated data. In production, you'd swap the internals for real service calls — the LLM-facing interface stays the same.

## Architecture

```
START -> call_llm -> [tools_condition]
                      |-- has tool calls -> tools -> call_llm (loop back)
                      +-- no tool calls  -> format_result -> END
```

Three tools, all simulated:

| Tool | Input | Returns |
|------|-------|---------|
| `check_customer_history` | email | Account age, prior orders, fraud flags |
| `verify_shipping_address` | address | Type (residential/warehouse), geo risk |
| `check_payment_pattern` | email, amount | Typical range, anomaly flag, velocity |

## What's New Since Phase 1

- **Tools** — the LLM can now look things up instead of guessing
- **`add_messages` reducer** — conversation history accumulates across loop iterations (without this, each call overwrites the previous one)
- **The loop** — the LLM decides when to stop calling tools. This is the first cycle in the graph — you can't draw this as a straight line:

```
                    ┌────────── loop ──────────┐
                    │                          │
                    ▼                          │
  START ──> [ call_llm ] ───> [ tools ] ──────┘
                │              execute tool calls,
                │              add results to messages
                │
                └── no tool calls ──> [ format_result ] ──> END
                    (done investigating)

  Each iteration, messages grow:
    [prompt] -> [LLM: call tool A] -> [tool A result]
            -> [LLM: call tool B] -> [tool B result]
            -> [LLM: final answer]
```

The conditional edge after `call_llm` is the decision point. If the LLM's response contains tool calls, the graph routes to `tools` and loops back. If not, the LLM is done investigating and the graph moves to `format_result`.

The mechanism is the same as Phase 1: a dependency between two functions, serialized into the graph as a conditional edge, mediated by state. In Phase 1, `score_fraud` wrote a number and `route_decision` read it — deterministic. Here, `call_llm` writes a response and `tools_condition` reads it — but now the LLM's output drives the routing. Same pattern, different driver.

This is a foundational change. Loops and branching have always been deterministic — a `while` loop checks a condition a developer wrote, an `if` branch follows rules a developer defined. Here, the LLM controls the flow. It reasons about what to check, acts, observes, and reasons again. That's powerful, but it's also dangerous. You lose predictability. The LLM could loop indefinitely, burn through tokens, or chase irrelevant leads. This is why Phase 4 adds guardrails, token budgets, and streaming visibility — the infrastructure you need once the LLM controls the flow.

- **`ToolNode` and `tools_condition`** — prebuilt components from LangGraph that handle tool execution and routing. Phase 3 replaces these with custom implementations for more control.

### The Fundamental Shift

Phase 1's conditional edge routed on a **data value** — `if score >= 80: reject`. A developer wrote that rule.

Phase 2's `tools_condition` routes on the **LLM's decision** — "does the response contain tool calls?" The LLM decides whether to investigate more or produce a final answer. No developer wrote the rule for when to check the shipping address. The LLM figured that out from the order data and tool descriptions.

This is the shift from traditional state machines to LLM-driven workflows. You define the structure (which nodes exist, what edges are possible). The LLM chooses the path through that structure each time.

## The Moving Parts — Progressive Growth

| # | Moving Part | Phase 0 | Phase 1 | Phase 2 |
|---|-------------|---------|---------|---------|
| 1 | **Nodes** | 0 — no graph | 3 (parse, score, format) | 3 (call_llm, tools, format_result) |
| 2 | **Edges** | 0 — no graph | Linear: A->B->C | **Loop: call_llm<->tools** |
| 3 | **Conditional edges** | 0 — no graph | 1 (node dependency — score threshold) | 1 (LLM-driven — has tool calls?) |
| 4 | **Cases** | 6 | 6 | 6 — same cases, better results |
| 5 | **Tools** | 0 | 0 | **3 (history, address, payment)** |
| 6 | **LLM** | 1 (Sonnet) | 1 (Sonnet) | 1 (Sonnet) |
| 7 | **Prompts** | 1 (scorer) | 1 (scorer) | 1 (investigate this order) |

**What changed:** The loop (row 2) and tools (row 5) are the additions. Everything else stays the same count-wise, but the nature changes — the prompt shifts from "score this" to "investigate this," and the conditional edge routes on LLM behavior instead of a score threshold.

## Results

| Case | Phase 1 Score | Phase 2 Score | Decision | Tools Called |
|------|:---:|:---:|:---:|---|
| 1: Obviously Legit | 5 | 15 | APPROVE | all 3 |
| 2: Mildly Suspicious | 75 | 45 | REVIEW | all 3 |
| 3: High Risk | 85 | 95 | REJECT | all 3 |
| 4: Conflicting Signals | **5** | **75** | **REVIEW** | all 3 |
| 5: Historical Fraud | 25 | 35 | REVIEW | all 3 |
| 6: Tool Error | 15 | 25 | APPROVE | all 3 |

## The Key Win: Case 4

Phase 1: 5/100 (approve). The LLM saw a loyal customer and stopped thinking.

Phase 2: 75/100 (review). The tools surfaced what the LLM couldn't see:
- `verify_shipping_address` → warehouse, shared by 3 accounts
- `check_payment_pattern` → amount 1.9x above typical maximum

The LLM synthesized these tool results with the customer history and produced a score that reflects the conflicting signals. This validates the core thesis: **tools + loops let the LLM investigate rather than guess.**

## What's Still Missing

The LLM called all 3 tools for every case — no selective investigation. It also scores risk itself (from its own reasoning), which isn't auditable. Phase 3 fixes both:
- A system prompt that guides selective tool usage
- `search_fraud_database` and `calculate_risk_score` tools (the LLM investigates, Python scores)
- Structured `Evidence` accumulation instead of just message history

## How to Run

```bash
python3 phases/phase2-tools/graph.py
```
