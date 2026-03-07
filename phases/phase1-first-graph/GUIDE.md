# Phase 1: The Setup — You Don't Need LangGraph Yet

## What This Phase Does

Phase 0 called the LLM from a plain Python function. Phase 1 wraps that same logic in a LangGraph graph. Same prompt, same scores, same decisions. The graph doesn't improve results — it sets up the scaffolding that Phase 2 needs when it adds tools and the loop.

### What's a LangGraph Graph?

A LangGraph graph is a **state machine for LLM workflows**. You define:

- **State** — a shared data object (a `TypedDict`) that flows through the graph. Every node reads from it and writes back to it.
- **Nodes** — Python functions. Each receives the current state, does work (call an LLM, run a calculation, format output), and returns state updates.
- **Edges** — transitions between nodes. Fixed edges (A always goes to B) or conditional edges (go to B or C depending on state values).

You build the graph, compile it, and invoke it with an initial state. LangGraph handles the execution order.

```
                     { state: order, risk_score, decision }
                                    |
                                    v
  START --> [ parse_order ] --> [ score_fraud ] --> [ format_result ] --> END
               node      edge      node      edge      node
```

Three nodes (functions), connected by edges (transitions), sharing one state object. Each node reads what it needs from state and writes back its updates.

### Why Set Up the Graph Now?

Phase 0 proved that a plain function works for a single LLM call. But Phase 2 adds tools and a loop — the LLM calls a tool, observes the result, decides whether to call another tool or stop. That loop requires a graph: nodes for the LLM call and tool execution, a conditional edge for the routing decision, and state to accumulate the conversation across iterations.

The setup cost is paid here, once. Phase 1 teaches StateGraph, state schemas, nodes, edges, and conditional edges — all with the same single-LLM-call logic from Phase 0 — so that Phase 2 can focus on what's new: tools and the loop.

## Architecture

Two variants, same outcome:

**graph.py — Linear:**
```
START -> parse_order -> score_fraud -> format_result -> END
```

**graph_conditional.py — Conditional edges:**
```
START -> parse_order -> score_fraud -> [route_decision]
                                        |-- score >= 80 -> flag_order -> END
                                        |-- score >= 50 -> review_order -> END
                                        +-- score < 50  -> approve_order -> END
```

Same scores, same decisions. The conditional variant teaches `add_conditional_edges` — the mechanism Phase 2 uses when the LLM drives routing instead of a threshold.

In the conditional variant, `score_fraud` and `route_decision` are two functions that don't call each other and don't know about each other. `score_fraud` writes `risk_score: 85` to state. `route_decision` reads `risk_score` from state and returns `"flag_order"`. State is the intermediary. This is the LangGraph pattern at every phase — nodes communicate through shared state, not through function calls.

## What It Teaches

- **StateGraph** — define a graph, add nodes, add edges, compile, invoke
- **State schema** — `TypedDict` with `order`, `risk_score`, `decision`
- **Nodes as functions** — receive state, do work, return state updates
- **Edges** — fixed transitions (A -> B -> C)
- **Conditional edges** — routing based on state values (score thresholds)

## The Moving Parts

| # | Moving Part | Phase 0 | Phase 1 |
|---|-------------|---------|---------|
| 1 | **Nodes** | 0 — no graph | 3 linear (parse, score, format) / 5 conditional (+flag, review, approve) |
| 2 | **Edges** | 0 — no graph | Linear: A->B->C |
| 3 | **Conditional edges** | 0 — no graph | 0 linear / 1 conditional (node dependency — score threshold) |
| 4 | **Cases** | 6 | 6 |
| 5 | **Tools** | 0 | 0 |
| 6 | **LLM** | 1 (Sonnet) | 1 (Sonnet) |
| 7 | **Prompts** | 1 (scorer) | 1 (same scorer, now in a node) |

## Results

| Case | Score | Decision | What Happened |
|------|-------|----------|---------------|
| 1: Obviously Legit | 5 | APPROVE | Correct — mature account, small order |
| 2: Mildly Suspicious | 75 | REVIEW | Correct — new account, electronics |
| 3: High Risk | 85 | REJECT | Correct — temp email, industrial address, high amount |
| 4: Conflicting Signals | 5 | APPROVE | **WRONG** — missed warehouse, IP change, name mismatch |
| 5: Historical Fraud | 25 | APPROVE | Soft — invented "minor" to describe the fraud flag |
| 6: Tool Error | 15 | APPROVE | Correct — nothing to flag without tools |

## The Key Failure: Case 4

Score: 5/100. The LLM saw "2-year customer, 30 successful orders, office equipment" and called it safe. It completely missed:

- **Warehouse address** — couldn't verify (no address tool)
- **Address shared by 3 accounts** — couldn't cross-reference (no fraud DB tool)
- **Recent IP change** — metadata was there, but the LLM weighted account history more
- **Shipping name mismatch** — metadata was there, but the LLM dismissed it

Without tools, the LLM can't discover what it can't see. The order metadata contains hints (IP change, name mismatch), but the LLM has no way to investigate them. It defaults to the strongest surface signal — a loyal customer — and misses everything else.

This failure motivates Phase 2. The LLM needs tools.

## How to Run

```bash
# Linear variant
python3 phases/phase1-first-graph/graph.py

# Conditional variant
python3 phases/phase1-first-graph/graph_conditional.py
```
