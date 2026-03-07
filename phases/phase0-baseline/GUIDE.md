# Phase 0: The Baseline — A Python Function Calls an LLM

## What This Phase Does

Phase 0 is the starting point. A Python function takes an order, builds a prompt, calls the LLM, parses the response, and returns a fraud score. No framework. No graph. No tools. One function, one LLM call.

This is all you need for a single LLM call. Everything that follows in Phases 1-6 builds on this foundation.

This tutorial walks you through detecting fraud starting from this simple function-to-LLM call, progressively adding LangGraph scaffolding, tools, investigation loops, infrastructure, human-in-the-loop workflows, and multi-agent coordination — all using the same 6 fraud cases so you can see exactly what each addition contributes. By the end, you'll be able to determine whether LangGraph — and which parts of LangGraph — are necessary for your own scenarios.

## Architecture

```
score_fraud(order) -> prompt -> LLM -> parse JSON -> {score, decision, reasoning}
```

No graph. The function builds a prompt from the order data, passes it as a `HumanMessage` to `ChatAnthropic.invoke()` — a single API call to Claude — and parses the JSON response. The prompt includes the order details (customer info, amount, items, shipping address, account metadata) and asks for a 0-100 score with one sentence of reasoning.

Decision thresholds are Python `if` statements:

```python
if score >= 80: decision = "reject"
elif score >= 50: decision = "review"
else: decision = "approve"
```

## The Moving Parts

| # | Moving Part | Phase 0 |
|---|-------------|---------|
| 1 | **Nodes** | 0 — no graph |
| 2 | **Edges** | 0 — no graph |
| 3 | **Conditional edges** | 0 — no graph |
| 4 | **Cases** | 6 canonical fraud scenarios |
| 5 | **Tools** | 0 — the LLM sees only what's in the prompt |
| 6 | **LLM** | 1 (Sonnet) |
| 7 | **Prompts** | 1 (score this order 0-100) |

Rows 1-3 are empty. There is no graph — just a function.

## Results

| Case | Score | Decision | What Happened |
|------|-------|----------|---------------|
| 1: Obviously Legit | 5 | APPROVE | Correct — mature account, small order |
| 2: Mildly Suspicious | 75 | REVIEW | Correct — new account, electronics |
| 3: High Risk | 85 | REJECT | Correct — temp email, industrial address, high amount |
| 4: Conflicting Signals | 5 | APPROVE | **WRONG** — missed warehouse, IP change, name mismatch |
| 5: Historical Fraud | 25 | APPROVE | Soft — invented "minor" to describe the fraud flag |
| 6: Tool Error | 15 | APPROVE | Correct — nothing to flag without tools |

These are the same scores Phase 1 produces. The LLM, the prompt, and the parsing are identical — Phase 1 wraps them in a StateGraph, but the outcome doesn't change.

## The Key Failure: Case 4

Score: 5/100. The LLM saw "2-year customer, 30 successful orders, office equipment" and called it safe. It completely missed:

- **Warehouse address** — couldn't verify (no address tool)
- **Address shared by 3 accounts** — couldn't cross-reference (no fraud DB tool)
- **Recent IP change** — metadata was there, but the LLM weighted account history more
- **Shipping name mismatch** — metadata was there, but the LLM dismissed it

Without tools, the LLM can't discover what it can't see. The order metadata contains hints (IP change, name mismatch), but the LLM has no way to investigate them. It defaults to the strongest surface signal — a loyal customer — and misses everything else.

## Why Phase 1

Phase 0 works. For a single LLM call, this is all you need. Phase 1 wraps this same logic in LangGraph's StateGraph — not because it improves results, but because Phase 2 adds the loop, and the loop requires the graph scaffolding. The setup cost is paid once.

## How to Run

```bash
python3 phases/phase0-baseline/score_fraud.py
```
