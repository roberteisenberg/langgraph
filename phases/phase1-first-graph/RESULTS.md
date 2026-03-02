# Phase 1 Results — Simple Fraud Scorer

Run date: 2026-03-01

## Summary

Phase 1 has no tools. The LLM scores fraud from surface data alone (order details + metadata passed in the prompt). No database lookups, no address verification, no fraud DB checks.

## Results: graph.py (Linear)

| Case | Score | Decision | Reasoning |
|------|-------|----------|-----------|
| 1: Obviously Legit | 5/100 | APPROVE | Mature account, 50 prior orders, no flags, reasonable amount. |
| 2: Mildly Suspicious | 75/100 | REVIEW | New account (12 days), no history, substantial resellable electronics. |
| 3: High Risk | 85/100 | REJECT | High-value electronics, 2-day account, temp email, industrial address. |
| 4: Conflicting Signals | 5/100 | APPROVE | Saw 2-year account + 30 orders, ignored warehouse/IP/name signals. |
| 5: Historical Fraud | 25/100 | APPROVE | Downplayed "1 minor fraud flag among 8 orders." |
| 6: Tool Error | 15/100 | APPROVE | Normal-looking order, nothing to flag without tools. |

## Results: graph_conditional.py (1B — Conditional Edges)

Identical scores and decisions. The only difference is routing — score determines which node runs (flag_order/review_order/approve_order) instead of setting a string in format_result.

## Analysis

**What worked (Cases 1, 2, 3):**
The LLM handles clear-cut cases well from surface data alone. Case 1 (loyal customer, small order) is correctly low. Case 3 (temp email, high amount, industrial address) is correctly high. Case 2 (new account, electronics) lands in the middle. These cases have enough signal in the order data itself — the LLM's training knowledge about fraud patterns is sufficient.

**What failed (Case 4 — the key failure):**
Score: 5/100. The LLM saw "2-year customer, 30 successful orders, office equipment" and called it safe. It completely missed:
- Warehouse address (couldn't verify — no address tool)
- Same address shared by 2 other accounts (couldn't check — no cross-reference tool)
- Recent IP change (metadata was there but LLM weighted account age more heavily)
- Name mismatch (metadata was there but LLM dismissed it)

The IP change and name mismatch were in the metadata, but the LLM chose to weight the strong customer history over these signals. Without a tool to verify the address or cross-reference it against other accounts, the LLM couldn't discover the most damning evidence. This is exactly the failure that motivates Phase 2.

**What was soft (Case 5):**
Score: 25/100. The prior fraud flag was in the metadata (`prior_fraud_flags: 1`), but the LLM described it as "one minor prior fraud flag" — it invented "minor" on its own. Without a tool to retrieve WHY the customer was flagged, it can't make an informed judgment. In Phase 2+, `check_customer_history` would return the actual flag details.

## Key Takeaway

The LLM's training knowledge handles obvious patterns. Conflicting signals and historical context require tools. Phase 2 adds 3 tools and the loop — Case 4 should improve significantly.

## Observations About 1 vs 1B

Same logic, different structure. 1B teaches the `add_conditional_edges` mechanism but doesn't change any outcomes. The routing function (`route_decision`) uses the same hardcoded thresholds as `format_result` in 1. Both are deterministic after scoring. The mechanism matters in Phase 2 when the LLM — not a threshold — drives routing.
