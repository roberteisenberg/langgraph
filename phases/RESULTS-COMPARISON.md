# Phase Comparison — Accuracy Trend

## Ground Truth (from test case design)

| Case | Correct Decision | Ideal Score Range | Why |
|---|---|---|---|
| 1: Obviously Legit | approve | 0–15 | 5-year customer, 50 orders, small purchase |
| 2: Mildly Suspicious | approve | 30–45 | New account but normal address, no fraud hits |
| 3: High Risk | reject | 85–100 | Disposable email, freight forwarder, high-value |
| 4: Conflicting Signals | review | 55–70 | Trusted customer, but warehouse + spending anomaly |
| 5: Historical Fraud | approve | 5–20 | Prior flag was resolved as false positive |
| 6: Tool Error | approve | 5–20 | Low-risk customer, one tool unavailable |

## Scores Side by Side

| Case | Ideal | Phase 1 | Phase 2 | Phase 3 |
|---|:---:|:---:|:---:|:---:|
| 1: Obviously Legit | 0–15 | 5 ✓ | 15 ✓ | 0 ✓ |
| 2: Mildly Suspicious | 30–45 | **75** ✗ | 45 ✓ | 33 ✓ |
| 3: High Risk | 85–100 | 85 ✓ | 95 ✓ | 100 ✓ |
| 4: Conflicting Signals | 55–70 | **5** ✗ | **75** ~ | 61 ✓ |
| 5: Historical Fraud | 5–20 | 25 ~ | **35** ✗ | 10 ✓ |
| 6: Tool Error | 5–20 | 15 ✓ | **25** ~ | 6 ✓ |

## Decision Accuracy

| | Phase 1 | Phase 2 | Phase 3 |
|---|:---:|:---:|:---:|
| Correct decisions | 4/6 | 4/6 | **6/6** |
| Wrong | Case 2 (REVIEW), Case 4 (APPROVE) | Case 4 (borderline high), Case 5 (REVIEW) | — |

Phase 2 fixed Case 4 but *broke* Case 5. Adding tools gave the LLM more data, but it couldn't distinguish a resolved false positive from an active flag — it just saw "1 fraud flag" and bumped the score. More information without better interpretation made one case worse.

## Score Accuracy (in range + correct decision)

| Case | Ideal Range | Phase 1 | Phase 2 | Phase 3 |
|---|:---:|:---:|:---:|:---:|
| 1: Obviously Legit | 0–15 | 5 ✓ in range | 15 ✓ in range | 0 ✓ in range |
| 2: Mildly Suspicious | 30–45 | 75 ✗ 30 over | 45 ✓ in range | 33 ✓ in range |
| 3: High Risk | 85–100 | 85 ✓ in range | 95 ✓ in range | 100 ✓ in range |
| 4: Conflicting Signals | 55–70 | 5 ✗ 50 under | 75 ✗ 5 over | 61 ✓ in range |
| 5: Historical Fraud | 5–20 | 25 ✗ 5 over | 35 ✗ 15 over | 10 ✓ in range |
| 6: Tool Error | 5–20 | 15 ✓ in range | 25 ✗ 5 over | 6 ✓ in range |
| **Scores in range** | | **3/6** | **3/6** | **6/6** |
| **Correct decisions** | | **4/6** | **4/6** | **6/6** |

## The Trend

```
Scores in ideal range:

Phase 1:  ███░░░  3/6
Phase 2:  ███░░░  3/6
Phase 3:  ██████  6/6
```

Phase 1 and Phase 2 each get 3 scores in range — but they miss *different* cases:

- **Phase 1** nails Cases 1, 3, 6 from surface data alone. It fails on Case 2 (overreacts to a new account), Case 4 (can't see the warehouse address without tools), and Case 5 (slightly over).

- **Phase 2** fixes Case 2 (tools confirm the address is fine) but breaks Case 5 (the LLM sees "1 fraud flag" in tool output and bumps the score — it can't tell the flag was resolved) and Case 6 (tool error adds uncertainty the LLM inflates). Same 3/6 hit rate, different misses.

- **Phase 3** hits 6/6 because scoring moved to deterministic Python. `_create_evidence` maps tool outputs to signals with fixed rules, and `_calculate_risk_score_impl` does the math. The LLM only decides which tools to call.

The key insight: Phases 1 → 2 added more information but the same 3/6 score accuracy. More data didn't help because the LLM was still interpreting and scoring in prose. Phase 3 improved accuracy by narrowing what the LLM is responsible for — not by giving it more to work with.

## Architecture Progression

| | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| Nodes | 3 | 3 | 5 |
| Tools | 0 | 3 | 5 |
| Loop | No | Yes (prebuilt) | Yes (custom) |
| Scoring | LLM in prose | LLM in JSON | Deterministic Python |
| Evidence trail | None | Messages only | Structured `Evidence` records |
| Termination | Linear | No tool calls | `investigation_complete` flag |
| Tool execution | N/A | Prebuilt `ToolNode` | Custom with arg validation |
