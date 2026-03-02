# Phase 3: The Investigator — Results

## Summary

Phase 3 implements a full ReAct agent with structured evidence accumulation and deterministic scoring. The LLM investigates selectively, Python scores via `_calculate_risk_score_impl`, and LangGraph manages the loop with a custom `execute_tools` node that replaces the prebuilt `ToolNode`.

## Architecture

```
START → parse_order → call_llm → [should_continue]
                                    ├── has tool calls → execute_tools → call_llm (loop)
                                    └── investigation_complete or no tool calls → assess_risk → format_report → END
```

5 nodes, 5 tools (3 carried from Phase 2, 2 new), deterministic scoring formula.

## Results

| Case | Decision | Score | Evidence Items | Tools Used | LLM Calls |
|---|---|:---:|:---:|:---:|:---:|
| 1: Obviously Legit | APPROVE | 0 | 4 | 5 | 6 |
| 2: Mildly Suspicious | APPROVE | 33 | 5 | 6 | 7 |
| 3: High Risk | REJECT | 100 | 5 | 6 | 7 |
| 4: Conflicting Signals | REVIEW | 61 | 5 | 6 | 7 |
| 5: Historical Fraud | APPROVE | 10 | 5 | 6 | 7 |
| 6: Tool Error | APPROVE | 6 | 5 | 6 | 7 |

## Key Improvements Over Phase 2

1. **Deterministic scoring** — `_calculate_risk_score_impl` uses a weighted formula (base 20, high_risk=+30, medium_risk=+15, low_risk=-10, error=+5) multiplied by confidence, with a +10 bonus for high-confidence high-risk signals. Same evidence always produces the same score.

2. **Structured evidence trail** — Every tool call produces an `Evidence` record with tool name, finding, risk_signal, confidence, raw_data, and timestamp. Evidence accumulates via `operator.add` — never overwritten.

3. **Custom execute_tools node** — Replaces prebuilt `ToolNode`. Intercepts `calculate_risk_score` to run deterministic scoring on real evidence. Creates `Evidence` records from tool outputs via `_create_evidence()` with deterministic signal/confidence mapping.

4. **Tool argument validation** — `execute_tools` overrides LLM-provided arguments for `verify_shipping_address`, `check_payment_pattern`, and `check_customer_history` with actual order data from state. This prevents LLM hallucination of addresses and amounts — a real issue observed during development where the LLM consistently fabricated addresses and dollar amounts instead of using the exact order values.

5. **Clean termination** — `calculate_risk_score` sets `investigation_complete=True`, which `should_continue` checks to route to `assess_risk`. The `assess_risk` node acts as a safety net, running scoring if the LLM exited without calling `calculate_risk_score`.

## Evidence Extraction Rules

| Tool | Signal Source | Confidence Logic |
|---|---|---|
| check_customer_history | risk_level → direct map | 0.95 if >10 orders, 0.80 if >0, 0.70 otherwise |
| verify_shipping_address | geo_risk → direct map | 0.90 residential, 0.95 warehouse, 0.50 unknown |
| check_payment_pattern | anomaly true→medium, false→low, unknown→medium | 0.80 anomaly, 0.85 normal, 0.50 first order |
| search_fraud_database | not found→neutral, found→risk_level map | 0.5 + fraud_rate × 0.4 |
| any error | → "error" | Fixed 0.50 |

## Scoring Formula (Case 4 walkthrough)

```
base_score = 20
+ check_customer_history:  low_risk × 0.95  = -10 × 0.95 = -9.5
+ check_payment_pattern:   medium_risk × 0.80 = 15 × 0.80 = +12.0
+ verify_shipping_address: high_risk × 0.95  = 30 × 0.95 = +28.5
+ search_fraud_database:   neutral × 0.70    =  0 × 0.70 =   0.0
+ search_fraud_database:   neutral × 0.70    =  0 × 0.70 =   0.0
= 51.0
+ high_confidence high_risk bonus: +10
= 61 → REVIEW
```

## Comparison Across Phases

| Case | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| 1: Obviously Legit | APPROVE (5) | APPROVE (15) | APPROVE (0) |
| 2: Mildly Suspicious | REVIEW (75) | REVIEW (45) | APPROVE (33) |
| 3: High Risk | REJECT (85) | REJECT (95) | REJECT (100) |
| 4: Conflicting Signals | APPROVE (5) | REVIEW (75) | REVIEW (61) |
| 5: Historical Fraud | APPROVE (25) | REVIEW (35) | APPROVE (10) |
| 6: Tool Error | APPROVE (15) | APPROVE (25) | APPROVE (6) |

Phase 3 produces the most nuanced results: Case 4 correctly lands in REVIEW (conflicting signals), Case 5 correctly approves (historical flag was false positive), and Case 3 maxes out at 100 (multiple high-risk indicators compounding).

## Observations

- **Tool selectivity is limited in practice.** The LLM tends to call all available tools rather than picking only what's relevant. It errs on the side of thoroughness, which isn't wrong — just not the selective behavior the prompt aims for.
- **LLM hallucination of tool arguments required a guardrail.** During development, the model consistently fabricated addresses and amounts instead of using the exact values from the order (e.g., inventing "1234 Freight Way, Miami" when the order address was "Bay 12, 5500 Commerce Way, Reno"). This happened even with explicit prompt instructions to use exact values. The fix: `execute_tools` overrides args for `verify_shipping_address`, `check_payment_pattern`, and `check_customer_history` with actual order data from state. This is a structural guardrail, not a prompt-level one — the LLM decides *which* tools to call, but the system ensures the inputs are correct.
- **`search_fraud_database` args are not validated** — the LLM chooses what indicator to cross-reference, which is intentional (it might search an email domain, a full address, etc.). The tradeoff is that some matches may be missed if the LLM searches for the wrong thing.
- **Full auditability.** Every score traces back to specific tool outputs and deterministic mapping rules. No LLM interpretation in the scoring path.
