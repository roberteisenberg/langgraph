# Phase 6: Multi-Agent Results

## Architecture

Supervisor (Sonnet) + 3 Specialists (Haiku) with Command() routing.

```
START -> parse_order -> supervisor -> [Command routing]
                            ^         |-- customer_analyst -> supervisor
                            |         |-- address_analyst  -> supervisor
                            |         |-- payment_analyst  -> supervisor
                            |         +-- assess_risk -> [route_after_assessment]
                            |                              |-- review -> human_review -> format_report -> END
                            +------------------------------+-- approve/reject -> format_report -> END
```

| Agent | Model | Tools |
|-------|-------|-------|
| Supervisor | claude-sonnet-4 | Routing tools (consult_*, finalize) |
| Customer Analyst | claude-haiku-4.5 | check_customer_history, search_fraud_database |
| Address Analyst | claude-haiku-4.5 | verify_shipping_address |
| Payment Analyst | claude-haiku-4.5 | check_payment_pattern |

## Results — All 6 Cases

| Case | Decision | Score | HITL? | Specialists | Tokens |
|------|----------|-------|-------|-------------|--------|
| 1: Obviously Legit | APPROVE | 0 | No | customer, address, payment | 7,331 |
| 2: Mildly Suspicious | APPROVE | 33 | No | customer, address, payment | 7,481 |
| 3: High Risk | REJECT | 100 | No | customer, address, payment | 7,775 |
| 4: Conflicting Signals | REVIEW | 61 | Yes | customer, address, payment | 7,933 |
| 5: Historical Fraud | APPROVE | 10 | No | customer, address, payment | 7,457 |
| 6: Tool Error | APPROVE | 6 | No | payment, customer, address | 7,255 |

## Single-Agent vs Multi-Agent Comparison

| Case | Phase 5 Decision | Phase 5 Score | Phase 5 Tokens | Phase 6 Decision | Phase 6 Score | Phase 6 Tokens |
|------|------------------|---------------|----------------|------------------|---------------|----------------|
| 1: Obviously Legit | APPROVE | 0 | 11,075 | APPROVE | 0 | 7,331 |
| 2: Mildly Suspicious | APPROVE | 33 | 13,468 | APPROVE | 33 | 7,481 |
| 3: High Risk | REJECT | 100 | 13,751 | REJECT | 100 | 7,775 |
| 4: Conflicting Signals | REVIEW | 61 | 13,669 | REVIEW | 61 | 7,933 |
| 5: Historical Fraud | APPROVE | 10 | 13,416 | APPROVE | 10 | 7,457 |
| 6: Tool Error | APPROVE | 6 | 10,760 | APPROVE | 6 | 7,255 |
| **TOTAL** | | | **76,139** | | | **45,232** |

### Cost Estimate (6 cases)

- Single-agent (all Sonnet): ~$0.46
- Multi-agent (Sonnet supervisor + Haiku specialists): ~$0.15

## Phase 3 Comparison

All 6 cases produce identical decisions and scores to Phase 3:

| Case | Phase 3 | Phase 6 |
|------|---------|---------|
| 1: Obviously Legit | APPROVE (0) | APPROVE (0) |
| 2: Mildly Suspicious | APPROVE (33) | APPROVE (33) |
| 3: High Risk | REJECT (100) | REJECT (100) |
| 4: Conflicting Signals | REVIEW (61) | REVIEW (61) |
| 5: Historical Fraud | APPROVE (10) | APPROVE (10) |
| 6: Tool Error | APPROVE (6) | APPROVE (6) |

## Key Observations

1. **Decisions and scores match** Phase 3 and Phase 5 exactly across all 6 cases.

2. **Token usage lower** in multi-agent (45K vs 76K total). Haiku specialists use fewer tokens than Sonnet for the same tool calls.

3. **Case 4 HITL works.** Score 61 falls in the review range (50-79), triggers interrupt, auto-approved in demo.

4. **Case 6 tool error handled.** Payment analyst encounters the error, supervisor routes to remaining specialists, investigation completes normally.

## The Duplicate Evidence Bug — A Multi-Agent Lesson

The first working version of Phase 6 produced wrong scores. Not crashes, not errors — *subtly* wrong numbers that only showed up when comparing against the Phase 3 baseline. This is the kind of bug that ships to production if you're not careful, and it's worth walking through because it illustrates a fundamental challenge of multi-agent systems.

### What happened

The initial design gave `search_fraud_database` to both the customer analyst and the address analyst. The reasoning seemed sound: the customer analyst would search by email, the address analyst would search by address. Same tool, different inputs, different evidence. The plan even called this out explicitly as a feature.

### The first run (Case 4 — Robert Chen, conflicting signals)

The single-agent (Phase 3/5) produced this evidence for Case 4:

```
1. [ok] check_customer_history: low_risk (0.95)    → -9.5
2. [--] search_fraud_database: neutral (0.70)       →  0.0  (searched email)
3. [!!] verify_shipping_address: high_risk (0.95)   → +28.5
4. [! ] check_payment_pattern: medium_risk (0.80)   → +12.0
5. [--] search_fraud_database: neutral (0.70)       →  0.0  (searched address)
                                        base: 20 + 31.0 + 10 bonus = 61 → REVIEW
```

The multi-agent produced this:

```
1. [ok] check_customer_history: low_risk (0.95)     → -9.5   (customer analyst)
2. [--] search_fraud_database: neutral (0.70)        →  0.0   (customer analyst — email)
3. [! ] search_fraud_database: medium_risk (0.64)    → +9.6   (customer analyst — address)
4. [!!] verify_shipping_address: high_risk (0.95)    → +28.5  (address analyst)
5. [--] search_fraud_database: neutral (0.70)        →  0.0   (address analyst — email)
6. [! ] search_fraud_database: medium_risk (0.64)    → +9.6   (address analyst — address)
7. [! ] check_payment_pattern: medium_risk (0.80)    → +12.0  (payment analyst)
                                        base: 20 + 50.2 + 10 bonus = 80 → REJECT
```

Score jumped from 61 to 80. The decision flipped from REVIEW to REJECT. HITL didn't trigger.

### The root cause

Both specialists searched the fraud database for the Commerce Way address, and both found the same warehouse match (medium_risk, confidence 0.64). The same real-world signal — "this address appears in fraud records" — was counted twice, adding +19.2 points to the score instead of +9.6.

This isn't a bug in any individual component. The tools worked correctly. The specialists worked correctly. The scoring formula worked correctly. The bug is *emergent* — it only exists because of how the agents were composed together. Each specialist did exactly what it was told to do. The problem is that nobody was deduplicating overlapping investigations across specialist boundaries.

### Why this is hard to catch

- All 6 decisions were still *plausible*. Case 4 going from REVIEW to REJECT isn't obviously wrong — it's a genuinely ambiguous case.
- The score (80) was right at the threshold, so it could easily be attributed to "multi-agent collects slightly different evidence."
- Cases 1, 2, 3, 5, and 6 all produced correct results. The bug only manifested on Case 4 because it's the only case where the address has a fraud DB match AND the customer has a clean fraud DB record — creating the specific overlap pattern.
- Without the Phase 3 baseline to compare against, you'd never know the score was inflated.

### The fix

Remove `search_fraud_database` from the address analyst. Each specialist gets non-overlapping tools:

| Agent | Tools | Searches fraud DB for... |
|-------|-------|--------------------------|
| Customer Analyst | check_customer_history, search_fraud_database | Email |
| Address Analyst | verify_shipping_address | (nothing — just verifies) |
| Payment Analyst | check_payment_pattern | (nothing — just checks patterns) |

The customer analyst owns all fraud DB lookups. The address analyst focuses purely on address verification. No overlap, no double-counting.

### The broader lesson

Multi-agent systems introduce a class of bugs that single-agent systems don't have: **cross-agent evidence correlation**. When you split investigation across specialists, you need to think about:

1. **Tool overlap** — If two agents can call the same tool, they might produce duplicate evidence. The scoring system doesn't know (and shouldn't need to know) which agent produced each piece of evidence.

2. **Non-overlapping domains** — The cleanest fix isn't deduplication at scoring time (fragile, requires understanding all possible overlaps). It's giving each specialist a non-overlapping tool set so duplicates can't happen.

3. **Baseline comparison is essential** — Without comparing multi-agent results against the single-agent baseline, this bug would have shipped. The scores were plausible, the decisions were defensible, and the system ran without errors. The only signal was a 19-point score difference that required understanding the scoring formula to explain.

4. **Emergent behavior scales with agents** — With 3 specialists, we hit one overlap. With 20 specialists, the overlap surface area grows combinatorially. This is the real complexity cost of multi-agent architectures.

## Honest Assessment

With 5 tools, the single-agent (Phase 5) is simpler and produces equivalent results. The multi-agent pattern adds coordination overhead (supervisor routing calls) without improving investigation quality at this scale.

The multi-agent pattern earns its keep at scale (200+ tools) where:
- Specialists with 10-15 domain tools each make better tool selection decisions
- Haiku specialists at ~$0.80/MTok vs Sonnet at ~$3/MTok provides real cost savings
- The supervisor only needs "which domain" not "which specific tool"

If all agents used the same model, you'd have orchestration theatre.
