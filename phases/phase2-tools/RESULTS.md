# Phase 2: The Loop — Results

## Architecture

```
START → call_llm → [tools_condition]
                     ├── has tool calls → tools → call_llm (loop back)
                     └── no tool calls → format_result → END
```

Three simulated tools:
- `check_customer_history` — account age, prior orders, fraud flags
- `verify_shipping_address` — residential vs warehouse, geo risk
- `check_payment_pattern` — spending anomalies, velocity

## Results

| Case | Phase 1 Score | Phase 2 Score | Phase 2 Decision | Tools Called |
|------|:---:|:---:|:---:|---|
| 1: Obviously Legit | 5 | 15 | APPROVE | customer_history, shipping, payment |
| 2: Mildly Suspicious | 75 | 45 | REVIEW | customer_history, shipping, payment |
| 3: High Risk | 85 | 95 | REJECT | customer_history, shipping, payment |
| 4: Conflicting Signals | **5** | **75** | **REVIEW** | customer_history, shipping, payment |
| 5: Historical Fraud | 25 | 35 | REVIEW | customer_history, shipping, payment |
| 6: Tool Error | 15 | 25 | APPROVE | customer_history, shipping, payment |

## Key Observations

### Case 4 — The Hinge Case

Phase 1 scored Case 4 at **5/100** (approve) because the LLM's training weighted "2-year account, 30 prior orders" as overwhelmingly safe. Without tools, it had no way to verify that:

- The shipping address is a **warehouse** shared by **3 accounts** this week
- The order amount ($1,500) is **1.9x above** the customer's typical maximum ($800)

Phase 2 scored it at **75/100** (review). The tools surfaced these signals, and the LLM correctly synthesized them into a higher risk assessment. This validates the core thesis: **tools + loops let the LLM investigate rather than guess**.

### Case 3 — Even Higher with Evidence

Phase 1 already caught Case 3 at 85/100 (temp email + industrial address were in the raw data). Phase 2 pushed it to **95/100** because the tools confirmed the address is a "known freight forwarding facility" and the email is from a "disposable email provider" — stronger signal than metadata alone.

### Case 6 — Graceful Error Handling

The payment tool returns an error for Case 6 (`{"error": "Payment processing service unavailable"}`). The LLM handled this without crashing, noting the unavailable service in its reasoning and scoring based on the other two tools.

### Tool Usage Pattern

The LLM called all 3 tools for every case. In Phase 3, the system prompt will guide more selective tool usage — not every order needs every tool checked.

## What This Proves

1. **The loop works** — LLM calls tools, gets results, then produces final assessment
2. **Tools improve accuracy** — Case 4 went from 5 → 75 (the whole point of Phase 2)
3. **Error resilience** — Case 6 handled a tool error without crashing
4. **LangGraph earns its keep** — a linear pipeline can't do this; you need the conditional loop

## What's Missing (Phase 3)

- Structured `Evidence` accumulation (currently just messages)
- `calculate_risk_score` tool (LLM scores directly — less auditable)
- Selective tool usage (LLM calls all tools every time)
- Sophisticated system prompt for conflict handling
- `search_fraud_database` tool
