# LangGraph Deep Dive — Internal Planning

Private working notes. Starting point — expect structure and conclusions to change as we build and learn.

---

## Key Tensions (Read These First)

These are the questions we don't have answers to yet. They'll shape decisions throughout the build. If something feels wrong during implementation, check whether one of these tensions is the root cause.

1. **Framework vs. simplicity.** LangGraph adds real complexity. For Phases 1-2, a plain function call is objectively simpler. The bet is that the graph structure pays off starting in Phase 3. If it doesn't, we need to be honest about that.

2. **Prebuilt vs. hand-built.** `create_react_agent` gets you 80% of the way fast. But the research says "prebuilt components are easy to replicate." Do we outgrow them, or are they good enough? Building both ways in Phase 3 will answer this. One concrete factor: hand-built nodes are far easier to step through in LangSmith traces than `create_react_agent` abstractions. If debugging matters (and it will), that's a real argument for hand-building.

3. **Single agent vs. multi-agent.** Phase 6 assumes multi-agent is worth the overhead. The research warns it often isn't. We'll know after Phase 3 whether the fraud domain genuinely benefits from specialist agents or if one well-prompted agent with more tools does better.

4. **LangChain dependency.** Core LangGraph has no LangChain dependency. Prebuilt components pull it in. How much of LangChain do we actually need, and can we cleanly remove it? Track this from Phase 1. Practical reality: even if you avoid LCEL (LangChain Expression Language), you'll almost certainly depend on `langchain-core` for message types and formatting (`HumanMessage`, `AIMessage`, `ToolMessage`). That's likely the irreducible minimum.

5. **Managed vs. self-hosted deployment.** LangGraph Cloud is the easy path but has pricing opacity and lock-in implications (especially for the future composition guide with Dapr). Self-hosted means more infra work. This decision affects everything from Phase 4 onward.

6. **Ecosystem stability.** LangGraph 1.0 promises no breaking changes until 2.0, but `langgraph-prebuilt` already had a post-1.0 breaking change (Issue #6363). How much can we trust the API surface? Pin to specific minor versions (1.x) and use `langgraph-checkpoint` for persistence rather than rolling your own — it's the stable path through the churn.

7. **Teaching clarity vs. production realism.** The canonical cases are designed for clean demonstration — conflicting signals resolve neatly, tools return structured data, latency is acceptable. Real fraud data is messier: partial tool responses, ambiguous signals, 10x the noise. Decide early whether the tutorial optimizes for teaching clarity (clean traces, predictable behavior) or production realism (ugly edge cases, noisy data). We probably want teaching clarity for Phases 1-5 and production realism for Phase 7 — but name the switch explicitly when it happens.

---

## Thesis

Learn LangGraph properly by progressively building a fraud investigation system. Start with a simple LLM call, end with a production multi-agent system. Same business domain as the Dapr demo repo — when the composition guide comes later, the reader already knows both sides.

**Why fraud specifically:** Fraud investigation demands high auditability — every decision needs a trail. This plays to LangGraph's strengths: checkpointed state, human-in-the-loop interrupts, and inspectable graph execution. We introduce HITL formally in Phase 5, but the auditability requirement should inform design from Phase 3 onward (e.g., structured investigation notes in state, not just chat messages).

**Scope discipline:** This is a LangGraph tutorial wearing a fraud costume — not the other way around. Keep the fraud logic believable but minimal. Every feature must teach a LangGraph concept. If we're adding fraud complexity that doesn't illuminate a framework capability, we've drifted.

A separate composition guide (Dapr + LangGraph together) comes after both tutorials are done. This tutorial stands alone.

## Audience

Developers who want to learn LangGraph. May or may not know Dapr. No Dapr prerequisite.

Two sub-audiences:
- **LLM-curious developers** who've called the API but haven't built agents. Need to understand why a graph framework exists.
- **LangChain users** who know chains/prompts but haven't used LangGraph's stateful agent patterns. Need the state machine mental model.

## Through-Line: Fraud Investigation

Every phase builds on the same domain — analyzing orders for fraud. This gives progressive complexity a reason to exist:

| Phase | The fraud system can... |
|-------|------------------------|
| 1 | Score an order's risk (single LLM call) |
| 2 | Look things up before scoring (tool calling) |
| 3 | Decide what to investigate and when to stop (agent loop) |
| 4 | Remember what it learned about a customer (memory) |
| 5 | Escalate to a human analyst (human-in-the-loop) |
| 6 | Delegate to specialist investigators (multi-agent) |
| 7 | Run in production without falling over (hardening) |

---

## Phase Summary

| Phase | Name | Status |
|-------|------|--------|
| 1 | Your First Graph (Simple Fraud Scorer) | Planned |
| 2 | Tool Calling (Fraud Checker with Lookups) | Planned |
| 3 | ReAct Agent (Fraud Investigator) | Planned |
| 4 | Memory and State (Customer History) | Planned |
| 5 | Human-in-the-Loop (Analyst Escalation) | Planned |
| 6 | Multi-Agent (Investigation Team) | Planned |
| 7 | Production Hardening | Planned |

---

## Canonical Test Cases

Six cases used across all phases. Same inputs, different capabilities — this is how you demonstrate progression, not just claim it.

### Case 1: Obviously Legit
- Returning customer, 50 prior orders, no fraud flags
- Normal amount ($45), residential address
- **Expected:** Low risk across all phases. Phase 3: 0-1 tool calls. Baseline.

### Case 2: Mildly Suspicious
- New customer, no history
- Slightly high amount ($380), apartment address
- **Expected:** Some investigation, medium risk. Phase 3: 2-3 tool calls.

### Case 3: High Risk
- New email domain (disposable provider)
- High amount ($2,800), known freight forwarder address
- **Expected:** High risk, escalation. HITL triggered in Phase 5+. Phase 3: 3-5 tool calls.

### Case 4: Conflicting Signals
- 2-year customer, 30 successful orders
- High amount ($1,500), warehouse address
- Recent IP address change (new geo), slight shipping name mismatch vs account name
- Same warehouse address used by 2 other accounts this week
- **Expected:** Multi-tool reasoning, evidence weighing. The agent must not short-circuit on either signal. Requires sequential investigation: address lookup → cross-reference → history check → synthesis. This is the case that separates Phase 3 from Phase 2.
- **Pre-build validation:** Run this as a single prompt ("Score this fraud risk") before coding Phase 3. If the LLM resolves it without tools, add more conflicting signals until it can't.

### Case 5: Historical Fraud
- Customer flagged once 6 months ago, current order looks normal ($60, residential)
- **Expected:** Phase 3 scores low (no memory). Phase 4 scores higher (retrieves flag). This is the clearest illustration of why memory matters.

### Case 6: Tool Error / Edge Case
- Normal-looking order, but payment tool returns an error or incomplete data
- **Expected:** Phase 3 may handle poorly. Phase 7 handles gracefully (fallback, error reporting). Tests degradation behavior.

Build these before Phase 1. Use them throughout.

---

## Metrics (Track from Phase 3 Onward)

Don't wait for Phase 7 to start measuring.

| Metric | What it tells you | Warning sign |
|--------|-------------------|--------------|
| LLM calls per investigation | Reasoning depth | Simple cases >3 = prompt inefficiency. Complex cases <2 = not investigating. |
| End-to-end latency | User experience | >15s for any case = problem |
| Token usage (prompt + completion) | Cost per investigation | Memory phases will spike this — expected |
| Max recursion depth | Loop control | Hitting `recursion_limit` regularly = weak stop condition |
| Human escalation rate (Phase 5+) | Agent confidence | >50% escalation = agent isn't decisive enough |
| Accuracy vs. canonical cases | Correctness | Cases 4-5 are the real test |

---

## Pre-Build: Architecture Decisions

Lock these in before writing code. They propagate through every phase.

### 1. State Schema (Phase 3 Target, Backward-Compatible)

Phase 1 starts simple and each phase extends. The Phase 3 schema is the real target — design it now so Phases 1-2 are subsets, not rewrites.

```python
# Phase 1 — minimal
class FraudState(TypedDict):
    order: dict                          # raw order input
    risk_score: int                      # 0-100
    decision: str                        # "approve" | "flag" | "reject"

# Phase 2 — add messages for tool calling
class FraudState(TypedDict):
    order: dict
    risk_score: int
    decision: str
    messages: Annotated[list, add_messages]  # LLM + tool conversation

# Phase 3 — full investigation state (THE TARGET)
class FraudState(TypedDict):
    # Input
    order: dict                          # raw order: {id, customer_email, amount, items, shipping_address}

    # Conversation (LLM reasoning + tool results)
    messages: Annotated[list, add_messages]

    # Structured investigation output (NOT just chat)
    evidence: Annotated[list[Evidence], operator.add]  # accumulates per tool call
    risk_score: int                      # 0-100, set by calculate_risk_score tool
    decision: str                        # "approve" | "review" | "reject"
    investigation_complete: bool         # explicit termination signal

# Phase 4 adds (extends, doesn't change Phase 3 fields):
#   customer_history: list[PastInvestigation]  # retrieved from Store
#
# Phase 5 adds:
#   human_decision: str | None           # "approved" | "override" | None
#   human_notes: str | None              # analyst comments
```

**Key decisions:**
- `evidence` is a list with an `add` reducer — each node appends, never overwrites. This makes the investigation trail auditable.
- `investigation_complete` is **set only by `calculate_risk_score`**, never by the LLM directly. The LLM signals "I'm done investigating" by calling `calculate_risk_score`. That tool sets `investigation_complete = True` as a side effect. The routing function checks the flag and exits. This makes termination deterministic — the LLM can't skip scoring by just deciding it's done.
- `risk_score` and `decision` are set by a deterministic tool, not parsed from LLM prose. The LLM gathers evidence; the scoring is mechanical.
- Phase 1 → 2 → 3 is additive. No fields removed or renamed.

### 2. Investigation Note Format

Every tool call produces a structured `Evidence` record. This is what accumulates in `state["evidence"]` and what gets stored in long-term memory in Phase 4.

```python
class Evidence(TypedDict):
    tool: str              # which tool produced this: "check_customer_history", "verify_shipping_address", etc.
    finding: str           # human-readable summary: "Address matches known freight forwarder warehouse"
    risk_signal: str       # "low_risk" | "medium_risk" | "high_risk" | "neutral" | "error"
    confidence: float      # 0.0-1.0 — how confident the tool result is
    raw_data: dict         # full tool output for debugging/audit
    timestamp: str         # ISO 8601 — when the check was performed
```

**Why TypedDict (not Pydantic) for now:** Keep Phase 3 focused on LangGraph mechanics. Phase 7 upgrades to Pydantic for boundary validation — that's a hardening concern, not a learning concern.

**Why this structure:**
- `tool` + `finding` make investigation reports readable without replaying messages
- `risk_signal` lets the deterministic `calculate_risk_score` tool aggregate without LLM interpretation
- `confidence` matters for partial data (e.g., address check returns "unknown" vs "confirmed match")
- `raw_data` is for debugging — never shown to the LLM, always available in traces
- `timestamp` enables ordering and staleness checks in Phase 4 memory

**Phase 4 memory format** — when storing investigation results for cross-session retrieval:

```python
class InvestigationRecord(TypedDict):
    investigation_id: str       # unique ID
    customer_email: str         # lookup key
    order_id: str
    timestamp: str              # ISO 8601
    risk_score: int             # final score
    decision: str               # final decision
    evidence_summary: list[str] # finding strings from Evidence records (not full raw_data)
    flags: list[str]            # e.g., ["freight_forwarder", "high_amount", "new_customer"]
```

The `flags` list enables fast pre-filtering: "has this customer ever been flagged for freight_forwarder?" without replaying full investigations.

### 3. Single-Agent Baseline Architecture (Phase 3)

This is the architecture Phase 6 measures against. Make it good — a weak baseline makes multi-agent look better than it is.

```
                    ┌──────────────┐
                    │    START     │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  parse_order │  Extract order fields into state
                    └──────┬───────┘
                           │
               ┌───────────▼───────────┐
               │      call_llm         │◄─────────────────┐
               │  (system prompt +     │                   │
               │   evidence so far +   │                   │
               │   order context)      │                   │
               └───────────┬───────────┘                   │
                           │                               │
                ┌──────────▼──────────┐                    │
                │   should_continue   │                    │
                │   routing function  │                    │
                └──┬──────────────┬───┘                    │
                   │              │                        │
          has tool calls   investigation_complete          │
                   │        (set by calculate_risk_score)  │
                   │         OR no tool calls (fallback)   │
                   │              │                        │
          ┌────────▼────────┐     │                        │
          │  execute_tools  │─────┘ (loop back)            │
          │  (run tools,    │                              │
          │   append to     │──────────────────────────────┘
          │   evidence[])   │
          └─────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │   assess_risk       │  Final report assembly
                   │   (deterministic)   │  (score already computed by tool)
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │   format_report     │  Structured output
                   └──────────┬──────────┘
                              │
                       ┌──────▼──────┐
                       │     END     │
                       └─────────────┘
```

**Nodes (5):**

| Node | Input | Output | LLM? |
|------|-------|--------|------|
| `parse_order` | `state["order"]` (raw dict) | Validates fields, normalizes | No |
| `call_llm` | messages + evidence + order context | Tool calls or final assessment | Yes |
| `execute_tools` | Tool call messages | Tool results → messages + evidence (includes `calculate_risk_score` setting score + flag) | No (runs tools) |
| `assess_risk` | All state (score already set by tool) | Validates score was set, applies any final policy rules | No (deterministic) |
| `format_report` | All state | Final structured investigation report | No |

**Tools (5 core + 1 optional):**

| Tool | Input | Returns | Risk signal |
|------|-------|---------|-------------|
| `check_customer_history` | email | order count, fraud flags, account age | low/medium/high |
| `verify_shipping_address` | address | type (residential/commercial/warehouse/PO box), geo risk | low/medium/high |
| `check_payment_pattern` | email, amount | typical range, velocity, anomaly flag | low/medium/high |
| `search_fraud_database` | indicator (email/address/pattern) | known fraud matches | neutral/high |
| `calculate_risk_score` | evidence list | weighted score 0-100, decision, **sets `investigation_complete = True`** | N/A (deterministic) |

Optional (experiment, not core baseline):
| `get_investigation_notes` | None | current evidence summary (for LLM self-reflection) | N/A |

**Why `get_investigation_notes` is optional:** Giving the LLM a tool to summarize its own evidence risks reflection loops (calling it repeatedly for no new information), artificial latency, and prompt redundancy. The evidence is already in state — the LLM sees it in messages. Try without it first. Add it only if the agent demonstrably struggles to synthesize across tool results.

**`calculate_risk_score` logic (deterministic, not LLM):**
```python
def calculate_risk_score(evidence: list[Evidence]) -> dict:
    weights = {"high_risk": 30, "medium_risk": 15, "low_risk": -10, "neutral": 0, "error": 5}
    base_score = 20  # start at 20, not 0 — new orders aren't automatically safe
    score = base_score + sum(weights[e["risk_signal"]] * e["confidence"] for e in evidence)

    # Policy override: high-confidence high-risk signal gets extra weight
    # Demonstrates that deterministic != simplistic
    if any(e["risk_signal"] == "high_risk" and e["confidence"] > 0.8 for e in evidence):
        score += 10

    score = max(0, min(100, int(score)))  # clamp

    if score >= 80:
        decision = "reject"
    elif score >= 50:
        decision = "review"  # becomes HITL trigger in Phase 5
    else:
        decision = "approve"

    # This tool owns termination — the LLM doesn't set this directly
    return {"risk_score": score, "decision": decision, "investigation_complete": True}
```

**Routing function:**
```python
def should_continue(state: FraudState) -> Literal["execute_tools", "assess_risk"]:
    # calculate_risk_score sets this — deterministic termination
    if state.get("investigation_complete"):
        return "assess_risk"

    last_message = state["messages"][-1]

    # Agent wants to call tools (including calculate_risk_score)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tools"

    # Agent stopped calling tools without scoring — force exit
    # This is a fallback, not the normal path
    return "assess_risk"
```

**Termination flow:**
```
LLM decides "I have enough evidence"
    → calls calculate_risk_score(evidence)
        → tool sets investigation_complete = True
            → routing function sees flag → exits to assess_risk
```
The LLM never directly mutates `investigation_complete`. It signals completion by calling the scoring tool. The tool owns the flag. This makes termination deterministic and auditable.

**System prompt (core of the agent's behavior):**
The prompt tells the agent to:
1. Read the order details
2. Decide which checks are relevant (not all tools for every order)
3. Call tools, observe results, reason about next steps
4. When you have enough evidence, call `calculate_risk_score` — this finalizes the investigation

The prompt explicitly instructs: "Do NOT score risk yourself. Do NOT decide when the investigation is complete. Call `calculate_risk_score` with your evidence when you're ready. That tool handles scoring and completion."

This separation — LLM investigates, deterministic function scores and terminates — is critical for auditability.

### 4. Investigator System Prompt (v1)

This is the source of truth for Phase 3 agent behavior. It must handle Case 4 (conflicting signals, warehouse address reuse) without hallucinating risk.

```
You are a fraud investigation agent. Your job is to investigate orders for potential fraud by gathering evidence using your tools.

## Your Role

You INVESTIGATE. You do NOT score. You do NOT make final decisions.

When you have gathered enough evidence, call the `calculate_risk_score` tool. That tool computes the score and finalizes the investigation. Do not attempt to assign risk scores yourself.

## How to Investigate

1. Read the order details carefully: customer email, amount, items, shipping address.

2. Decide which checks are relevant. Not every order needs every tool:
   - New customer with no history? Check `check_customer_history` and `check_payment_pattern`.
   - Unusual shipping address? Check `verify_shipping_address`.
   - High amount from known customer? Check `check_payment_pattern`.
   - Any suspicious signal? Cross-reference with `search_fraud_database`.

3. After each tool result, reason about what you learned:
   - Does this signal increase or decrease suspicion?
   - Does this new information make another check more relevant?
   - Do you have conflicting signals that need resolution?

4. When you have enough evidence to form a complete picture, call `calculate_risk_score` with all gathered evidence.

## Critical Rules

- NEVER invent or assume data you haven't retrieved from a tool. If you haven't checked the address, don't claim it's suspicious.
- NEVER skip investigation because something "looks fine." Low-risk orders should still have at least one confirming check.
- When signals conflict (e.g., long customer history but suspicious address), investigate MORE, not less. Conflicting signals require additional evidence, not early termination.
- If a tool returns an error or incomplete data, note it as evidence with risk_signal "error" and continue investigating with other tools. Do not halt.
- Do not call the same tool twice with the same inputs.

## Evidence Format

For each tool call, record a structured finding:
- What tool you used
- What you found
- Whether it's a low_risk, medium_risk, high_risk, neutral, or error signal
- Your confidence in the finding (0.0-1.0)

## When to Stop Investigating

Call `calculate_risk_score` when:
- You have checked all signals you consider relevant
- Conflicting signals have been investigated with at least one additional cross-reference
- You have at least 2 pieces of evidence (no single-check investigations except for obviously routine orders from established customers)

Do not keep investigating after calling `calculate_risk_score`. That tool finalizes the investigation.
```

**Why this prompt works for Case 4:**
- "When signals conflict, investigate MORE, not less" prevents premature convergence on the warehouse address
- "Conflicting signals require additional evidence, not early termination" forces the cross-reference step
- "NEVER invent or assume data" prevents hallucinating that the warehouse is definitely suspicious without checking `search_fraud_database`
- The 2-evidence minimum prevents single-signal snap judgments

**What to watch for:**
- Does the agent follow the "investigate more on conflict" instruction consistently?
- Does it call `calculate_risk_score` at the right time, or does it over-investigate?
- Does it handle Case 1 (obviously legit) efficiently — one confirming check then score?
- Does the prompt need to explicitly mention the fast path, or does the agent figure it out from "not every order needs every tool"?

This prompt will evolve during Phase 3. v1 is a starting point, not a final artifact. Keep prompt versions in the repo (`prompts/v1_investigator.md`, `v2_investigator.md`, etc.) and reference the version in traces when accuracy regresses — you'll need to diff prompts to find what changed.

---

## Phase 1: Your First Graph (Simple Fraud Scorer)

**Goal:** Understand StateGraph, nodes, edges, and state flow. Build a graph that scores an order for fraud — same thing the Dapr demo does in Phase 7-A, but in Python with LangGraph.

**Setup cost framing:** "You don't need LangGraph for this. A function call would work. But we're paying the setup cost once — StateGraph, state schemas, node functions, compile/invoke — because Phase 3 adds loops, and loops are where LangGraph earns its keep. If the graph concepts feel heavy for what we're building here, that's the right instinct. Keep going."

This phase will also take longer than the code suggests — environment setup (Python, dependencies, API keys, first successful LLM call) is real work. Budget accordingly.

**Build:**
```
START → parse_order → score_fraud → format_result → END
```

Three nodes, linear flow. The `score_fraud` node calls an LLM to score risk 0-100.

**Covers:**
- `StateGraph`, `START`, `END`
- State schema with `TypedDict`
- Nodes as functions that receive state and return updates
- Normal edges (A → B)
- `graph.compile()` and `graph.invoke()`

**Then add:** A conditional edge after `score_fraud` — if score >= 80, route to `flag_order`; otherwise route to `approve_order`. First branching logic.

**Covers additionally:**
- Conditional edges
- Routing functions with `Literal` return types
- The pattern: LLM output drives the next step

**What the reader should feel:** "OK, I see how state flows through nodes. The graph is explicit about what happens in what order. But this is still just a pipeline — I could do this without LangGraph."

---

## Phase 2: Tool Calling (Fraud Checker with Lookups)

**Goal:** Give the LLM tools. Instead of scoring blind, the fraud checker can look things up before deciding.

**Tools:**
- `check_customer_history(customer_email)` → returns past order count, past fraud flags
- `verify_shipping_address(address)` → returns whether address is a known warehouse/PO box
- `check_payment_pattern(customer_email, amount)` → returns whether amount is unusual for this customer

**Build:**
```
START → call_llm → [has tool calls?]
                     ├── yes → execute_tools → call_llm (loop back)
                     └── no → END
```

**Covers:**
- `@tool` decorator and tool definitions
- `model.bind_tools(tools)` — giving the LLM awareness of available tools
- `ToolNode` (prebuilt tool executor)
- `tools_condition` (prebuilt router: tool calls → tools node, no tool calls → END)
- The first loop: LLM → tools → LLM → tools → ... → END
- `MessagesState` and the `add_messages` reducer

**Critical teaching moment — reducers:** This is where `add_messages` must be explicitly taught. It's the single most common LangGraph confusion ("why did my previous messages disappear?"). Show the wrong way first:
1. Start with `messages: list` — show that returning new messages overwrites the old ones
2. Switch to `messages: Annotated[list, add_messages]` — show that messages now accumulate
3. Explain: "LangGraph doesn't mutate state. Nodes return deltas. Reducers define how deltas merge. This is the mental model for everything that follows."

**Reducer reference (use this table in Phase 2, reuse in Phase 3 when evidence is added):**

| Field | Type | Reducer | What happens when a node returns this field |
|-------|------|---------|---------------------------------------------|
| `order` | `dict` | (default) | Overwrite — new value replaces old |
| `messages` | `list` | `add_messages` | Append — new messages added to existing list |
| `evidence` | `list` | `operator.add` | Concatenate — new findings appended, never overwritten |
| `risk_score` | `int` | (default) | Overwrite — only `calculate_risk_score` sets this |
| `investigation_complete` | `bool` | (default) | Overwrite — only `calculate_risk_score` sets this |

**Teaching point:** "The LLM decides which tools to call and when to stop. You didn't hardcode 'check address then check history.' The LLM reads the order and decides what's relevant. This is the first step toward an agent."

**What the reader should feel:** "The loop is the new thing. The LLM is making decisions about control flow, not just producing output."

---

## Phase 3: ReAct Agent (Fraud Investigator)

**Goal:** Build a proper ReAct agent. The fraud investigator reasons about what to check, calls tools, observes results, reasons again, and decides when it has enough evidence.

**This is where LangGraph earns its keep.**

**Build:** Start with `create_react_agent` (prebuilt) to show the simplest version. Then rebuild it by hand for full understanding.

**Hand-built version covers:**
- The ReAct loop: reason → act → observe → repeat
- Custom `call_llm` node with system prompt (fraud investigation instructions)
- Custom `execute_tools` node
- Custom routing function (`should_continue`)
- Why the loop terminates (LLM returns message without tool calls)
- State schema with messages + custom fields (investigation notes, risk assessment)

**New tools added:**
- `search_fraud_database(indicator)` → check known fraud patterns
- `calculate_risk_score(evidence)` → deterministic scoring from gathered evidence

**Prompting matters:** Include the system prompt that makes the agent good at fraud investigation. Discuss what happens when you change the prompt — the same graph behaves very differently. "The graph is the skeleton. The prompt is the brain."

**Why deterministic scoring matters (visualize this):**
```
WHAT WE BUILD:                          WHAT WE DON'T:

LLM Investigation Loop                  LLM
    ↓                                       ↓
Evidence[] (structured)                 "Risk Score: 87, this looks
    ↓                                   suspicious because..."
calculate_risk_score()                      ↓
    ↓                                   ??? (how do you audit this?)
risk_score = 73 (deterministic)
    ↓
decision = "review" (policy)
```
Left side: a regulator can inspect the weights. Right side: a black box. This isn't just pedagogy — it's industry realism for FinTech systems. The LLM generates hypotheses; Python enforces policy.

**Fast-path optimization:** For obviously legit orders (Case 1), the graph should be able to exit after `parse_order` → one tool call → `assess_risk`, skipping the full investigation loop. This shows cost awareness and demonstrates graph branching power. Add a conditional edge from `parse_order`: if all signals are clean (returning customer, low amount, residential address), short-circuit to `assess_risk` with minimal evidence.

**Grounding test:** Add a no-op tool: `manual_review_not_required()` that returns "No manual review needed." If the agent calls it, the prompt is too tool-hungry — it's calling tools for the sake of calling tools rather than because the investigation requires it. This is a cheap diagnostic for prompt quality.

**Honest notes section:**
- What happens when the agent loops too many times? (recursion_limit)
- What happens when the agent calls a tool that doesn't exist? (hallucinated tool calls)
- What happens when the agent gives a wrong answer confidently? (the hardest failure mode)

**Phase 3 success criteria (the hinge test):**
Phase 3 is where LangGraph either earns its keep or doesn't. You know it worked if:
- Complex cases (4, 5) require 3-6 LLM calls
- Tool usage differs per scenario (not the same sequence every time)
- Investigation notes accumulate meaningfully in state
- The agent sometimes changes its assessment mid-loop as new evidence arrives
- `recursion_limit` becomes a relevant concern, not theoretical

If all canonical cases resolve in 1-2 calls, the tutorial's thesis is weak. Redesign the prompt or tools before moving on.

**What the reader should feel:** "This is genuinely different from a pipeline. The agent is investigating, not just scoring. But it's also unpredictable — I need to think about guardrails."

---

## Phase 4: Memory and State (Customer History)

**Goal:** The fraud investigator remembers previous investigations. If it flagged this customer last week, it knows that context.

**Scope note:** This phase covers the memory APIs — how to save and retrieve state. Context management (what happens when state gets too large) moves to Phase 7 where it belongs alongside other production concerns. We'll note the problem here but solve it there.

**Two types of memory:**

### Short-term (within one investigation)
- Checkpointing with `MemorySaver`
- Thread IDs for conversation continuity
- "The agent investigated three tools, then was interrupted. When it resumes, it picks up where it left off."

### Long-term (across investigations)
- `Store` API with namespaces
- After each investigation, store findings: `store.put(("customers", email), investigation_id, findings)`
- Before each new investigation, retrieve past findings: `store.search(("customers", email))`
- The agent's system prompt includes relevant history

**Covers:**
- `MemorySaver` (dev) and `PostgresSaver` (production)
- `get_state()`, `get_state_history()`, `update_state()` — time travel
- `InMemoryStore` for long-term memory
- Namespaces and semantic search
- The difference between graph state (ephemeral) and stored memory (persistent)

**Foreshadowing (solved in Phase 7):** "What happens when conversation history grows beyond the context window? We'll hit this problem. For now, know that `trim_messages` and summarization nodes exist. Phase 7 covers the actual engineering."

**What the reader should feel:** "Memory makes the agent dramatically more useful. The APIs are straightforward. But I can already see that managing what's in memory will get complicated at scale."

---

## Phase 5: Human-in-the-Loop (Analyst Escalation)

**Goal:** High-risk orders pause for human review. The agent presents its evidence, a human analyst approves or overrides, and the investigation continues.

**Build:**
```
... → assess_risk → [score >= 70?]
                      ├── yes → interrupt("Review this investigation") → human decides
                      │         ├── approved → finalize_report
                      │         └── override → re_investigate (with human guidance)
                      └── no → finalize_report → END
```

**Covers:**
- `interrupt()` function — dynamic breakpoints
- `Command(resume=value)` — resuming with human input
- Compile-time breakpoints (`interrupt_before`, `interrupt_after`) vs runtime `interrupt()`
- Editing state mid-execution (`graph.update_state()`)
- Why checkpointing is required (state must persist while waiting for human)

**Patterns demonstrated:**
- **Approval gate:** Agent proposes action, human approves/rejects
- **Review and edit:** Agent generates report, human edits before finalization
- **Guided re-investigation:** Human says "check the shipping address more carefully" — agent re-enters the loop with new instructions
- **State forking / "What-If" analysis:** Analyst looks at the investigation, disagrees with the address check result, and wants to see what happens with different evidence. Use `get_state_history()` to find the checkpoint after `execute_tools` ran the address check, then `update_state(values={"evidence": corrected_evidence}, as_node="execute_tools")` to inject a corrected result, and re-run from that point. The graph forks — original investigation preserved, corrected branch runs independently. This is the killer feature for fraud audit workflows: "What would the agent have concluded if the address wasn't flagged?"

**Production consideration:** "In a real system, the interrupt goes to a queue (email, Slack, dashboard). The graph resumes hours or days later. This is where checkpointing to Postgres matters — MemorySaver won't survive a restart."

**What the reader should feel:** "Human-in-the-loop isn't bolted on — it's a first-class pattern. The interrupt/resume model is clean. And state forking means I can ask 'what if?' without losing the original investigation."

---

## Phase 6: Multi-Agent (Investigation Team)

**Time budget warning:** This phase will likely take 2-3x longer than any previous phase. Multi-agent involves debugging agent-to-agent communication, state transformation at subgraph boundaries, and prompt engineering for the supervisor's routing decisions. Each of these is independently tricky. Together, they compound. Plan accordingly and don't rush — this is where most people get stuck.

**Open question (tension #3):** We may discover during Phase 3 that a single well-prompted agent with more tools outperforms a multi-agent setup for fraud investigation. If so, this phase becomes "here's how multi-agent works and here's why we didn't need it." That's a valid outcome — be ready for it.

**Goal:** Split the monolithic fraud investigator into specialized agents coordinated by a supervisor.

**Agents (with heterogeneous models — this is what justifies multi-agent):**

| Agent | Role | Model | Why this model |
|-------|------|-------|----------------|
| **Supervisor** | Route to specialists, synthesize final assessment | High-reasoning (Claude Sonnet / o1) | Cross-signal synthesis requires strong reasoning |
| **Address Analyst** | Shipping address verification, geo-risk, warehouse detection | Cheap + fast (GPT-4o-mini) | String parsing and pattern matching |
| **Payment Analyst** | Payment patterns, amount anomalies, velocity checks | Cheap + fast (GPT-4o-mini) | Numeric anomaly detection |
| **Customer Analyst** | Customer history, account age, past fraud flags | Mid-tier (Claude Haiku / GPT-4o) | Long history summarization needs decent context handling |

If all agents use the same model with the same reasoning complexity, you've built orchestration theatre. The specialist test: do agents genuinely require different capabilities? If not, single-agent wins.

**Build (two approaches):**

### Approach 1: Supervisor pattern (recommended first)
```python
from langgraph_supervisor import create_supervisor

workflow = create_supervisor(
    [address_agent, payment_agent, customer_agent],
    model=llm,
    prompt="You are a fraud investigation supervisor..."
)
```

### Approach 2: Hand-built with subgraphs
Each specialist is a subgraph with its own state schema. The supervisor graph invokes them as nodes, transforming state at boundaries.

**Covers:**
- `create_supervisor` (prebuilt)
- Handoff tools — supervisor delegates to agents
- Subgraphs — nesting graphs within graphs
- Shared state vs isolated state per agent

**Parallelism deferred:** Parallel specialist execution (address and payment checks simultaneously) is tempting but adds state merge complexity, ordering issues, trace confusion, and reducer subtlety. Multi-agent + parallelism + subgraphs is a cognitive explosion. Stage it:
- Phase 6: multi-agent, sequential execution. Learn coordination first.
- Phase 7 (or optional 6B): parallel specialization as an optimization. By then the reader understands both multi-agent and production concerns.

**Honest assessment:**
- When is multi-agent better than one agent with more tools? (Often it isn't — tool routing in a single agent is simpler)
- What's the latency cost? (Each agent handoff is another LLM call)
- Does the supervisor actually make good routing decisions? (Depends heavily on the prompt)

**Measurement approach (don't rely on vibes):**
1. Clone Phase 3 single-agent as baseline
2. Measure all 6 canonical cases: accuracy, latency, LLM calls, token usage
3. Build supervisor + specialists
4. Run the same 6 cases
5. Compare side by side

If single-agent performs within striking distance at half the latency, say so. "We tried multi-agent. It was more complex and slower. For this domain, single-agent is better." That conclusion makes the tutorial stronger, not weaker.

**The unit economics acid test (run on Case 4):**
Multi-agent isn't just about organization — it's about optimizing the cost of reasoning.
- **Single agent:** ~3,000 tokens on an expensive model (all reasoning happens in one context)
- **Multi-agent:** ~500 expensive tokens (supervisor routing + synthesis) + ~2,500 cheap tokens (specialists doing string parsing, numeric checks)
- **If multi-agent uses fewer expensive tokens at comparable accuracy, it wins on unit economics even if total tokens increase.**

This reframes the question from "is multi-agent worth the complexity?" to "does routing cheap work to cheap models save money at scale?" That's a real production question.

**Comparison metrics (relative to single-agent baseline — absolute numbers rot as pricing shifts):**

| Metric | Single-Agent (Phase 3) | Multi-Agent (Phase 6) | What it means |
|--------|----------------------|---------------------|---------------|
| Expensive-model tokens per Case 4 | baseline (all tokens) | delta % (supervisor only) | The real cost comparison |
| Cheap-model tokens per Case 4 | 0 | specialist tokens | The "offloading" dividend |
| Trace depth | 5-10 nodes | 15-20 nodes | High depth suggests loop-de-loop handoffs |
| Time to first tool call | baseline | delta | How fast does actual investigation start? |
| End-to-end latency | baseline | delta | The cost of coordination |
| Accuracy on Cases 3-5 | baseline | delta | Does complexity buy correctness? |

**What the reader should feel:** "Multi-agent is powerful for complex domains where specialists genuinely have different expertise. But it's not always better than a well-prompted single agent. The overhead is real."

---

## Phase 7: Production Hardening

**Goal:** Everything that makes the difference between a demo and a system.

### Schema Hardening (TypedDict → Pydantic)
- Phase 3 uses `TypedDict` for `Evidence` and state — simpler, less cognitive load while learning LangGraph
- Phase 7 upgrades to Pydantic models — validation at node boundaries, type coercion, clear schema errors
- Show what happens when a tool returns malformed evidence (TypedDict: silent corruption; Pydantic: caught at boundary)
- This creates a natural "hardening" arc: prototype fast, validate later

### Error Handling
- `recursion_limit` — prevent infinite loops (default 25, tune per use case)
- `handle_tool_errors=True` on ToolNode — report hallucinated tool calls back to the LLM
- Fallback nodes — catch errors and degrade gracefully
- Circuit breaker pattern — after N consecutive errors, exit with error message

### The "Dead End" Node
What if the LLM never calls `calculate_risk_score` after 10 loops? It just keeps re-checking tools or apologizing. Phase 3's deterministic termination handles the normal path, but the failure path needs coverage:
- Count loop iterations in state (add `loop_count: int` field)
- After N iterations without scoring, force-exit to a `dead_end` node
- `dead_end` calls `calculate_risk_score` with whatever evidence exists, flags the investigation as `"decision": "review"` with a note: "investigation did not converge"
- This prevents infinite loops AND preserves auditability — you can see what the agent was doing when it got stuck

### Token Budget Circuit Breaker
Kill the graph if it exceeds a dollar threshold in a single thread:
- Track cumulative token usage across nodes (prompt + completion)
- Apply model-specific pricing to compute running cost
- If cost exceeds budget (e.g., $0.50 per investigation), interrupt with `"decision": "review"` and flag for human attention
- This is especially important for multi-agent (Phase 6) where chatter between supervisor and specialists can compound costs silently

### Streaming
- `stream_mode="messages"` — token-by-token streaming for the UI
- `stream_mode="updates"` — node completion events for progress tracking
- `stream_mode="custom"` — application-specific events from inside nodes
- Multiple modes simultaneously

### Observability
- LangSmith setup — traces, evaluations, datasets
- What LangSmith shows that print statements don't
- Cost monitoring (token usage per investigation)
- Latency profiling (which agent/tool is the bottleneck?)

### Evaluation
- Offline: curated fraud cases with known outcomes, run the system, score accuracy
- Online: monitor production investigations for quality drift
- LLM-as-judge for open-ended assessment quality

### Context Engineering (moved from Phase 4)
- What happens when conversation history grows beyond the context window
- `trim_messages` — keep only recent messages
- Summarization node — periodically compress old messages into a summary
- Tool output compression — post-process verbose tool results
- Token counting and monitoring
- "This is the part everyone underestimates. LLM state is massive. You will hit context limits."

**State compression strategy:** Prune `messages` (verbose, LLM conversation) but keep `evidence` (structured, compact) intact. The evidence list is the audit trail — never compress it. Messages are the reasoning scratchpad — summarize and discard. This is why the two-layer state design (messages + evidence) pays off: you can aggressively manage one without touching the other.

### The Double-Write Problem (Dapr bridge)
- What happens when the agent writes to its internal `Store` but the parent system fails before the transaction completes?
- This is distributed systems 101 — not unique to LangGraph, but relevant when the composition guide adds Dapr
- Don't solve 2PC. Instead: idempotent investigation IDs, status flags (started/completed/failed), write-ahead pattern
- Acknowledge eventual consistency. The agent may have recorded a finding that the orchestrator doesn't know about yet.
- This is the bridge to the composition guide — mention it here, solve it there

### Deployment and Cost Reality
- LangGraph Cloud vs self-hosted (FastAPI + PostgresSaver)
- When LangGraph Cloud is worth it vs rolling your own
- Docker containerization for self-hosted

**Deployment cost/lock-in discussion (be honest here):**
- **LangGraph Cloud**: Managed hosting, built-in persistence, streaming, cron. But pricing is opaque (usage-based, no public calculator as of Feb 2026). Once you depend on their managed checkpointing and deployment APIs, migration to self-hosted is non-trivial.
- **Self-hosted**: Full control, clear costs (your compute + Postgres). But you own the infra: health checks, scaling, zero-downtime deploys, persistence management. This is real ops work.
- **Composition guide implication**: If this tutorial deploys to LangGraph Cloud, the later Dapr composition guide gets harder — Dapr sidecars don't naturally fit LangGraph Cloud's managed runtime. Self-hosted (FastAPI + Docker) composes cleanly with Dapr. This may influence our recommendation.
- **What to cover**: Both paths, with honest trade-offs. Let the reader choose, but flag the composition guide consideration for readers coming from the Dapr side.

**What the reader should feel:** "Production is where the real work is. The agent logic is maybe 30% of the effort. Hardening, monitoring, and evaluation are the other 70%. And the deployment choice has real long-term consequences."

---

## What's NOT Covered Here (Future Composition Guide)

This tutorial is standalone LangGraph. A separate guide will cover:

- Running LangGraph agents inside Dapr-sidecarred services
- Dapr infrastructure (mTLS, secrets, resiliency) applied to LangGraph agents
- Dapr actors for stateful agent hosting (ConversationActor pattern)
- Zipkin + LangSmith observability side by side
- Scaling LangGraph agents with Dapr HPA and actor placement
- Decision framework: when to use Dapr alone, LangGraph alone, or composed

The composition guide depends on finishing both this tutorial and the Dapr tutorial (Phases 1-12 of dapr_demo). It tests the thesis: "Dapr owns infrastructure, LangGraph owns agent authoring, they compose."

---

## Connections to Dapr Demo Repo

| LangGraph Phase | Dapr Demo Phase | Shared Concept |
|----------------|----------------|----------------|
| Phase 1 (fraud scorer) | Phase 7-A (fraud check in saga) | Same domain, different tool |
| Phase 4 (memory) | Phase 12 (actors) | Stateful entities, different mechanism |
| Phase 5 (human-in-the-loop) | Phase 7 (manual AI review) | Human oversight of AI, different UX |
| Phase 7 (production) | Phase 11 (resilience) | Hardening, different layer |

These connections are for the composition guide later. This tutorial doesn't require the Dapr repo.

---

## When Tensions Resolve

See **Key Tensions** at the top. Here's when we expect answers:

| Tension | Resolved by |
|---------|-------------|
| #1 Framework vs. simplicity | Phase 3 — does the graph structure pay off when loops arrive? |
| #2 Prebuilt vs. hand-built | Phase 3 — we build both ways and compare |
| #3 Single vs. multi-agent | Phase 3 first impression, Phase 6 definitive answer |
| #4 LangChain dependency | Phase 1 starts tracking, accumulates through build |
| #5 Managed vs. self-hosted | Phase 7 — cost and composition guide implications |
| #6 Ecosystem stability | Every phase — track what breaks or shifts |
| #7 Teaching clarity vs. production realism | Phase 7 — explicit switch from clean demos to ugly reality |

---

## Research Notes

### When You Need LangGraph vs When You Don't

**You need LangGraph when:**
- Workflow has conditional branching based on LLM output
- You have cycles/loops (reason → act → observe → repeat)
- You need persistent state across steps
- You're building multi-agent systems
- You need human-in-the-loop (pause, review, resume)

**You don't need LangGraph when:**
- Single LLM call with a prompt
- Linear pipeline (prompt → LLM → parse → return)
- Basic RAG without agentic behavior
- Simple chatbot without complex branching

Native SDK calls show ~40% speed improvement over framework-mediated calls for simple tasks. "If you can draw your workflow as a straight line, do not use a graph framework."

### What's Actually Hard About LangGraph

From production users and community reports:

1. **The abstraction tax.** For simple agents, you're paying complexity costs for features you don't need.
2. **Debugging.** When a multi-node graph misbehaves, tracing state through LangGraph's abstractions is significantly harder than debugging a linear script. LangSmith helps but adds another dependency.
3. **Ecosystem churn.** Prebuilt module deprecated, Platform renamed to LangSmith Deployment, LangChain/LangGraph boundary shifting. APIs continue evolving.
4. **Latency is inherent.** Agent loops mean multiple LLM round-trips. 3-8 calls at 1-3 seconds each = 5-20+ second response times. Fundamental, not a bug.
5. **Learning curve is front-loaded.** Once you internalize state → node → updated state, building is fast. Getting there takes real investment (one developer reported 16 days for first agent).
6. **Memory management at scale is manual.** Write, select, compress, isolate — all require per-use-case design. No magic.

### LangGraph vs Alternatives (Current Landscape)

| Framework | Best For | Watch Out |
|-----------|----------|-----------|
| **LangGraph** | Complex stateful workflows, precise control | Steep curve, overkill for simple agents |
| **CrewAI** | Collaborative multi-agent (role-based) | Overhead for single-agent tasks |
| **AutoGen/Semantic Kernel** | Azure/.NET shops, enterprise | Microsoft-centric, merging into unified framework |
| **OpenAI Agents SDK** | Simple single agents with OpenAI models | Limited to OpenAI ecosystem |
| **Native provider SDKs** | Simple tool calling, one-shot tasks | No state management, no graphs |

### LangGraph + LangChain: The Dependency Question

- Core graph framework (StateGraph, nodes, edges, checkpointing) has **no LangChain dependency**
- Prebuilt components (`ToolNode`, `create_react_agent`) use LangChain types
- As of 1.0: `langgraph.prebuilt` deprecated, moved to `langchain.agents`
- Many production teams use LangGraph without LangChain, building custom nodes
- For this tutorial: we'll start with LangChain for convenience, note where it can be removed

### Current Versions (February 2026)

- `langgraph` core: 1.0.x (GA since October 2025)
- `langgraph-sdk`: 0.3.1
- `langgraph-prebuilt`: 1.0.2 (deprecated, moved to `langchain.agents`)
- Commitment: no breaking changes until 2.0

---

## Breaking Change Watchlist (Living Section)

Update this as you build. Check before starting each new phase.

| Date | Package | What broke | Impact | Fix |
|------|---------|-----------|--------|-----|
| 2025-Q4 | `langgraph-prebuilt` 1.0.2 | Deprecated, moved to `langchain.agents` | Import paths changed | Update imports, pin `langchain` version |
| 2025-Q4 | `langgraph-prebuilt` | Issue #6363 — breaking change post-1.0 | Prebuilt component API shifted | Pin to known-good version |

**Pinned versions for this tutorial (update as validated):**
```
langgraph==1.0.x
langgraph-checkpoint==x.x.x
langchain-core==x.x.x
langchain-anthropic==x.x.x  # or langchain-openai
```
Fill in exact versions after Phase 1 setup. These become the `requirements.txt` baseline.
