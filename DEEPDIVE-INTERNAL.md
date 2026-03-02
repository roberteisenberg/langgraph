# LangGraph Deep Dive — Internal Planning

Private working notes. Starting point — expect structure and conclusions to change as we build and learn.

---

## In-Process Thoughts (Living Section — Update As We Learn)

**Core understanding (arrived at through questioning, not documentation):**

LangGraph is a state machine for workflows where the LLM decides what happens next.

**The key mental model:** LangGraph doesn't reason — the LLM does. LangGraph is the stateless LLM's body: it accumulates state between calls, re-feeds context each time, executes the tools the LLM asks for (the LLM can't run code), enforces guardrails (recursion limits, token budgets), and handles pause/resume for human-in-the-loop. The LLM is the brain that decides what to investigate and when to stop. LangGraph is the loop that keeps calling it back.

**The brain isn't empty — it comes with years of medical school.** The LLM brings general intelligence from training: what fraud looks like, what's normal in e-commerce, what "reshipping" means, what account takeover patterns are, common sense about geography and human behavior. The tools bring specific evidence: THIS customer's history, THIS address's risk profile, THIS payment's anomaly score. The LLM applies its general intelligence to the specific data — that's the synthesis. Without training data, tool results are just dictionaries. Without tools, the LLM is guessing (that's Phase 1). You need both. The system prompt focuses the intelligence: what role to play, how to investigate, when to stop. **Training data + tool results + system prompt = intelligent investigation.** Remove any one and it degrades.

Specifically: LangGraph routes to state endpoints on the graph intelligently based on context the LLM derives by selecting the most appropriate tools available to it. The loop — not a linear pipeline — is the primary pattern.

**The fundamental shift:** Traditional state machines route on **data values** (`if score >= 80: reject`). LLM-driven state machines route on **data meaning**. The LLM synthesizes multiple signals and makes a judgment call about what to do next — no developer wrote that rule. A warehouse address alone might be fine, but a warehouse address + IP change + name mismatch means something different than any of those signals individually. You can't draw this decision tree in advance because it depends on context, not thresholds. LangGraph makes this new kind of non-deterministic, meaning-based branching manageable — visualization, state accumulation, and guardrails for decisions you can't predict at design time.

**What makes this different from traditional workflow engines (Dapr Workflow, Temporal):**
- Traditional: deterministic transitions. Step A always goes to step B or C based on a known condition. All paths defined at design time.
- LangGraph: LLM-determined transitions at runtime. The agent decides which tool to call, whether to keep investigating, and when it's done. You define the structure (which nodes exist, what edges are possible), but the LLM chooses the path through that structure each time.

**Three facets of one concept (LLM-driven loop with accumulated state):**
1. **LLM-driven routing** — the state machine where the LLM picks the path (the structure)
2. **Dynamic tool selection** — the LLM choosing WHICH tools to call based on what it sees (the mechanism — this IS how the LLM drives routing)
3. **Progressive investigation** — each loop iteration builds on accumulated context, so the LLM makes different decisions on iteration 3 than iteration 1 (the payoff)

These aren't independent benefits. They're one thing. Dynamic tool selection is how the loop works. Progressive investigation is why the loop matters.

**The utilities serve the loop — and these are the concrete value props over a plain while loop:**

- **Graph visualization** — `graph.get_graph().draw_mermaid_png()` renders your entire agent as a diagram. You can SEE the nodes, edges, and routing. For debugging, onboarding teammates, and explaining to stakeholders, this is huge. A while loop with if-statements is invisible. A graph is inspectable. Not mentioned enough in our doc — needs prominence.
- **Streaming** — `stream_mode="messages"` gives token-by-token output; `stream_mode="updates"` gives node-completion events. In a while loop, you get nothing until the whole thing finishes. LangGraph streams progress as it happens — which tool is running, what the LLM is thinking, when a node completes. For any UI, this is table stakes.
- **Recursion limits** — `recursion_limit=25` is one line. In a while loop, you hand-build a counter, decide what happens when it triggers, handle cleanup. LangGraph makes runaway agents a solved problem.
- **Checkpointing / HITL** — pause mid-loop, persist state, resume hours later with human input. Building this from scratch is weeks of work.
- **Reducers** — `add_messages`, `operator.add` for evidence accumulation. Declarative rules for how state merges across loop iterations. In a while loop, you manage list appending manually and eventually get it wrong.
- **Error handling** — `handle_tool_errors=True` catches hallucinated tool calls and feeds errors back to the LLM. In a while loop, that's a try/except you forget to write.

**What LangGraph does NOT optimize:** Token management (you manage context size), LLM call caching (that's provider-level — Anthropic prompt caching, etc.), cost per call (same API calls as a while loop). The savings is developer time and plumbing, not runtime efficiency.

**Mechanisms for dynamic progression (from research):**
- `add_conditional_edges` — routing functions that inspect state after a node runs
- `Command()` API — nodes return routing instructions alongside state updates (`Command(update={...}, goto="next_node")`). Enables "edgeless" architectures where routing lives inside nodes, not just in edge functions. We mention this for HITL resume in Phase 5, but it's more fundamental.
- Message state persistence (`Annotated[list, add_messages]`) — accumulated conversation history that the LLM uses to make routing decisions
- Plan-and-execute pattern — alternative to ReAct where a planner generates a DAG of tasks, executor runs them, a "joiner" decides whether to replan. Different from our sequential investigation loop. Worth exploring.

**Memory is NOT a core LangGraph benefit:**
- Graph state: ephemeral, gone after `invoke()` completes
- Checkpointing (`MemorySaver`/`PostgresSaver`): thread-scoped pause/resume — "save game," not memory. Every workflow engine does this.
- Store API (`InMemoryStore`): cross-thread, but `InMemoryStore` dies on restart. With a persistent backend, it's just a database abstraction.
- Cross-investigation context is YOUR responsibility. LangGraph provides lifecycle hooks and patterns for injecting external context, not the persistence itself.

**Honest pitch:** "Whether LangGraph is worth it depends on whether you need the loop. For Case 4 (conflicting signals requiring sequential investigation with dynamic tool selection), you do. For a single LLM call with tools, you don't."

**The data is deterministic. The reasoning is not.** The LLM only knows what you give it — customer profiles from your database, order details from the request, address checks from your API, fraud matches from your fraud DB, external data from MCPs. All of it is facts, numbers, records. No intelligence in the data itself. The LLM's job is synthesis across that data: given `account_age: 730 days` + `prior_orders: 30` + `address_type: warehouse` + `shared_accounts: 2` + `ip_changed: true` + `name_mismatch: true`, the LLM weighs these together and concludes "compromised account, not a loyal customer going rogue." No single field triggers that — the combination does, and the LLM reads the combination without you pre-coding every permutation. But it can only reason about data it can see. If your tools don't surface the right data, the LLM is guessing. **Rich deterministic data + good tool descriptions + good system prompt = intelligent reasoning. Skimp on any of those three and the LLM fails.**

**Data architecture (hybrid):** Pass known context at invocation time (customer history, profile). Use tools/MCPs for runtime discoveries (cross-referencing suspicious addresses, checking fraud databases for patterns discovered during investigation).

**Tutorial story arc (implemented in phase structure below):**

| Act | Phases | Message |
|-----|--------|---------|
| "You don't need it yet" | 1 | Linear pipeline. Honest that LangGraph adds nothing here. |
| "Now you do" | 2-3 | Tools arrive, the loop appears, the investigator emerges. THE HINGE. |
| "Here's what you get" | 4 | Infrastructure payoffs: viz, streaming, guardrails. Why not just a while loop. |
| "Real workflows" | 5 | Checkpointing, HITL, context injection. Production workflow patterns. |
| "Scale and specialize" | 6 | Multi-agent. When one agent isn't enough. Honest assessment. |
| "Ship it" | 7 | Hardening, deployment, observability. The other 70%. |

**Dynamic routing to departments (explore in Phase 6):**
The `Command()` pattern and conditional edges enable routing to specialist nodes based on what the LLM discovers. In our fraud domain: the supervisor could route to finance (payment anomalies), fraud department (known fraud patterns), or customer service (account issues) based on what the investigation reveals. This is the multi-agent payoff — not just splitting work, but routing to the right specialist dynamically.

### Research Findings — How People Actually Use LangGraph (Feb 2026)

**Does the real world validate "the loop is the core value"?**

Yes. Every production implementation (LinkedIn recruiting, Uber code migration, Replit, Elastic security, Exa web research) is built around the call-model → check-for-tools → execute-tools → loop-back cycle. However, a common Hacker News critique: "Every single 'graph' was the same single solution" — most graphs collapse into one loop pattern. If you're only using the loop, a plain `while` is simpler. The real value practitioners cite is the **infrastructure around the loop** — checkpointing, HITL interrupts, durable state. LangGraph itself acknowledged this by shipping a Functional API (2025) that lets you express the same pattern as a literal Python while loop with decorators.

**Most "fraud detection" LangGraph examples are not truly agentic.** Many articles labeled "multi-agent fraud detection" are sequential pipelines (node A → B → C → D) dressed up with graph syntax. The LLM doesn't decide what to investigate — it just processes in a fixed order. Watch for "agent washing." The exceptions that ARE truly agentic:
- A Neo4j + LangGraph system using plan-act-reflect loops for relationship analysis
- The FAA Framework (academic, arxiv:2506.11635) — LLM autonomously plans investigation steps, averages 7 steps per case, 71-72% high-impact evidence collection
- Amazon's Tier 3 compliance system — multi-agent LLM investigation, but only for the tiny fraction of 2B daily transactions that survive cheap filters

**Amazon's 3-tier architecture is the production pattern:**
1. Tier 1: Cheap fuzzy matching + vector embeddings (high recall, fast)
2. Tier 2: ML models to reduce volume before expensive processing
3. Tier 3: Multi-agent LLM investigation (expensive, only for complex cases)
Our tutorial focuses on Tier 3. But production fraud systems are funnels — the LLM agent is the last resort, not the first.

**Key gotchas newcomers hit:**
- `InvalidUpdateError` when parallel nodes update the same state key without a reducer (#1 trap)
- Side effects before `interrupt()` re-execute on resume (duplicate API calls, charges)
- `MemorySaver` is a demo trap — ephemeral, dies on restart. Multiple devs burned deploying prototypes
- State explosion: "Every additional field increases complexity exponentially"
- Recursion limits are emergency stops, not flow control
- Learning curve: ~2-3 weeks dedicated. One dev documented 16 days to confidence.
- Memory: apps consuming 2GB RAM for basic operations due to state management overhead

**What works better WITHOUT LangGraph:** Simple RAG pipelines, single-agent tool use (plain while loop), multi-topic routing (supervisor pattern picks one route, misses multi-topic queries), truly open-ended exploratory agents (graph structure "boxes in" the agent if no edge exists for an unexpected condition).

**Honest community one-liner:** "LangGraph is the right tool when you need durable, interruptible, multi-step agent workflows in production. For everything else, it is overhead."

**Fraud investigation tools in the real world:**
| Category | Examples |
|----------|----------|
| Transaction analysis | Historical lookup, pattern detection, velocity checks |
| Identity/entity | KYC, watchlist screening, PEP/sanctions checks |
| Geolocation | IP geolocation, travel distance, OpenStreetMap |
| Open source intelligence | Web research, adverse media, social media |
| Graph/relationship | Neo4j entity relationships, transaction clusters |
| Risk scoring | ML model scores, rule-based risk flags |
| Policy/compliance | RAG against policy databases, regulatory rules |

**Production companies/systems reviewed:**
- LinkedIn (recruiting agents, HITL, hierarchical multi-agent)
- Uber (code migration, 21K dev hours saved, hybrid deterministic + LLM)
- Replit (code gen, time travel via checkpoints)
- Elastic (security, Search/Analyze/Reflect cycle, 350+ users)
- Exa (web research, context engineering — clean outputs between agents, not intermediate reasoning)
- Qodo (coding agent, uses LangGraph WITHOUT LangChain for control)
- Amazon (compliance, 2B transactions/day, 3-tier funnel)
- SymphonyAI Sensa (Summary/Narrative/Web Research agents for AML)
- Fravity (70+ ready agents, drag-and-drop workflow, 2-3x efficiency gain)
- McKinsey reports 200-2000% productivity gains in KYC/AML with agentic AI

**Sources:** See full research notes at `/tmp/claude-1000/-home-reisenberg/tasks/aa7f04a.output`, `afa6d2a.output`, `afbc7c3.output`

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

| Phase | The fraud system can... | Story beat |
|-------|------------------------|------------|
| 1 | Score an order's risk (single LLM call) | "You don't need LangGraph yet" |
| 2 | Look things up before scoring (tools + first loop) | "Now you do — the loop appears" |
| 3 | Decide what to investigate and when to stop (full ReAct agent) | "THE HINGE — LangGraph earns its keep" |
| 4 | Be visualized, streamed, and guardrailed (infrastructure) | "Why not just a while loop?" |
| 5 | Pause for humans, resume with context (checkpointing + HITL) | "Real workflow patterns" |
| 6 | Delegate to specialist investigators (multi-agent) | "Scale and specialize" |
| 7 | Run in production without falling over (hardening) | "The other 70%" |

---

## Phase Summary

| Phase | Name | Status |
|-------|------|--------|
| 1 | The Setup (You Don't Need LangGraph Yet) | Built |
| 2 | The Loop (Tools + First Agent) | Planned |
| 3 | The Investigator (Full ReAct Agent) | Planned |
| 4 | The Infrastructure (Why Not Just a While Loop?) | Planned |
| 5 | Real Workflows (Checkpointing + Human-in-the-Loop) | Planned |
| 6 | Multi-Agent (When One Agent Isn't Enough) | Planned |
| 7 | Ship It (The Other 70%) | Planned |

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
- **Expected progression:**
  - **Phase 1:** Scores low — no tools, no history. The LLM sees a $60 residential order and approves.
  - **Phases 2-3:** Scores higher — `check_customer_history` tool retrieves the prior fraud flag from a **data store** (database/API), raising the score. The tool retrieves facts.
  - **Phase 4:** Agent also retrieves its **prior investigation reasoning** — not just "flagged 6 months ago" but "I investigated this customer and concluded the flag was justified because the shipping address matched a known reshipping service." This reasoning is stored in a database. LangGraph's Store API provides the injection pattern, but the persistence is yours.
  - **Key distinction:** Memory stores conclusions, not facts. Tools retrieve facts.

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

## Phase 1: The Setup (You Don't Need LangGraph Yet)

**Goal:** Learn StateGraph mechanics. Be honest that this is overpowered for the task.

**Status:** Built. Code at `phases/phase1-first-graph/`.

**Story beat:** "You don't need LangGraph yet."

**Setup cost framing:** "You don't need LangGraph for this. A function call would work. But we're paying the setup cost once — StateGraph, state schemas, node functions, compile/invoke — because Phase 3 adds loops, and loops are where LangGraph earns its keep. If the graph concepts feel heavy for what we're building here, that's the right instinct. Keep going."

**What it teaches:**
- `StateGraph`, `START`, `END`
- State schema with `TypedDict`
- Nodes as functions: receive state, return deltas
- Normal edges (A → B)
- Conditional edges with `Literal` return types
- `graph.compile()` and `graph.invoke()`

**Build:**
```
# Linear variant (graph.py)
START → parse_order → score_fraud → format_result → END

# Conditional variant (graph_conditional.py)
START → parse_order → score_fraud → [route_decision]
                                      ├── score >= 80 → flag_order → END
                                      ├── score >= 50 → review_order → END
                                      └── score < 50  → approve_order → END
```

**Run all 6 cases.** Key observation: Case 4 (conflicting signals) scores ~5/100 because the LLM can't check anything. It sees a 2-year customer with 30 orders and approves despite the warehouse address, IP change, and name mismatch. This failure motivates Phase 2.

**Reader feeling:** "I see how state flows through nodes. But this is just a pipeline — I could do this without LangGraph. Why am I here?"

**Answer:** "Because Case 4 just failed badly. You need tools."

---

## Phase 2: The Loop (Tools + First Agent)

**Goal:** Introduce tools and the loop. This is where LangGraph starts to matter.

**Story beat:** "Now you do — the loop appears."

**What it teaches:**
- `@tool` decorator and tool definitions
- `model.bind_tools(tools)` — the LLM sees what's available and CHOOSES
- `ToolNode` (prebuilt executor) and `tools_condition` (prebuilt router)
- `add_messages` reducer — THE critical teaching moment
- The first loop: LLM → tools → LLM → tools → ... → END
- **The brain/body mental model:** The LLM decides which tools to call. LangGraph manages the loop. Tools bridge to your data.
- **The fundamental shift:** Traditional if-statements route on data values. The LLM routes on data meaning. No developer wrote the rule "check the fraud database when you see a shared warehouse address." The LLM figured that out.

**Critical teaching moment — reducers:** This is where `add_messages` must be explicitly taught. It's the single most common LangGraph confusion ("why did my previous messages disappear?"). Show the wrong way first:
1. Start with `messages: list` — show that returning new messages overwrites the old ones
2. Switch to `messages: Annotated[list, add_messages]` — show that messages now accumulate
3. Explain: "LangGraph doesn't mutate state. Nodes return deltas. Reducers define how deltas merge. This is the mental model for everything that follows."

**Reducer reference (reuse in Phase 3 when evidence is added):**

| Field | Type | Reducer | What happens when a node returns this field |
|-------|------|---------|---------------------------------------------|
| `order` | `dict` | (default) | Overwrite — new value replaces old |
| `messages` | `list` | `add_messages` | Append — new messages added to existing list |
| `evidence` | `list` | `operator.add` | Concatenate — new findings appended, never overwritten |
| `risk_score` | `int` | (default) | Overwrite — only `calculate_risk_score` sets this |
| `investigation_complete` | `bool` | (default) | Overwrite — only `calculate_risk_score` sets this |

**Tools (3, deliberately limited):**
- `check_customer_history(email)` → past orders, fraud flags, account age
- `verify_shipping_address(address)` → residential/commercial/warehouse, geo risk
- `check_payment_pattern(email, amount)` → typical range, velocity, anomaly

**Build:**
```
START → call_llm → [has tool calls?]
                     ├── yes → execute_tools → call_llm (loop back)
                     └── no → END
```

**Run all 6 cases.** Compare to Phase 1: Case 4 should now score higher because the LLM can check things. But the investigation may be shallow — Phase 3 makes it deep.

**Reader feeling:** "The loop is the new thing. The LLM is making decisions about control flow, not just producing output. This is different from a pipeline."

---

## Phase 3: The Investigator (Full ReAct Agent)

**Goal:** Build the full fraud investigator. This is where LangGraph earns its keep.

**Story beat:** "THE HINGE — LangGraph earns its keep."

This is the hinge of the entire tutorial. If this phase doesn't demonstrate genuine value, the thesis fails.

**What it teaches:**
- The ReAct pattern: reason → act → observe → repeat
- Custom `call_llm` node with system prompt (fraud investigation instructions)
- Custom `execute_tools` node (not just ToolNode)
- Custom routing function (`should_continue`)
- Evidence accumulation: `Annotated[list[Evidence], operator.add]` — structured audit trail
- Deterministic termination: `calculate_risk_score` tool sets `investigation_complete` flag
- "The LLM investigates. Python scores. LangGraph manages the loop between them."

**New tools added (5 total + scoring):**
- `search_fraud_database(indicator)` → known fraud patterns
- `calculate_risk_score(evidence)` → deterministic score, sets `investigation_complete`

**Show the while loop version FIRST, then the graph version:**
- While loop: works, but tangles error handling, streaming, state management
- Graph: separates concerns, same logic, inspectable structure
- This is the core argument for LangGraph

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
Left side: a regulator can inspect the weights. Right side: a black box. The LLM generates hypotheses; Python enforces policy.

**"Agent washing" callout:** Show what a fake agent looks like (sequential pipeline with graph syntax — our Phase 1) vs what a real agent does (LLM-driven dynamic investigation — our Phase 3). Most "multi-agent fraud detection" examples online are the former dressed up as the latter. We should call this out explicitly.

**Run all 6 cases. Success criteria (the hinge test):**
- Case 4 requires 3-6 LLM calls with different tools per iteration
- Tool usage differs per case (not same sequence every time)
- Agent changes assessment mid-loop as new evidence arrives
- Case 1 exits quickly (1-2 calls), Case 3 investigates deeply
- `recursion_limit` becomes a relevant concern, not theoretical

If all canonical cases resolve in 1-2 calls, the tutorial's thesis is weak. Redesign the prompt or tools before moving on.

**Reader feeling:** "This is genuinely different from a pipeline. The agent is investigating, not just scoring. But it's also unpredictable — I need guardrails."

---

## Phase 4: The Infrastructure (Why Not Just a While Loop?)

**Goal:** Show the concrete value LangGraph provides over hand-building the loop.

**Story beat:** "Here's what you get."

This is NEW — the old outline buried these across Phases 3 and 7. They deserve their own phase because they're the answer to "why use a framework?"

**What it teaches:**

- **Graph visualization** — `graph.get_graph().draw_mermaid_png()`. See your agent as a diagram. Show it to stakeholders. Debug by looking, not by reading logs. A while loop with if-statements is invisible. A graph is inspectable.

- **Streaming** — `stream_mode="messages"` (token-by-token), `stream_mode="updates"` (node events), `stream_mode="custom"` (app events). In a while loop, you wait. In a graph, you watch. For any UI, this is table stakes.

- **Recursion limits** — `recursion_limit=25`. One line vs hand-building a counter + cleanup logic. LangGraph makes runaway agents a solved problem.

- **Error handling** — `handle_tool_errors=True` catches hallucinated tool calls and feeds errors back to the LLM. Fallback nodes for graceful degradation.

- **Dead-end prevention** — loop counter in state, force-exit after N iterations without scoring. What if the LLM never calls `calculate_risk_score`? A `dead_end` node calls it with whatever evidence exists, flags `"decision": "review"` with a note: "investigation did not converge."

- **Token budget circuit breaker** — kill if cost exceeds threshold. Track cumulative token usage, apply model-specific pricing, interrupt if budget exceeded.

**Concrete comparison:** Take Phase 3's while-loop version. Add streaming, recursion limits, error handling, and visualization. Count the lines. Then show the graph version with the same features. The graph version is shorter and cleaner.

**Reader feeling:** "OK, I see what the framework gives me. These are things I'd have to build myself, and some of them (visualization, streaming) I wouldn't bother building — but they're genuinely useful."

---

## Phase 5: Real Workflows (Checkpointing + Human-in-the-Loop)

**Goal:** Add pause/resume and human oversight. Also covers context injection patterns (absorbs old Phase 4 memory content).

**Story beat:** "Real workflow patterns."

### Checkpointing

- `MemorySaver` (dev) / `PostgresSaver` (production)
- Thread IDs for conversation continuity
- `get_state()`, `get_state_history()`, `update_state()` — time travel within a thread
- Honest framing: "Every workflow engine does this (Temporal, Dapr Workflow). LangGraph's version is well-integrated with its graph model, but architecturally equivalent. It's 'save game,' not 'memory.'"

### Human-in-the-Loop

**Build:**
```
... → assess_risk → [score >= 70?]
                      ├── yes → interrupt("Review this investigation") → human decides
                      │         ├── approved → finalize_report
                      │         └── override → re_investigate (with human guidance)
                      └── no → finalize_report → END
```

- `interrupt()` — dynamic breakpoints
- `Command(resume=value)` — resuming with human input
- Compile-time breakpoints (`interrupt_before`, `interrupt_after`)
- State forking / "What-If" analysis — fork the investigation, try different evidence, compare outcomes. Use `get_state_history()` to find a checkpoint, `update_state()` to inject corrected evidence, re-run from that point. Original investigation preserved, corrected branch runs independently. This is the killer feature for fraud audit workflows.
- Critical gotcha: NO side effects before `interrupt()` — they re-execute on resume (duplicate API calls, charges)

**Production consideration:** "In a real system, the interrupt goes to a queue (email, Slack, dashboard). The graph resumes hours or days later. This is where `PostgresSaver` matters — `MemorySaver` won't survive a restart."

### Context Injection (Honest Memory Section)

- LangGraph does NOT provide long-term memory
- Two approaches side by side:
  1. **Manual:** Query DB in `parse_order`, pass history into state, save findings after `invoke()`. A database column does the job.
  2. **Store API:** `store.put(("customers", email), investigation_id, findings)` / `store.search(("customers", email))`. Same pattern, integrated into graph lifecycle.
- When Store adds value: semantic search, namespace organization, multiple agents sharing context
- When it doesn't: single keyed lookup, simple cases where a DB query is clearer
- `InMemoryStore` dies on restart — dev/testing only
- "Before invoke, load history. After invoke, save findings. LangGraph standardizes the pattern. The persistence is your database."

**Reader feeling:** "HITL is a first-class pattern, not bolted on. State forking is powerful for audit workflows. But memory is my responsibility."

---

## Phase 6: Multi-Agent (When One Agent Isn't Enough)

**Goal:** Split the investigator into specialists. Honest about when this helps and when it doesn't.

**Story beat:** "Scale and specialize."

**Time budget warning:** This phase will likely take 2-3x longer than any previous phase. Multi-agent involves debugging agent-to-agent communication, state transformation at subgraph boundaries, and prompt engineering for the supervisor's routing decisions.

**Open question (tension #3):** We may discover during Phase 3 that single-agent outperforms multi-agent. If so, this phase becomes "here's how it works and here's why we didn't need it." That's a valid outcome.

**What it teaches:**
- Supervisor + specialists pattern
- `create_supervisor()` (prebuilt), then hand-built with subgraphs
- `Command()` API for dynamic routing to specialists
- Heterogeneous models (expensive for reasoning, cheap for parsing)
- Department routing: fraud dept, finance, customer service — routing to the right specialist dynamically based on what the investigation reveals
- State transformation at subgraph boundaries

**Agents:**

| Agent | Role | Model | Why this model |
|-------|------|-------|----------------|
| **Supervisor** | Route to specialists, synthesize | High-reasoning (Claude Sonnet) | Cross-signal synthesis |
| **Address Analyst** | Shipping verification, geo-risk | Cheap + fast (GPT-4o-mini) | String parsing |
| **Payment Analyst** | Payment patterns, velocity | Cheap + fast (GPT-4o-mini) | Numeric anomaly detection |
| **Customer Analyst** | History, fraud flags | Mid-tier (Claude Haiku) | History summarization |

If all agents use the same model, you've built orchestration theatre.

**Unit economics acid test (measure, don't assume):**
1. Clone Phase 3 single-agent as baseline
2. Run all 6 cases with both approaches
3. Compare: accuracy, latency, LLM calls, token usage, cost
4. If single-agent wins, say so

**Parallelism deferred:** Multi-agent + parallel execution is a cognitive explosion. Sequential first. Parallel as optional optimization.

**Reader feeling:** "Multi-agent is for different capabilities, not just org charts. The overhead is real. For this domain, [measured answer] is better."

---

## Phase 7: Ship It (The Other 70%)

**Goal:** Everything between a demo and a system. Slimmer than if infrastructure hadn't been covered in Phase 4.

**Story beat:** "The other 70%."

**What it teaches:**

### Schema Hardening (TypedDict → Pydantic)
- Upgrade to Pydantic models — validation at node boundaries, type coercion, clear schema errors
- Show what happens with malformed evidence (TypedDict: silent corruption; Pydantic: caught at boundary)

### Context Engineering
- What happens when conversation history grows beyond the context window
- `trim_messages` — keep only recent messages
- Summarization node — periodically compress old messages into a summary
- **State compression strategy:** Prune `messages` (verbose) but keep `evidence` (structured, compact) intact. Evidence is the audit trail — never compress it. Messages are the scratchpad — summarize and discard. This is why the two-layer state design pays off.

### Observability
- LangSmith setup — traces, evaluations, datasets
- Cost monitoring (token usage per investigation)
- Latency profiling (which agent/tool is the bottleneck?)

### Evaluation
- Offline: curated fraud cases with known outcomes, run the system, score accuracy
- LLM-as-judge for open-ended assessment quality

### Deployment
- LangGraph Cloud vs self-hosted (FastAPI + PostgresSaver)
- **The Dapr bridge:** Self-hosted (FastAPI + Docker) composes cleanly with Dapr sidecars. LangGraph Cloud's managed runtime doesn't. This influences our recommendation.
- Docker containerization for self-hosted

### The Double-Write Problem (Dapr bridge)
- What happens when the agent writes to Store but the parent system fails before transaction completes?
- Idempotent investigation IDs, status flags, write-ahead pattern
- This is the bridge to the composition guide — mention it here, solve it there

**Reader feeling:** "The agent logic is 30% of the effort. Hardening, monitoring, and evaluation are the other 70%."

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
| Phase 5 (checkpointing + context) | Phase 12 (actors) | Stateful entities, different mechanism |
| Phase 5 (human-in-the-loop) | Phase 7 (manual AI review) | Human oversight of AI, different UX |
| Phase 7 (deployment) | Phase 11 (resilience) | Hardening, different layer |

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
