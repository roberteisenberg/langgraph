"""
Fraud investigation state schemas.

Designed to be additive across phases:
- Phase 1: FraudState (order, risk_score, decision)
- Phase 2: adds messages (Annotated[list, add_messages])
- Phase 3: adds evidence (Annotated[list[Evidence], operator.add]),
           investigation_complete
- Phase 4: adds customer_history
- Phase 5: adds human_decision, human_notes
"""

from typing import TypedDict


class FraudState(TypedDict):
    """Phase 1 state — minimal fraud scoring."""
    order: dict           # raw order input
    risk_score: int       # 0-100
    decision: str         # "approve" | "flag" | "reject"
