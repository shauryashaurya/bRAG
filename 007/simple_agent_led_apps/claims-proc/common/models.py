# Shared data shapes passed between pipeline stages.
# Role in an agentic app: stages exchange small typed records instead of free text, so each
# stage can be tested alone and the audit trail can store exactly what was decided.
# How it was developed: the router and logger already expected objects with attributes
# (confidence, amount), so these dataclasses were derived from how the scaffold used them.
# Cert: CCAR-F structured outputs and typed contracts; CCAR-P composing stages with clear interfaces.
from dataclasses import dataclass, field
from typing import Optional


# Output of the assessment stage; this is the input of the routing stage.
@dataclass
class Assessment:
    # How sure the assessor is, from 0.0 to 1.0; drives the routing thresholds.
    confidence: float
    # Claim amount in dollars; routing uses it for the $10,000 human-review rule.
    amount: float
    # The tokenized, schema-valid claim that was assessed (never raw PII).
    claim: dict
    # One score per rubric criterion in CLAIM_RUBRIC.
    rubric_scores: dict = field(default_factory=dict)
    # Human-readable explanation; stored in the database for every decision.
    reasoning: str = ""


# Final outcome of processing one claim; this is what gets logged and returned.
@dataclass
class Decision:
    # One of: auto_approve, review, escalate.
    action: str
    reason: str
    # Hash of the tokenized document, used to link database rows without storing raw PII.
    claim_ref: Optional[str] = None
    confidence: Optional[float] = None
    amount: Optional[float] = None
