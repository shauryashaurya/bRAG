# Routing stage: decides what happens to an assessed claim (auto-approve, review, escalate).
# Why this is plain code and not a model call: the thresholds are business and compliance rules
# that must behave the same way every time. The model supplies a confidence signal; deterministic
# code makes the final call. This split is a core pattern for reliable agentic systems.
# Thresholds: confidence above 0.9 and amount under $10,000 is auto-approved; confidence above
# 0.7 goes to a human review queue; everything else is escalated.
# Cert: CCAR-P human oversight, guardrails and escalation design; CCAR-F routing by confidence.
from common.stubs import auto_approve, escalate_to_human, queue_for_review


# Returns a Decision for an Assessment (needs .confidence and .amount).
# A high-confidence claim of $10,000 or more skips the first branch and lands in review.
def route_claim(assessment):
    if assessment.confidence > 0.9 and assessment.amount < 10000:
        return auto_approve(assessment)
    elif assessment.confidence > 0.7:
        return queue_for_review(assessment)
    else:
        return escalate_to_human(assessment)
