# Assessment stage: scores an extracted claim against a rubric and produces a confidence value.
# In a real agentic app this is an LLM-as-judge step: a model grades the claim on each rubric
# criterion and explains itself. Here it is a deterministic mock so results are repeatable;
# the signature assess_claim(claim, client) stays the same when a real model call replaces it.
# Cert: CCAR-P evaluation and quality measurement with rubrics; CCAR-F prompt and output design.
from common.models import Assessment

# The criteria the assessor must score, each one a question with a 0.0 to 1.0 answer.
CLAIM_RUBRIC = {
    "accuracy": "Does the assessment match the documented evidence?",
    "completeness": "Are all required fields populated correctly?",
    "compliance": "Does the decision comply with policy rules?",
    "fairness": "Is the assessment free from demographic bias?",
}


# Returns an Assessment. Mock logic: fixed scores, with completeness and overall confidence
# reduced for each missing optional field (description, supporting_docs).
def assess_claim(claim, client=None):
    missing = [f for f in ("description", "supporting_docs") if not claim.get(f)]
    scores = {
        "accuracy": 0.95,
        "completeness": 1.0 - 0.1 * len(missing),
        "compliance": 0.95,
        "fairness": 0.95,
    }
    # Confidence is the mean score minus an extra penalty per missing field.
    confidence = round(sum(scores.values()) / len(scores) - 0.05 * len(missing), 4)
    # The reasoning text is stored with the decision so it can be audited later.
    reasoning = "; ".join(f"{k}={v:.2f} ({CLAIM_RUBRIC[k]})" for k, v in scores.items())
    if missing:
        reasoning += f"; missing optional fields: {', '.join(missing)}"
    return Assessment(confidence=confidence, amount=claim["claim_amount"], claim=claim,
                      rubric_scores=scores, reasoning=reasoning)
