# Entry point: wires the four stages into one pipeline, process_claim().
# Data flow: raw document -> tokenize PII -> extract (schema + retries) -> assess (rubric)
#            -> route (auto-approve / review / escalate) -> log decision with reasoning.
# What makes this an agentic-style application: a model proposes structured output, and
# ordinary code validates, bounds, routes and records it. Humans are the fallback whenever
# the model output cannot be trusted. The model never gets the final say on money or on PII.
# How it was developed: CLAUDE.md fixed the architecture and rules; the stage modules were
# scaffolded first, missing helpers were stubbed (common/), then this wiring was added and
# verified with a fake client that can force failures. Real components can be swapped in later.
# Cert: CCAR-P end-to-end production architecture; CCAR-F tool use, structured output, retries.
# Note: the CCAR-P / CCAR-F domain labels in these comments are best-effort and should be
# checked against the official exam guides.
import dataclasses
import hashlib
import sys
from pathlib import Path

from assessment.evaluator import assess_claim
from common.fake_client import FakeClient
from common.models import Decision
from db.store import get_default_conn, log_decision
from extraction.extractor import extract_claim
from extraction.pii_boundary import tokenize_pii
from routing.router import route_claim

# Same limit as in the router and in auto_approve; checked again here as a final safety net.
AUTO_APPROVE_LIMIT = 10000


# Runs one claim through the whole pipeline and returns the Decision.
# client defaults to the offline fake; conn defaults to a shared in-memory database.
def process_claim(document, client=None, conn=None):
    client = client or FakeClient()
    conn = conn or get_default_conn()

    # Step 1: PII boundary. After this line nothing raw is used, sent to a model or stored.
    safe_doc = tokenize_pii(document)
    # A short hash of the tokenized text links database rows without exposing any PII.
    claim_ref = hashlib.sha256(safe_doc.encode()).hexdigest()[:12]

    # Step 2: extraction with up to 3 validated attempts.
    claim = extract_claim(safe_doc, client)
    # A Decision here means extraction gave up and escalated; log it and stop.
    if isinstance(claim, Decision):
        decision = dataclasses.replace(claim, claim_ref=claim_ref)
        log_decision(conn, claim_ref, decision, decision.reason)
        return decision

    # Step 3: rubric assessment produces a confidence value and written reasoning.
    assessment = assess_claim(claim, client)
    # Step 4: deterministic routing based on confidence and amount.
    decision = route_claim(assessment)
    # Step 5: defense in depth. Even if routing is changed or buggy, never auto-approve $10,000+.
    if decision.action == "auto_approve" and assessment.amount >= AUTO_APPROVE_LIMIT:
        decision = Decision("review", "auto-approve blocked: amount >= $10,000",
                            confidence=assessment.confidence, amount=assessment.amount)
    decision = dataclasses.replace(decision, claim_ref=claim_ref)
    # Step 6: persist the decision with its full reasoning for the audit trail.
    log_decision(conn, claim_ref, decision, assessment.reasoning, claim=claim)
    return decision


# Demo runner: processes every .txt file in a folder (default test_data) and prints the outcome.
if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "test_data")
    for path in sorted(folder.glob("*.txt")):
        d = process_claim(path.read_text())
        print(f"{path.name} -> {d.action} ({d.reason})")
