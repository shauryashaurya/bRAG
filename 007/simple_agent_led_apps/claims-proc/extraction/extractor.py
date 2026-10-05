# Extraction stage: turns a (tokenized) claim document into a JSON object that matches CLAIM_SCHEMA.
# How an agentic loop works here: call the model, check its output against a contract (the
# schema), and if the check fails try again, up to a fixed limit. When the limit is reached the
# loop stops and hands the case to a human instead of guessing. The model proposes, code verifies.
# How it was developed: the schema and loop came from the scaffold; helpers were stubbed in
# common/stubs.py. The PII boundary is applied by the caller (main.process_claim) before this runs.
# Known gap: on retry the validation error is not sent back to the model yet.
# Cert: CCAR-F structured output via tool use and JSON schema; CCAR-P bounded retries and escalation.
from common.stubs import escalate_to_human, validate_schema

# The contract for model output. Required fields make a claim without an incident date invalid.
CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claimant_name": {"type": "string"},
        "policy_number": {"type": "string"},
        "incident_date": {"type": "string", "format": "date"},
        "claim_amount": {"type": "number", "minimum": 0},
        "description": {"type": "string"},
        "supporting_docs": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["claimant_name", "policy_number", "incident_date", "claim_amount"],
}


# Returns a valid claim dict, or a Decision (escalation) if 3 attempts all fail validation.
# The document must already be tokenized by the caller.
def extract_claim(document, client):
    # Max 3 attempts, per the CLAUDE.md validation-retry convention.
    for attempt in range(3):
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[
                {"role": "user", "content": f"Extract claim details:\n{document}"}],
            # Declaring the schema as a tool and forcing tool_choice makes the model answer with
            # structured arguments instead of free text.
            tools=[{"name": "extract_claim", "input_schema": CLAIM_SCHEMA}],
            tool_choice={"type": "tool", "name": "extract_claim"},
        )
        # With a forced tool call the structured result is in the tool-use block's input.
        result = response.content[0].input
        # Never trust model output: validate it in code before using it.
        if validate_schema(result, CLAIM_SCHEMA):
            return result
        # Retry with error feedback logic goes here
    # Retries exhausted: escalate to a human rather than passing bad data downstream.
    return escalate_to_human(document)
