# Mocked helper functions so the pipeline runs end to end without external services.
# How it was developed: the scaffold called these names but never defined them, so they were
# written stub-first with the exact call signatures the scaffold used. Each one can later be
# replaced by a real implementation (a PII vault, a ticketing system) without touching callers.
# Cert: CCAR-P production architecture (swappable components, stub-first delivery);
# CCAR-F tool and helper design with narrow, single-purpose functions.
import re

from common.models import Assessment, Decision

# Hard business limit from CLAUDE.md: claims at or above this need a human.
AUTO_APPROVE_LIMIT = 10000

# One regex per PII kind. Names are only matched when they follow a label such as "Claimant: ".
_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "name": re.compile(r"(?<=Claimant: )[A-Z][a-z]+(?: [A-Z][a-z]+)+|(?<=Name: )[A-Z][a-z]+(?: [A-Z][a-z]+)+"),
}


# Replaces every PII match of the given kind with a placeholder token and returns the text.
# Cert: CCAR-P safety and compliance (PII must not reach the model).
def replace_with_token(text, kind, token):
    return _PATTERNS[kind].sub(token, text)


# Maps JSON schema type names to Python types for the fallback validator.
_TYPES = {"string": str, "number": (int, float), "array": list, "object": dict}


# Returns True when obj conforms to the schema. Uses the jsonschema library when installed,
# otherwise a small built-in checker covering required, type, minimum and date format.
# Cert: CCAR-F structured output (every model output must conform to a JSON schema).
def validate_schema(obj, schema):
    try:
        import jsonschema
        try:
            jsonschema.validate(obj, schema)
            return True
        except jsonschema.ValidationError:
            return False
    except ImportError:
        pass

    # Fallback path below: only runs when jsonschema is not installed.
    if not isinstance(obj, dict):
        return False
    if any(key not in obj for key in schema.get("required", [])):
        return False
    for key, spec in schema.get("properties", {}).items():
        if key not in obj:
            continue
        value = obj[key]
        expected = _TYPES[spec["type"]]
        # bool is a subclass of int in Python, so it is rejected explicitly for numbers.
        if isinstance(value, bool) or not isinstance(value, expected):
            return False
        if "minimum" in spec and value < spec["minimum"]:
            return False
        if spec.get("format") == "date" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return False
    return True


# Hands a claim to a person. Called from two places: extraction (after 3 failed attempts,
# item is the document) and routing (low confidence, item is an Assessment).
# The returned Decision never echoes the document, so no PII can leak into the log.
# Cert: CCAR-P human oversight and escalation design; CCAR-F error handling in agent loops.
def escalate_to_human(item):
    if isinstance(item, Assessment):
        return Decision("escalate", "confidence too low for review queue",
                        confidence=item.confidence, amount=item.amount)
    return Decision("escalate", "extraction failed schema validation after 3 attempts")


# Approves a claim automatically. The assert is a deterministic guard that holds even if the
# router has a bug: an AI-driven or buggy path must never auto-approve $10,000 or more.
# Cert: CCAR-P guardrails and deterministic enforcement of business rules.
def auto_approve(assessment):
    assert assessment.amount < AUTO_APPROVE_LIMIT, "auto-approve above $10,000 is forbidden"
    return Decision("auto_approve", "high confidence and amount under $10,000",
                    confidence=assessment.confidence, amount=assessment.amount)


# Puts a claim in the human review queue (mocked as a Decision record).
# Cert: CCAR-P human-in-the-loop routing.
def queue_for_review(assessment):
    return Decision("review", "needs human review (moderate confidence or amount >= $10,000)",
                    confidence=assessment.confidence, amount=assessment.amount)
