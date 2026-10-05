# Offline stand-in for the Anthropic client so the pipeline runs without an API key.
# How it works: it exposes client.messages.create(...) like the real SDK and returns an object
# shaped like a tool-use response (response.content[0].input is a dict). It parses the already
# tokenized document with regexes instead of calling a model.
# Why it exists: agentic apps are hard to test when the model is the only source of output.
# A fake client lets us force failures (fail_first_n, always_invalid) to prove that the
# validation-retry loop and the escalation path behave correctly.
# Cert: CCAR-P testing and reliability of agent loops; CCAR-F tool-use response shape.
import re
from types import SimpleNamespace

# Regex per simple text field; the amount and docs list are parsed separately below.
_FIELDS = {
    "claimant_name": r"Claimant:\s*(.+)",
    "policy_number": r"Policy Number:\s*(\S+)",
    "incident_date": r"Incident Date:\s*(\d{4}-\d{2}-\d{2})",
    "description": r"Description:\s*(.+)",
}


class FakeClient:
    # fail_first_n: return an invalid (empty) result for the first N calls to exercise retries.
    # always_invalid: never return a valid result, to exercise escalation after 3 attempts.
    def __init__(self, fail_first_n=0, always_invalid=False):
        self.fail_first_n = fail_first_n
        self.always_invalid = always_invalid
        # Test hooks: how many times the model was called and what prompts it received.
        self.calls = 0
        self.prompts = []
        # Mirrors the real SDK call path: client.messages.create(...).
        self.messages = SimpleNamespace(create=self._create)

    # Mimics messages.create; only the user prompt is inspected.
    def _create(self, **kwargs):
        self.calls += 1
        prompt = kwargs["messages"][0]["content"]
        self.prompts.append(prompt)
        if self.always_invalid or self.calls <= self.fail_first_n:
            return self._response({})
        return self._response(self._parse(prompt))

    # Pulls claim fields out of the text; fields that are absent are simply left out, which
    # is what makes the schema validator reject a claim with a missing required field.
    @staticmethod
    def _parse(text):
        result = {}
        for key, pattern in _FIELDS.items():
            m = re.search(pattern, text)
            if m:
                result[key] = m.group(1).strip()
        m = re.search(r"Claim Amount:\s*\$?([\d,]+(?:\.\d+)?)", text)
        if m:
            result["claim_amount"] = float(m.group(1).replace(",", ""))
        m = re.search(r"Supporting Docs:\s*(.+)", text)
        if m:
            result["supporting_docs"] = [d.strip() for d in m.group(1).split(",")]
        return result

    # Wraps a dict the way the real SDK wraps a tool-use block: content[0].input.
    @staticmethod
    def _response(data):
        return SimpleNamespace(content=[SimpleNamespace(input=data)])
