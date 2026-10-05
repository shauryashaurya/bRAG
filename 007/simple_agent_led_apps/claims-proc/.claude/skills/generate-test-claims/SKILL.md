---
name: generate-test-claims
description: Generates synthetic insurance claim documents to test the pipeline.
allowed-tools:
    - Write
---

# Generate Synthetic Claims

When asked to generate test claims:

1. Create a folder called `test_data/`
2. Write 3 separate text files inside representing raw claims:
    - `claim_1.txt`: A valid, low-value claim (under $10k) with clear incident details.
    - `claim_2.txt`: A high-value claim over $10,000.
    - `claim_3.txt`: A suspicious claim missing an incident date.
3. Include PII (names, emails) in the text to test the tokenization boundary.
