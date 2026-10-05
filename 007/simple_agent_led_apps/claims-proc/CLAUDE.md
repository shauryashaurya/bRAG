# Claims Processing System

## Architecture

- `extraction/` — JSON schema extraction from claim documents
- `assessment/` — Rubric-based claim quality evaluation
- `routing/` — Confidence-based routing to auto-approve/review/escalate
- `db/` — SQLite persistence for claims, decisions, audit trail

## Conventions

- All model outputs must conform to JSON schemas
- Validation-retry loop: max 3 attempts before escalation
- Log every claim decision with full reasoning
- PII must be tokenized before any model call
- USE ONLY ASCII CHARACTERS IN CODE
- ADD COMMENTS ACROSS THE CODEBASE TO EXPLAIN HOW THE APPLICATION WORKS
- ADD COMMENTS ACROSS THE CODEBASE TO EXPLAIN HOW AGENTIC APPS WORK AND HOW THIS WAS DEVELOPED AND ARCHITECTED
- ADD COMMENTS ACROSS THE CODEBASE AGAIN TO CLEARLY EXPLAIN HOW EACH BIT CONNECTS TO CCAR-P AND CCAR-F CERTIFICATION DOMAINS

## Do NOT

- Never auto-approve claims above $10,000 without human review
- Never log raw PII to the database
- DO NOT USE DOC-STRING TYPE MULTILINE COMMENTS - ONLY SINGLE LINE COMMENTS
- NO EMOJIS, NO SUPERFLOUS DECORATION IN THE CODE
