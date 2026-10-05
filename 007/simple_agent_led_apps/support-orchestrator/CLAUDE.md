# Customer Support Orchestrator

## Architecture

- `coordinator.py` — intent classification and routing
- `agents/` — specialist agent definitions
- `mcp_servers/` — MCP servers for CRM, Orders, Knowledge, Payments
- `traces/` — observability and tracing

## Conventions

- Coordinator uses Haiku for intent classification
- Specialist agents use Sonnet for resolution
- Every tool call is traced with input, output, and latency
- Escalation threshold: refund > $500, or confidence < 0.7

## MCP Servers

- CRM: query_customer, update_customer, get_history
- Orders: get_order, check_status, create_return
- Knowledge: search_articles, get_article
- Payments: process_refund, check_payment_status

## Do NOT

- Never process a refund above $500 without human approval
- Never share customer PII across agents without tokenization
