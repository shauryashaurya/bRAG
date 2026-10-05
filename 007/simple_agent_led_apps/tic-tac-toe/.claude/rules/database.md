---
paths:
  - "game/database.py"
  - "scripts/*.py"
---

# Database Rules

- Always use parameterized queries — never f-string SQL
- Enable WAL mode for concurrent reads: `PRAGMA journal_mode=WAL`
- Every migration must be idempotent (use `CREATE TABLE IF NOT EXISTS`)
- Log all write operations to `logs/db_ops.log`