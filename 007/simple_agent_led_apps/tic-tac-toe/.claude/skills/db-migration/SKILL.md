---
name: db-migration
description: Create and apply SQLite migrations for the tic-tac-toe database. Use when the schema needs to change or when adding new tables.
allowed-tools:
  - Read
  - Write
  - Bash(python scripts/migrate.py *)
---

# Database Migration Skill

## Steps
1. Read the current schema from `game/database.py` (look for `SCHEMA_VERSION`)
2. Create a new file `migrations/00X_description.sql` with the DDL
3. Increment `SCHEMA_VERSION` in `game/database.py`
4. Run `python scripts/migrate.py` to apply
5. Verify with `sqlite3 tictactoe.db ".schema"`

## Rules
- Never drop a column without a backup table
- All migrations must be reversible (include `-- DOWN` section)
- Test migration against a copy of the database first