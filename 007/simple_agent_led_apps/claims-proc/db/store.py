# SQLite persistence for claims, decisions and the audit trail.
# Role in an agentic app: the database is the system of record. Every decision is stored with
# its reasoning so a human or auditor can reconstruct why the system acted as it did.
# Hard rule from CLAUDE.md: only tokenized data is ever written here, never raw PII.
# How it was developed: kept minimal (in-memory by default) so the pipeline can be tested
# quickly; swap the path for a file to persist across runs.
# Cert: CCAR-P auditability, observability and compliance; CCAR-F state outside the model.
import json
import sqlite3
from datetime import datetime, timezone

# Lazily created shared connection used when the caller does not pass one.
_default_conn = None

# One statement per tuple entry (single-line strings, joined when the schema is applied).
SCHEMA = (
    "CREATE TABLE IF NOT EXISTS claims (claim_ref TEXT PRIMARY KEY, tokenized_claim TEXT);",
    "CREATE TABLE IF NOT EXISTS decisions (id INTEGER PRIMARY KEY AUTOINCREMENT, claim_ref TEXT, "
    "action TEXT, reason TEXT, confidence REAL, amount REAL, reasoning TEXT, created_at TEXT);",
    "CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT, claim_ref TEXT, "
    "event TEXT, detail TEXT, created_at TEXT);",
)


# Opens a database and makes sure the three tables exist.
def connect(path=":memory:"):
    conn = sqlite3.connect(path)
    conn.executescript("\n".join(SCHEMA))
    return conn


# Returns the shared default connection, creating it on first use.
def get_default_conn():
    global _default_conn
    if _default_conn is None:
        _default_conn = connect()
    return _default_conn


# Writes one decision plus an audit entry. Callers must pass tokenized data only.
# The decisions table holds the full reasoning; audit_log holds a short event line.
# Cert: CCAR-P decision logging with full reasoning (CLAUDE.md convention).
def log_decision(conn, claim_ref, decision, reasoning, claim=None):
    now = datetime.now(timezone.utc).isoformat()
    # The claim record is optional because failed extractions have no valid claim to store.
    if claim is not None:
        conn.execute("INSERT OR REPLACE INTO claims VALUES (?, ?)", (claim_ref, json.dumps(claim)))
    conn.execute(
        "INSERT INTO decisions (claim_ref, action, reason, confidence, amount, reasoning, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (claim_ref, decision.action, decision.reason, decision.confidence, decision.amount, reasoning, now),
    )
    conn.execute(
        "INSERT INTO audit_log (claim_ref, event, detail, created_at) VALUES (?, ?, ?, ?)",
        (claim_ref, "decision", f"{decision.action}: {decision.reason}", now),
    )
    conn.commit()
