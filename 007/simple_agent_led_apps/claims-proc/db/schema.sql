CREATE TABLE claims (
    id INTEGER PRIMARY KEY,
    claim_hash TEXT NOT NULL,
    extracted_data JSON NOT NULL,
    assessment JSON NOT NULL,
    confidence REAL NOT NULL,
    routing_decision TEXT NOT NULL,  -- 'auto_approve', 'review', 'escalate'
    created_at TEXT NOT NULL
);

CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY,
    claim_id INTEGER REFERENCES claims(id),
    action TEXT NOT NULL,
    agent TEXT NOT NULL,
    reasoning TEXT,
    timestamp TEXT NOT NULL
);