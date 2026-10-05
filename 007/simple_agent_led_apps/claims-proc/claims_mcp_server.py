# MCP server that exposes the claims pipeline to an AI client such as Claude Code.
# Run over stdio: python claims_mcp_server.py
# Environment: CLAIMS_DB (sqlite path, default claims.db), CLAIMS_DIR (claim files, default test_data).
# How MCP works: the client starts this program as a child process and exchanges JSON-RPC
# messages over stdin and stdout. The client asks which tools exist (tools/list), then the model
# picks one and sends arguments (tools/call). The model sees only each tool's name, description
# and argument schema, so those texts are part of the design. Never print() to stdout here:
# stdout is the protocol channel and any stray text corrupts it.
# Key design decision: PII safety. Anything a tool returns goes into the model's context, and
# CLAUDE.md says PII must be tokenized before any model call. So no tool accepts or returns raw
# claim text. Claims are processed by file name from an allow-listed folder, the pipeline
# tokenizes inside process_claim, and tools return only decisions and tokenized data.
# Second decision: the model can run and inspect claims but cannot approve or override one.
# Human review decisions stay with humans, which is the human-oversight pattern of CLAUDE.md.
# How it was developed: the pipeline functions were written first as plain testable code; this
# file only wraps them, mirroring the structure of the sibling tic-tac-toe MCP server.
# Cert: CCAR-F tool design and MCP integration (clear descriptions, typed schemas, errors that
# guide the model); CCAR-P safe tool boundaries, least privilege and auditability.
import os
from pathlib import Path
from typing import Annotated, Any

# MCPServer owns the tool registry and runs the protocol loop.
from mcp.server.mcpserver import MCPServer

# ToolError is the expected-failure exception: its message is returned to the model as an
# error result so the model can correct the call, instead of crashing the server.
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from assessment.evaluator import CLAIM_RUBRIC
from db.store import connect
from main import process_claim

# Cap on list results: tool output goes into the model's context window, so keep it bounded.
MAX_DECISIONS = 100

# Annotations are hints for clients, not enforcement. open_world_hint=False means the tool only
# touches local files and our own database, never the internet.
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
# This tool writes decision rows but never deletes anything, and re-running adds a new row.
WRITES_DECISION = ToolAnnotations(read_only_hint=False, destructive_hint=False,
                                  idempotent_hint=False, open_world_hint=False)


# Lists the .txt claim files the server is allowed to process. File names only, no contents.
def list_claim_files(claims_dir):
    folder = Path(claims_dir)
    if not folder.is_dir():
        raise ToolError(f"Claims folder not found: {folder.name}")
    return sorted(p.name for p in folder.glob("*.txt"))


# Runs one claim file through the pipeline and returns the decision.
# Validation is done in code, not in prompts: only a bare existing .txt name inside the folder.
def run_claim_file(claims_dir, db_path, file_name):
    folder = Path(claims_dir).resolve()
    allowed = list_claim_files(claims_dir)
    # Checking against the directory listing blocks path traversal such as "..\\secrets.txt".
    if file_name not in allowed:
        raise ToolError(f"Unknown claim file '{file_name}'. Call list_claim_files to see valid names.")
    document = (folder / file_name).read_text()
    conn = connect(db_path)
    try:
        # The pipeline tokenizes PII, extracts, assesses, routes and logs the decision.
        decision = process_claim(document, conn=conn)
    finally:
        conn.close()
    # Decision holds only a hash reference, the action and a reason, never raw PII.
    return {
        "file": file_name,
        "claim_ref": decision.claim_ref,
        "action": decision.action,
        "reason": decision.reason,
        "confidence": decision.confidence,
        "amount": decision.amount,
    }


# Returns the most recent decisions, newest first, with their stored reasoning.
def list_decisions(db_path, limit, action=None):
    if not 1 <= limit <= MAX_DECISIONS:
        raise ToolError(f"limit must be between 1 and {MAX_DECISIONS}")
    if action is not None and action not in ("auto_approve", "review", "escalate"):
        raise ToolError("action must be one of: auto_approve, review, escalate")
    conn = connect(db_path)
    try:
        # Values are bound as parameters; nothing from the model is placed into the SQL text.
        sql = ("SELECT claim_ref, action, reason, confidence, amount, reasoning, created_at "
               "FROM decisions")
        params = []
        if action:
            sql += " WHERE action = ?"
            params.append(action)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        cols = ["claim_ref", "action", "reason", "confidence", "amount", "reasoning", "created_at"]
        return [dict(zip(cols, row)) for row in conn.execute(sql, params)]
    finally:
        conn.close()


# Returns the stored tokenized claim and all decisions for one claim_ref.
def get_claim_record(db_path, claim_ref):
    conn = connect(db_path)
    try:
        row = conn.execute("SELECT tokenized_claim FROM claims WHERE claim_ref = ?",
                           (claim_ref,)).fetchone()
        decisions = list_decisions_for(conn, claim_ref)
    finally:
        conn.close()
    if row is None and not decisions:
        raise ToolError(f"No record for claim_ref '{claim_ref}'. Call list_decisions to find valid refs.")
    # A failed extraction has decisions but no stored claim, so the claim may be None.
    return {"claim_ref": claim_ref, "tokenized_claim": row[0] if row else None,
            "decisions": decisions}


# Helper: all decisions for one claim, oldest first.
def list_decisions_for(conn, claim_ref):
    cols = ["action", "reason", "confidence", "amount", "reasoning", "created_at"]
    rows = conn.execute("SELECT action, reason, confidence, amount, reasoning, created_at "
                        "FROM decisions WHERE claim_ref = ? ORDER BY id", (claim_ref,))
    return [dict(zip(cols, r)) for r in rows]


# Builds the server bound to a claims folder and database. A factory keeps configuration out of
# module globals and lets tests create a server with temporary paths.
def create_server(claims_dir, db_path):
    server = MCPServer(
        "claims-pipeline",
        # Sent to the client at initialize and added to the model's context: facts that apply
        # across all tools. Per-tool detail belongs in each tool description instead.
        instructions=(
            "Claims processing pipeline running in mock mode (offline fake model). "
            "Claims are processed by file name; raw claim text and PII are never returned. "
            "Decisions are auto_approve, review or escalate. Claims of $10,000 or more are never "
            "auto-approved. review and escalate items need a human; you cannot approve them."
        ),
    )

    @server.tool(
        name="list_claim_files",
        description="List the claim file names available to process. Call this first to get valid names.",
        annotations=READ_ONLY,
    )
    def _list_claim_files() -> list[str]:
        return list_claim_files(claims_dir)

    @server.tool(
        name="process_claim_file",
        description=("Run one claim file through the full pipeline (tokenize PII, extract, assess, route) "
                     "and record the decision. Returns the decision, not the claim text. "
                     "Each call adds a new decision row."),
        annotations=WRITES_DECISION,
    )
    def _process_claim_file(
        file_name: Annotated[str, Field(
            description="Bare file name from list_claim_files, for example claim_1.txt.")],
    ) -> dict[str, Any]:
        return run_claim_file(claims_dir, db_path, file_name)

    @server.tool(
        name="list_decisions",
        description="List recent claim decisions, newest first, with reasoning. Optionally filter by action.",
        annotations=READ_ONLY,
    )
    def _list_decisions(
        limit: Annotated[int, Field(
            description=f"Number of decisions to return (1-{MAX_DECISIONS}).")] = 10,
        action: Annotated[str | None, Field(
            description="Optional filter: auto_approve, review or escalate.")] = None,
    ) -> list[dict[str, Any]]:
        return list_decisions(db_path, limit, action)

    @server.tool(
        name="get_claim_record",
        description="Get the tokenized claim and full decision history for one claim_ref.",
        annotations=READ_ONLY,
    )
    def _get_claim_record(
        claim_ref: Annotated[str, Field(
            description="Reference from process_claim_file or list_decisions.")],
    ) -> dict[str, Any]:
        return get_claim_record(db_path, claim_ref)

    @server.tool(
        name="get_rubric",
        description="Return the assessment rubric: the criteria every claim is scored on.",
        annotations=READ_ONLY,
    )
    def _get_rubric() -> dict[str, str]:
        return dict(CLAIM_RUBRIC)

    return server


# Reads configuration from the environment so .mcp.json can set it without code changes.
def main():
    claims_dir = os.environ.get("CLAIMS_DIR", "test_data")
    db_path = os.environ.get("CLAIMS_DB", "claims.db")
    # run("stdio") blocks and serves requests until the client disconnects.
    create_server(claims_dir, db_path).run("stdio")


# Only start serving when run as a script, so tests can import the plain functions above.
if __name__ == "__main__":
    main()
