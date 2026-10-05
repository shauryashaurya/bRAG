# MCP server exposing read-only tools over the tic-tac-toe SQLite database.
# Run over stdio: python mcp_server.py
# Set TICTACTOE_DB to point at a database other than the default tictactoe.db.
#
# =============================================================================
# MCP IN BRIEF (read this first)
# =============================================================================
# MCP (Model Context Protocol) is a standard way for an AI client (Claude Code,
# Claude Desktop, an IDE) to talk to a separate program that offers it
# capabilities. The separate program is the "server"; the AI app is the "client".
# The client never imports your code. It starts your program as a child process
# and exchanges JSON messages with it, so the server can be written in any language.
#
# A server can offer three kinds of things (called primitives):
# - Tools: functions the model can decide to call (this file only uses these).
#   Use for actions and queries. The model chooses when to call them.
# - Resources: read-only data addressed by URI, such as a file or a table dump.
#   Use for context the application or user attaches, not the model's choice.
# - Prompts: reusable prompt templates the user can pick, such as a slash command.
#
# The wire format is JSON-RPC 2.0. A session looks like this:
# 1. initialize: client and server swap protocol versions and capabilities. The
#    server also sends its name and its "instructions" text here (see create_server).
# 2. tools/list: client asks what tools exist. The server replies with each tool's
#    name, description and JSON Schema for its arguments. The model sees only this.
# 3. tools/call: the model picks a tool and arguments; the client sends them; the
#    server runs the function and returns the result (or an error result).
# 4. The client closes the pipe when the session ends and the process exits.
# The SDK handles all of the JSON-RPC plumbing. Our job is to write plain Python
# functions and describe them well.
#
# Transports (how the messages travel):
# - stdio: client launches "python mcp_server.py" and talks over stdin/stdout.
#   Simple, local, one client per process. This is what we use.
# - Streamable HTTP: server runs as a web service; many clients can connect over a
#   URL. Use this for shared or remote servers, and add authentication.
#
# IMPORTANT stdio rule: stdout IS the protocol channel. Never print() to stdout in
# a stdio server; it corrupts the message stream. Log to stderr or to a file.
#
# How Claude Code finds this server: .mcp.json in the project root maps a server
# name to a launch command. Tools then appear to the model as
# mcp__<server name>__<tool name>, e.g. mcp__tictactoe-db__get_player_stats.
# =============================================================================

import os
from typing import Annotated, Any

# MCPServer is the SDK class that owns the tool registry and runs the protocol loop.
from mcp.server.mcpserver import MCPServer

# ToolError is the "expected failure" exception: raise it and the SDK sends the
# message back to the model as an error result instead of crashing the server.
from mcp.server.mcpserver.exceptions import ToolError

# ToolAnnotations are optional hints about a tool's behavior (see READ_ONLY below).
from mcp.types import ToolAnnotations

# pydantic's Field attaches a human-readable description to a parameter.
from pydantic import Field

from game import database as db
from game.board import DRAW, O, X, board_from_moves, game_result

# Upper bound on one request. Tool output goes into the model's context window, so
# unbounded results waste tokens and can overflow it. Always cap list-style tools.
MAX_HISTORY = 100

# Annotations are hints for clients, not enforcement. Clients may use them to skip
# a permission prompt for safe tools or to warn before dangerous ones.
# - read_only_hint=True: the tool does not modify anything.
# - open_world_hint=False: the tool only touches our own database, not the internet
#   or other external systems.
# Other hints you may need in a larger server: destructive_hint (may delete data),
# idempotent_hint (repeating the call changes nothing more).
# The real read-only guarantee comes from db.connect_readonly, not from this hint.
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


# Helpers below turn database rows into plain dicts. Tool results must be
# JSON-serializable, so return dicts, lists, strings, numbers, booleans and None.
# Never return a sqlite3.Row or a custom object.

# Gives the model a readable label next to the raw winner value.
def _outcome(winner: str | None) -> str:
    if winner is None:
        return "unfinished"
    return DRAW if winner == DRAW else f"{winner} won"


# Stable, explicit key names are part of your tool's contract with the model.
# Choose them deliberately; renaming a key later changes what the model sees.
def _game_dict(game: Any) -> dict[str, Any]:
    return {
        "game_id": game["id"],
        "created_at": game["created_at"],
        "winner": game["winner"],
        "outcome": _outcome(game["winner"]),
        "total_moves": game["total_moves"],
    }


# Shared lookup that turns "row not found" into a clean tool error.
def _require_game(conn: Any, game_id: int) -> Any:
    game = db.get_game(conn, game_id)
    if game is None:
        # Write error messages for the model: say what was wrong and, where
        # possible, how to fix it. The model reads this and can retry correctly.
        raise ToolError(f"no game with id {game_id}")
    return game


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================
# Design choice: the real logic lives in ordinary module-level functions that take
# the database path explicitly. The MCP wrappers inside create_server are thin
# one-line adapters. Benefits:
# - Tests can call these functions directly with a temp database (see
#   tests/test_mcp_server.py) without starting a server or speaking JSON-RPC.
# - The logic stays reusable outside MCP.
# - Nothing relies on global state; the database path is passed in.
#
# Validate every argument yourself. The model fills in arguments and can send
# out-of-range or nonsensical values. The schema checks types only.

def query_game_history(db_path: str, limit: int = 10) -> list[dict[str, Any]]:
    # Enforce the cap server-side; the schema description alone is only advice.
    if not 1 <= limit <= MAX_HISTORY:
        raise ToolError(f"limit must be between 1 and {MAX_HISTORY}")
    # connect_readonly opens the file with mode=ro, so even a bug here cannot write.
    with db.connect_readonly(db_path) as conn:
        return [_game_dict(g) for g in db.list_games(conn, limit)]


def get_move_sequence(db_path: str, game_id: int) -> dict[str, Any]:
    with db.connect_readonly(db_path) as conn:
        game = _require_game(conn, game_id)
        moves = db.get_moves(conn, game_id)
    # Rows are converted to dicts after the connection closes; a Row is only
    # valid while its connection is open, so copy what you need inside the block
    # or, as here, finish all reads before leaving it.
    return {
        **_game_dict(game),
        "moves": [
            {"move_number": m["move_number"], "player": m["player"],
                "row": m["row"], "col": m["col"]}
            for m in moves
        ],
    }


# Shows a tool that combines database data with domain logic from game/board.py.
# MCP tools can do any computation, not just fetch rows.
def get_board_at_move(db_path: str, game_id: int, move_number: int) -> dict[str, Any]:
    with db.connect_readonly(db_path) as conn:
        _require_game(conn, game_id)
        moves = db.get_moves(conn, game_id)
    # Range check depends on data (how many moves this game has), so it cannot be
    # expressed in the schema. Include the valid range in the message.
    if not 0 <= move_number <= len(moves):
        raise ToolError(
            f"move_number must be between 0 and {len(moves)} for game {game_id}")
    sequence = [(m["row"], m["col"], m["player"]) for m in moves]
    board = board_from_moves(sequence, move_number)
    return {
        "game_id": game_id,
        "move_number": move_number,
        "total_moves": len(moves),
        # Structured form for programmatic use.
        "board": [[cell or "." for cell in row] for row in board],
        # Pre-rendered text form; models read a small ASCII grid easily, and it
        # saves them from reassembling the board from nested lists.
        "rendered": "\n".join(" ".join(cell or "." for cell in row) for row in board),
        "result": game_result(board),
    }


def get_player_stats(db_path: str, player: str) -> dict[str, Any]:
    # Normalize forgiving input ("x", " X ") instead of rejecting harmless variants.
    player = player.strip().upper()
    if player not in (X, O):
        # The error also teaches the model a fact about the data (no player names).
        raise ToolError(
            'player must be "X" or "O"; games are hot-seat with no player names')
    with db.connect_readonly(db_path) as conn:
        stats = db.player_stats(conn, player)
    finished = stats["wins"] + stats["losses"] + stats["draws"]
    return {
        "player": player,
        "games": stats["games"],
        "wins": stats["wins"],
        "losses": stats["losses"],
        "draws": stats["draws"],
        "unfinished": stats["unfinished"],
        # None (JSON null) when there is nothing to divide by, rather than 0 or an
        # error: "no data" and "0 percent" mean different things.
        "win_rate": round(stats["wins"] / finished, 3) if finished else None,
    }


# =============================================================================
# SERVER CONSTRUCTION AND TOOL REGISTRATION
# =============================================================================
# A factory function builds the server so tests and main() can create one bound to
# any database path. The tool wrappers below are closures: they capture db_path
# from this function's argument, which avoids a module-level global.
def create_server(db_path: str) -> MCPServer:
    server = MCPServer(
        # Server name reported during initialize. It identifies the server in
        # client logs and UIs; it does not have to match the key in .mcp.json.
        "tictactoe-db",
        # Sent to the client at initialize, and Claude Code adds it to the model's
        # context. Use it for facts that apply across all tools: domain rules,
        # conventions (0-based indices), and how to interpret values. Keep it short;
        # put per-tool details in each tool's description instead.
        instructions=(
            "Read-only access to recorded tic-tac-toe games. Players are X and O; "
            "X always moves first. Rows and columns are 0-based. "
            'A winner of "draw" means a draw; a null winner means the game was not finished.'
        ),
    )

    # @server.tool registers a function as a tool. The SDK builds the tool's JSON
    # Schema automatically from the function signature:
    # - Parameter names become argument names.
    # - Type hints become schema types (int, str, bool, lists, and so on).
    # - A default value makes the argument optional; no default makes it required.
    # - Annotated[type, Field(description=...)] adds a per-argument description.
    # - The return annotation describes the result shape.
    # The SDK also validates incoming arguments against that schema before calling
    # the function, so the function only ever sees correctly typed values.
    #
    # Two pieces of text matter most for tool quality, because they are all the
    # model sees when deciding whether and how to call a tool:
    # - description: say what it does AND when to use it, in plain language.
    # - parameter descriptions: say units, ranges, and where valid values come from.
    #
    # The decorated function names start with an underscore so they do not shadow
    # the module-level implementations of the same name defined above. The name
    # exposed to the model is set explicitly by name=.
    @server.tool(
        name="query_game_history",
        description="List the most recent games, newest first, with winner, outcome and move count.",
        annotations=READ_ONLY,
    )
    def _query_game_history(
        # The f-string keeps the description in sync with MAX_HISTORY.
        limit: Annotated[int, Field(
            description=f"Number of games to return (1-{MAX_HISTORY}).")] = 10,
    ) -> list[dict[str, Any]]:
        return query_game_history(db_path, limit)

    @server.tool(
        name="get_move_sequence",
        description="Return one game's summary and its ordered moves (move_number, player, row, col).",
        annotations=READ_ONLY,
    )
    def _get_move_sequence(
        # Pointing at the tool that produces valid ids lets the model chain calls:
        # query_game_history first, then this one.
        game_id: Annotated[int, Field(description="Game id from query_game_history.")],
    ) -> dict[str, Any]:
        return get_move_sequence(db_path, game_id)

    @server.tool(
        name="get_board_at_move",
        description="Replay a game to a given move and return the board position at that point.",
        annotations=READ_ONLY,
    )
    def _get_board_at_move(
        game_id: Annotated[int, Field(description="Game id from query_game_history.")],
        move_number: Annotated[int, Field(description="Moves to apply; 0 is the empty board.")],
    ) -> dict[str, Any]:
        return get_board_at_move(db_path, game_id, move_number)

    @server.tool(
        name="get_player_stats",
        description="Return the win/loss/draw record and win rate for player X or O across all games.",
        annotations=READ_ONLY,
    )
    def _get_player_stats(
        player: Annotated[str, Field(description='"X" or "O".')],
    ) -> dict[str, Any]:
        return get_player_stats(db_path, player)

    return server


# =============================================================================
# ENTRY POINT
# =============================================================================
def main() -> None:
    # Configuration comes from the environment, so .mcp.json can set it without
    # code changes: "env": {"TICTACTOE_DB": "path"} inside the server entry.
    db_path = os.environ.get("TICTACTOE_DB", db.DEFAULT_DB_PATH)
    # Make sure the schema exists so tools never hit a missing table on first run.
    db.init_db(db_path)
    # run("stdio") starts the protocol loop: it reads requests from stdin, calls the
    # registered tools, and writes responses to stdout until the client disconnects.
    # This call blocks. Pass a different transport name to serve over HTTP instead.
    create_server(db_path).run("stdio")


# Ensures the server starts only when run as a script, not when tests import this
# module to call the implementation functions.
if __name__ == "__main__":
    main()


# =============================================================================
# CHECKLIST FOR BUILDING YOUR OWN MCP SERVER
# =============================================================================
# 1. Decide what the model needs: a few focused tools beat many overlapping ones.
#    Name tools as verbs for one job each (get_x, list_x, create_x).
# 2. Write the logic as plain, testable functions with explicit inputs.
# 3. Wrap each as a tool with a clear description, typed parameters, and
#    parameter descriptions. Set annotations honestly (read-only, destructive).
# 4. Validate inputs and cap result sizes yourself; raise ToolError with messages
#    that tell the model how to correct the call.
# 5. Enforce safety in the code, not in prompts: a read-only database connection,
#    allow-listed paths, no shell or SQL built from model-provided text.
# 6. Return small, structured JSON. Add convenience fields (like "rendered") when
#    they save the model work.
# 7. Never write to stdout in a stdio server. Use a log file or stderr.
# 8. Register the server in .mcp.json, restart the client, and check that the
#    tools appear (the /mcp command in Claude Code lists connected servers).
# 9. Write tests that call the implementation functions directly, plus one test
#    that starts the server and lists its tools.
# 10. For write operations, return what changed, require explicit arguments, and
#     consider making destructive tools ask the user for confirmation.
