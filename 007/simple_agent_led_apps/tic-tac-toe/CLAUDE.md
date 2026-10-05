# Tic-Tac-Toe Project

## Build Commands

- Run: `python main.py`
- Test: `pytest tests/ -v`
- DB reset: `python scripts/reset_db.py`

## Architecture

- `main.py` — entry point, launches Tkinter window
- `game/board.py` — board state, win detection
- `game/database.py` — SQLite operations
- `ui/app.py` — Tkinter GUI
- `tests/` — pytest unit tests
- `mcp_server.py` — MCP server exposing read-only tools over the tic-tac-toe SQLite database.

## Coding Conventions

- Type hints on all function signatures
- No global mutable state; pass board as parameter
- Database connections via context manager (`with` statement)
- Use `sqlite3.Row` for row factory to enable dict-like access
- NO MULTI-LINE DOCSTRING TYPE COMMENTS, USE MULTIPLE SINGLE LINE COMMENTS INSTEAD
- NO EMOJIS OR SUPERFLUOUS DECORATION IN CODE - STRAIGHTFORWARD, LITERATE, TERSE CODE
- STICK TO ONLY ASCII CHARACTERS IN CODE

## Database Schema

- `games(id INTEGER PK, created_at TEXT, winner TEXT, total_moves INT)`
- `moves(id INTEGER PK, game_id INT FK, move_number INT, player TEXT, row INT, col INT)`
- Always enable foreign keys: `PRAGMA foreign_keys = ON`

## Do NOT

- Do not use `eval()` or `exec()` anywhere
- Do not commit the `.db` file — it is in `.gitignore`
- NO MULTI-LINE DOCSTRING TYPE COMMENTS, USE MULTIPLE SINGLE LINE COMMENTS INSTEAD
- NO EMOJIS OR SUPERFLUOUS DECORATION IN CODE - STRAIGHTFORWARD, LITERATE, TERSE CODE
- STICK TO ONLY ASCII CHARACTERS IN CODE
