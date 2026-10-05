# SQLite persistence for games and their move sequences.
# Every move is written as it is played so any game can be replayed later.

import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from game.board import DRAW

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1
DEFAULT_DB_PATH = str(PROJECT_ROOT / "tictactoe.db")
MIGRATIONS_DIR = PROJECT_ROOT / "migrations"
LOG_PATH = PROJECT_ROOT / "logs" / "db_ops.log"

logger = logging.getLogger("tictactoe.db")


def _ensure_log_handler() -> None:
    if logger.handlers:
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_PATH, encoding="ascii")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# Opens a connection with foreign keys and WAL enabled.
# Commits on success, rolls back on error, and always closes.
#
# How @contextmanager works:
# - The function is a generator: it has exactly one yield.
# - Code before the yield is setup; it runs when the "with" block starts.
# - The yielded value is what "as conn" receives in the caller.
# - Code after the yield runs when the "with" block ends normally.
# - If the "with" body raises, the exception is thrown into the generator at the
#   yield line, so the except and finally clauses below run.
#
# Why the return type is Generator and not Iterator:
# - contextmanager needs the generator to accept .throw() and .close(), which
#   Iterator does not promise. Annotating with Iterator is deprecated in the stubs.
# - Generator takes three type arguments: yielded type, sent type, return type.
#   This function yields a connection and neither receives nor returns a value,
#   so the arguments are (sqlite3.Connection, None, None).
# - Python 3.13 allows the short form Generator[sqlite3.Connection]; the full
#   form is used here so it also works on older versions such as 3.11.
@contextmanager
def connect(db_path: str = DEFAULT_DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    _ensure_log_handler()
    conn = sqlite3.connect(db_path)
    # Rows support access by column name, e.g. row["winner"].
    conn.row_factory = sqlite3.Row
    try:
        # SQLite ignores foreign keys unless this is set on every connection.
        conn.execute("PRAGMA foreign_keys = ON")
        # WAL lets the GUI write while the MCP server reads.
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        # Reached only if the with body finished without raising.
        conn.commit()
    except Exception:
        # Undo any partial writes, then let the caller see the original error.
        conn.rollback()
        raise
    finally:
        # Runs on success and on failure, so connections are never leaked.
        conn.close()


# Opens a read-only connection; any write attempt raises sqlite3.OperationalError.
# Used by external readers such as the MCP server so they can never modify game data.
# Same generator pattern as connect(); no commit or rollback is needed because
# nothing can be written.
@contextmanager
def connect_readonly(db_path: str = DEFAULT_DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    # Fail early with a clear message; sqlite3 would otherwise report a vague error.
    if not Path(db_path).is_file():
        raise FileNotFoundError(f"database not found: {db_path}")
    # mode=ro needs the URI form of the path, hence uri=True.
    conn = sqlite3.connect(
        Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    finally:
        conn.close()


# Returns the SQL between "-- UP" and "-- DOWN" in a migration file.
def _up_section(sql: str) -> str:
    up = sql.split("-- DOWN", 1)[0]
    return up.split("-- UP", 1)[-1]


def _migration_files() -> list[tuple[int, Path]]:
    files = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        files.append((int(path.name.split("_", 1)[0]), path))
    return files


def current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT MAX(version) AS v FROM schema_migrations").fetchone()
    return row["v"] or 0


# Applies every migration newer than the recorded version; safe to run repeatedly.
def apply_migrations(conn: sqlite3.Connection) -> int:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    applied = current_version(conn)
    for version, path in _migration_files():
        if version <= applied:
            continue
        conn.executescript(_up_section(path.read_text(encoding="ascii")))
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (version, _now()),
        )
        conn.commit()
        logger.info("applied migration %s", path.name)
    return current_version(conn)


def init_db(db_path: str = DEFAULT_DB_PATH) -> int:
    with connect(db_path) as conn:
        return apply_migrations(conn)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_game(conn: sqlite3.Connection) -> int:
    cur = conn.execute(
        "INSERT INTO games (created_at, winner, total_moves) VALUES (?, NULL, 0)",
        (_now(),),
    )
    game_id = int(cur.lastrowid)
    logger.info("create_game id=%d", game_id)
    return game_id


def record_move(
    conn: sqlite3.Connection,
    game_id: int,
    move_number: int,
    player: str,
    row: int,
    col: int,
) -> None:
    conn.execute(
        "INSERT INTO moves (game_id, move_number, player, row, col) VALUES (?, ?, ?, ?, ?)",
        (game_id, move_number, player, row, col),
    )
    conn.execute(
        "UPDATE games SET total_moves = ? WHERE id = ?",
        (move_number, game_id),
    )
    logger.info(
        "record_move game=%d n=%d player=%s row=%d col=%d",
        game_id, move_number, player, row, col,
    )


# winner is "X", "O" or "draw".
def finish_game(conn: sqlite3.Connection, game_id: int, winner: str, total_moves: int) -> None:
    conn.execute(
        "UPDATE games SET winner = ?, total_moves = ? WHERE id = ?",
        (winner, total_moves, game_id),
    )
    logger.info("finish_game id=%d winner=%s moves=%d",
                game_id, winner, total_moves)


# Newest first. A limit of None returns every game (SQLite treats LIMIT -1 as no limit).
def list_games(conn: sqlite3.Connection, limit: int | None = None) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, created_at, winner, total_moves FROM games ORDER BY id DESC LIMIT ?",
        (-1 if limit is None else limit,),
    ).fetchall()


def get_game(conn: sqlite3.Connection, game_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id, created_at, winner, total_moves FROM games WHERE id = ?",
        (game_id,),
    ).fetchone()


# Win/loss/draw record for "X" or "O". Unfinished games have a NULL winner.
def player_stats(conn: sqlite3.Connection, player: str) -> sqlite3.Row:
    return conn.execute(
        "SELECT COUNT(*) AS games, "
        "COALESCE(SUM(winner = ?), 0) AS wins, "
        "COALESCE(SUM(winner IS NOT NULL AND winner NOT IN (?, ?)), 0) AS losses, "
        "COALESCE(SUM(winner = ?), 0) AS draws, "
        "COALESCE(SUM(winner IS NULL), 0) AS unfinished "
        "FROM games",
        (player, player, DRAW, DRAW),
    ).fetchone()


def get_moves(conn: sqlite3.Connection, game_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT move_number, player, row, col FROM moves "
        "WHERE game_id = ? ORDER BY move_number",
        (game_id,),
    ).fetchall()
