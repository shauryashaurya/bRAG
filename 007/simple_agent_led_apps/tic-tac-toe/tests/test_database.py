import sqlite3
from pathlib import Path

import pytest

from game import database as db
from game.board import X, O, board_from_moves, game_result


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    path = str(tmp_path / "test.db")
    db.init_db(path)
    return path


def test_migrations_are_idempotent(db_path: str) -> None:
    assert db.init_db(db_path) == db.SCHEMA_VERSION
    with db.connect(db_path) as conn:
        assert db.current_version(conn) == db.SCHEMA_VERSION
        tables = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert {"games", "moves"} <= tables


def test_pragmas(db_path: str) -> None:
    with db.connect(db_path) as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_foreign_key_enforced(db_path: str) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        with db.connect(db_path) as conn:
            db.record_move(conn, 999, 1, X, 0, 0)


def test_duplicate_move_number_rejected(db_path: str) -> None:
    with db.connect(db_path) as conn:
        game_id = db.create_game(conn)
        db.record_move(conn, game_id, 1, X, 0, 0)
    with pytest.raises(sqlite3.IntegrityError):
        with db.connect(db_path) as conn:
            db.record_move(conn, game_id, 1, O, 1, 1)


def test_round_trip_and_replay(db_path: str) -> None:
    played = [(0, 0, X), (1, 1, O), (0, 1, X), (2, 2, O), (0, 2, X)]
    with db.connect(db_path) as conn:
        game_id = db.create_game(conn)
        for n, (row, col, player) in enumerate(played, start=1):
            db.record_move(conn, game_id, n, player, row, col)
        db.finish_game(conn, game_id, X, len(played))

    with db.connect(db_path) as conn:
        game = db.list_games(conn)[0]
        moves = db.get_moves(conn, game_id)

    assert game["id"] == game_id
    assert game["winner"] == X
    assert game["total_moves"] == 5
    assert [m["move_number"] for m in moves] == [1, 2, 3, 4, 5]
    replay = [(m["row"], m["col"], m["player"]) for m in moves]
    assert replay == played
    assert game_result(board_from_moves(replay, len(replay))) == X


def test_unfinished_game_tracks_move_count(db_path: str) -> None:
    with db.connect(db_path) as conn:
        game_id = db.create_game(conn)
        db.record_move(conn, game_id, 1, X, 2, 2)
        db.record_move(conn, game_id, 2, O, 0, 0)
        game = db.list_games(conn)[0]
    assert game["winner"] is None
    assert game["total_moves"] == 2


def test_list_games_newest_first(db_path: str) -> None:
    with db.connect(db_path) as conn:
        first = db.create_game(conn)
        second = db.create_game(conn)
        ids = [g["id"] for g in db.list_games(conn)]
    assert ids == [second, first]


def test_writes_are_logged(db_path: str) -> None:
    with db.connect(db_path) as conn:
        game_id = db.create_game(conn)
    for handler in db.logger.handlers:
        handler.flush()
    assert f"create_game id={game_id}" in db.LOG_PATH.read_text(encoding="ascii")
