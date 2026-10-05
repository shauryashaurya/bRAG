import asyncio
import sqlite3
from pathlib import Path

import pytest

import mcp_server
from mcp.server.mcpserver.exceptions import ToolError
from game import database as db
from game.board import DRAW, O, X

X_WIN = [(0, 0, X), (1, 1, O), (0, 1, X), (2, 2, O), (0, 2, X)]
DRAWN = [
    (0, 0, X), (0, 1, O), (0, 2, X), (1, 1, O), (1, 0, X),
    (1, 2, O), (2, 1, X), (2, 0, O), (2, 2, X),
]
ABANDONED = [(1, 1, X), (0, 0, O)]


def record(db_path: str, moves: list[tuple[int, int, str]], result: str | None) -> int:
    with db.connect(db_path) as conn:
        game_id = db.create_game(conn)
        for n, (row, col, player) in enumerate(moves, start=1):
            db.record_move(conn, game_id, n, player, row, col)
        if result is not None:
            db.finish_game(conn, game_id, result, len(moves))
    return game_id


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    path = str(tmp_path / "mcp.db")
    db.init_db(path)
    record(path, X_WIN, X)
    record(path, DRAWN, DRAW)
    record(path, ABANDONED, None)
    return path


def test_query_game_history(db_path: str) -> None:
    games = mcp_server.query_game_history(db_path, 5)
    assert [g["game_id"] for g in games] == [3, 2, 1]
    assert [g["outcome"] for g in games] == ["unfinished", "draw", "X won"]
    assert mcp_server.query_game_history(db_path, 1)[0]["game_id"] == 3


@pytest.mark.parametrize("limit", [0, mcp_server.MAX_HISTORY + 1])
def test_query_game_history_rejects_bad_limit(db_path: str, limit: int) -> None:
    with pytest.raises(ToolError):
        mcp_server.query_game_history(db_path, limit)


def test_get_move_sequence(db_path: str) -> None:
    result = mcp_server.get_move_sequence(db_path, 1)
    assert result["winner"] == X
    assert [(m["row"], m["col"], m["player"]) for m in result["moves"]] == X_WIN
    assert [m["move_number"] for m in result["moves"]] == [1, 2, 3, 4, 5]


def test_get_move_sequence_unknown_game(db_path: str) -> None:
    with pytest.raises(ToolError):
        mcp_server.get_move_sequence(db_path, 999)


def test_get_board_at_move(db_path: str) -> None:
    start = mcp_server.get_board_at_move(db_path, 1, 0)
    assert start["rendered"] == ". . .\n. . .\n. . ."
    assert start["result"] is None
    final = mcp_server.get_board_at_move(db_path, 1, 5)
    assert final["board"][0] == [X, X, X]
    assert final["result"] == X
    with pytest.raises(ToolError):
        mcp_server.get_board_at_move(db_path, 1, 6)


def test_get_player_stats(db_path: str) -> None:
    x = mcp_server.get_player_stats(db_path, "x")
    assert (x["games"], x["wins"], x["losses"], x["draws"], x["unfinished"]) == (3, 1, 0, 1, 1)
    assert x["win_rate"] == 0.5
    o = mcp_server.get_player_stats(db_path, "O")
    assert (o["wins"], o["losses"], o["draws"]) == (0, 1, 1)
    with pytest.raises(ToolError):
        mcp_server.get_player_stats(db_path, "alice")


def test_player_stats_on_empty_db(tmp_path: Path) -> None:
    path = str(tmp_path / "empty.db")
    db.init_db(path)
    stats = mcp_server.get_player_stats(path, X)
    assert (stats["games"], stats["wins"], stats["win_rate"]) == (0, 0, None)


def test_readonly_connection_rejects_writes(db_path: str) -> None:
    with pytest.raises(sqlite3.OperationalError):
        with db.connect_readonly(db_path) as conn:
            conn.execute("DELETE FROM games")


def test_readonly_connection_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        with db.connect_readonly(str(tmp_path / "missing.db")):
            pass


def test_server_registers_read_only_tools(db_path: str) -> None:
    server = mcp_server.create_server(db_path)
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == {
        "query_game_history",
        "get_move_sequence",
        "get_board_at_move",
        "get_player_stats",
    }
    assert all(t.annotations.read_only_hint for t in tools)


def test_server_call_tool(db_path: str) -> None:
    server = mcp_server.create_server(db_path)
    result = asyncio.run(server.call_tool("query_game_history", {"limit": 2}))
    assert not result.is_error
    games = result.structured_content["result"]
    assert [g["game_id"] for g in games] == [3, 2]


def test_server_call_tool_error(db_path: str) -> None:
    server = mcp_server.create_server(db_path)
    with pytest.raises(ToolError, match="no game with id 999"):
        asyncio.run(server.call_tool("get_move_sequence", {"game_id": 999}))
