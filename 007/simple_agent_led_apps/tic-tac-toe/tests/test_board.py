import pytest

from game.board import (
    DRAW,
    O,
    X,
    Board,
    apply_move,
    board_from_moves,
    game_result,
    is_full,
    is_valid_move,
    new_board,
    player_for_move,
    winner,
    winning_line,
)


def make(rows: list[str]) -> Board:
    return [["" if ch == " " else ch for ch in row] for row in rows]


def test_new_board_is_empty() -> None:
    board = new_board()
    assert board == [["", "", ""], ["", "", ""], ["", "", ""]]
    assert game_result(board) is None


@pytest.mark.parametrize(
    "rows, expected_line",
    [
        (["XXX", "OO ", "   "], [(0, 0), (0, 1), (0, 2)]),
        (["OO ", "XXX", "   "], [(1, 0), (1, 1), (1, 2)]),
        (["OO ", "   ", "XXX"], [(2, 0), (2, 1), (2, 2)]),
        (["XO ", "XO ", "X  "], [(0, 0), (1, 0), (2, 0)]),
        (["OX ", "OX ", " X "], [(0, 1), (1, 1), (2, 1)]),
        (["O X", "O X", "  X"], [(0, 2), (1, 2), (2, 2)]),
        (["XO ", "OX ", "  X"], [(0, 0), (1, 1), (2, 2)]),
        (["O X", "OX ", "X  "], [(0, 2), (1, 1), (2, 0)]),
    ],
)
def test_x_wins_every_line(rows: list[str], expected_line: list[tuple[int, int]]) -> None:
    board = make(rows)
    assert winner(board) == X
    assert winning_line(board) == expected_line
    assert game_result(board) == X


def test_o_wins() -> None:
    board = make(["OOO", "XX ", "X  "])
    assert winner(board) == O
    assert game_result(board) == O


def test_draw() -> None:
    board = make(["XOX", "XOO", "OXX"])
    assert winner(board) is None
    assert is_full(board)
    assert game_result(board) == DRAW


def test_in_progress() -> None:
    board = make(["XO ", "   ", "   "])
    assert not is_full(board)
    assert game_result(board) is None


def test_apply_move_returns_new_board() -> None:
    board = new_board()
    after = apply_move(board, 1, 1, X)
    assert after[1][1] == X
    assert board[1][1] == ""


def test_apply_move_rejects_occupied_cell() -> None:
    board = apply_move(new_board(), 0, 0, X)
    with pytest.raises(ValueError):
        apply_move(board, 0, 0, O)


@pytest.mark.parametrize("row, col", [(-1, 0), (0, 3), (3, 3)])
def test_apply_move_rejects_out_of_range(row: int, col: int) -> None:
    assert not is_valid_move(new_board(), row, col)
    with pytest.raises(ValueError):
        apply_move(new_board(), row, col, X)


def test_apply_move_rejects_bad_player() -> None:
    with pytest.raises(ValueError):
        apply_move(new_board(), 0, 0, "Z")


def test_player_for_move_alternates() -> None:
    assert [player_for_move(n) for n in range(1, 6)] == [X, O, X, O, X]


def test_board_from_moves() -> None:
    moves = [(0, 0, X), (1, 1, O), (0, 1, X), (2, 2, O), (0, 2, X)]
    assert board_from_moves(moves, 0) == new_board()
    assert board_from_moves(moves, 2) == make(["X  ", " O ", "   "])
    final = board_from_moves(moves, len(moves))
    assert final == make(["XXX", " O ", "  O"])
    assert game_result(final) == X
