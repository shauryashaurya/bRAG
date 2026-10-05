# Pure tic-tac-toe board logic. Boards are 3x3 lists of "", "X" or "O".
# Functions never mutate their inputs; they return new boards.

from typing import Sequence

Board = list[list[str]]

SIZE = 3
EMPTY = ""
X = "X"
O = "O"
DRAW = "draw"

LINES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1), (0, 2)),
    ((1, 0), (1, 1), (1, 2)),
    ((2, 0), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 1), (1, 1), (2, 1)),
    ((0, 2), (1, 2), (2, 2)),
    ((0, 0), (1, 1), (2, 2)),
    ((0, 2), (1, 1), (2, 0)),
)


def new_board() -> Board:
    return [[EMPTY] * SIZE for _ in range(SIZE)]


# Move numbers are 1-based; X always moves first.
def player_for_move(move_number: int) -> str:
    return X if move_number % 2 == 1 else O


def is_valid_move(board: Board, row: int, col: int) -> bool:
    return 0 <= row < SIZE and 0 <= col < SIZE and board[row][col] == EMPTY


def apply_move(board: Board, row: int, col: int, player: str) -> Board:
    if player not in (X, O):
        raise ValueError(f"invalid player: {player!r}")
    if not is_valid_move(board, row, col):
        raise ValueError(f"invalid move: ({row}, {col})")
    result = [list(r) for r in board]
    result[row][col] = player
    return result


def winning_line(board: Board) -> list[tuple[int, int]] | None:
    for line in LINES:
        (r0, c0), (r1, c1), (r2, c2) = line
        first = board[r0][c0]
        if first != EMPTY and first == board[r1][c1] == board[r2][c2]:
            return list(line)
    return None


def winner(board: Board) -> str | None:
    line = winning_line(board)
    if line is None:
        return None
    r, c = line[0]
    return board[r][c]


def is_full(board: Board) -> bool:
    return all(cell != EMPTY for row in board for cell in row)


# Returns "X", "O", "draw", or None while the game is still in progress.
def game_result(board: Board) -> str | None:
    win = winner(board)
    if win is not None:
        return win
    if is_full(board):
        return DRAW
    return None


# Rebuilds the position after the first `upto` moves of (row, col, player) tuples.
def board_from_moves(moves: Sequence[tuple[int, int, str]], upto: int) -> Board:
    board = new_board()
    for row, col, player in moves[:upto]:
        board = apply_move(board, row, col, player)
    return board
