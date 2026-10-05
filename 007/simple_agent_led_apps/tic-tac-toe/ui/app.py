# Tkinter GUI: a playable board on the left, game history and replay controls on the right.

import tkinter as tk
from datetime import datetime

from game import database as db
from game.board import (
    DRAW,
    EMPTY,
    SIZE,
    Board,
    apply_move,
    board_from_moves,
    game_result,
    is_valid_move,
    new_board,
    player_for_move,
    winning_line,
)

CELL_FONT = ("Helvetica", 32, "bold")
PLAYER_COLORS = {"X": "#1f5fbf", "O": "#c0392b", EMPTY: "black"}
CELL_BG = "#f4f4f4"
WIN_BG = "#b9f0b4"
LAST_MOVE_BG = "#fff1a8"
AUTOPLAY_MS = 700


def describe_result(result: str) -> str:
    return "Draw" if result == DRAW else f"{result} wins"


def format_timestamp(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")


class App(tk.Frame):
    def __init__(self, master: tk.Misc, db_path: str = db.DEFAULT_DB_PATH) -> None:
        super().__init__(master, padx=12, pady=12)
        self.db_path = db_path
        self.board: Board = new_board()
        self.game_id: int | None = None
        self.move_number = 0
        self.finished = False
        self.mode = "play"
        self.replay_game_id: int | None = None
        self.replay_moves: list[tuple[int, int, str]] = []
        self.replay_index = 0
        self.autoplay_job: str | None = None
        self.history_ids: list[int] = []

        self._build_board_panel()
        self._build_history_panel()
        self.new_game()

    def _build_board_panel(self) -> None:
        panel = tk.Frame(self)
        panel.grid(row=0, column=0, sticky="n", padx=(0, 16))

        grid = tk.Frame(panel)
        grid.pack()
        self.cells: list[list[tk.Button]] = []
        for r in range(SIZE):
            row_buttons = []
            for c in range(SIZE):
                button = tk.Button(
                    grid,
                    text="",
                    font=CELL_FONT,
                    width=3,
                    height=1,
                    bg=CELL_BG,
                    command=lambda r=r, c=c: self.on_cell_click(r, c),
                )
                button.grid(row=r, column=c, padx=2, pady=2)
                row_buttons.append(button)
            self.cells.append(row_buttons)

        self.status = tk.Label(panel, text="", font=("Helvetica", 14), pady=8)
        self.status.pack()
        tk.Button(panel, text="New Game", command=self.new_game).pack()

    def _build_history_panel(self) -> None:
        panel = tk.Frame(self)
        panel.grid(row=0, column=1, sticky="ns")

        tk.Label(panel, text="Game history", font=("Helvetica", 12, "bold")).pack(anchor="w")
        list_frame = tk.Frame(panel)
        list_frame.pack(fill="both", expand=True)
        self.history = tk.Listbox(list_frame, width=42, height=14, font=("Courier", 10))
        scrollbar = tk.Scrollbar(list_frame, command=self.history.yview)
        self.history.configure(yscrollcommand=scrollbar.set)
        self.history.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.history.bind("<<ListboxSelect>>", self.on_history_select)

        tk.Button(panel, text="Refresh", command=self.refresh_history).pack(anchor="w", pady=(4, 8))

        controls = tk.Frame(panel)
        controls.pack(anchor="w")
        self.replay_buttons = [
            tk.Button(controls, text="|<", width=4, command=lambda: self.replay_seek(0)),
            tk.Button(controls, text="<", width=4, command=lambda: self.replay_step(-1)),
            tk.Button(controls, text=">", width=4, command=lambda: self.replay_step(1)),
            tk.Button(
                controls,
                text=">|",
                width=4,
                command=lambda: self.replay_seek(len(self.replay_moves)),
            ),
        ]
        self.play_button = tk.Button(controls, text="Play", width=6, command=self.toggle_autoplay)
        self.replay_buttons.append(self.play_button)
        for button in self.replay_buttons:
            button.pack(side="left", padx=1)

    # Rendering

    def render(
        self,
        board: Board,
        highlight: list[tuple[int, int]] | None = None,
        last_move: tuple[int, int] | None = None,
    ) -> None:
        for r in range(SIZE):
            for c in range(SIZE):
                mark = board[r][c]
                bg = CELL_BG
                if highlight and (r, c) in highlight:
                    bg = WIN_BG
                elif last_move == (r, c):
                    bg = LAST_MOVE_BG
                self.cells[r][c].configure(text=mark, fg=PLAYER_COLORS[mark], bg=bg)

    def set_replay_controls(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in self.replay_buttons:
            button.configure(state=state)

    # Play mode

    def new_game(self) -> None:
        self.stop_autoplay()
        self.mode = "play"
        self.board = new_board()
        self.game_id = None
        self.move_number = 0
        self.finished = False
        self.set_replay_controls(False)
        self.history.selection_clear(0, "end")
        self.render(self.board)
        self.status.configure(text=f"{player_for_move(1)} to move")
        self.refresh_history()

    def on_cell_click(self, row: int, col: int) -> None:
        if self.mode != "play" or self.finished or not is_valid_move(self.board, row, col):
            return
        player = player_for_move(self.move_number + 1)
        self.board = apply_move(self.board, row, col, player)
        self.move_number += 1
        result = game_result(self.board)

        with db.connect(self.db_path) as conn:
            if self.game_id is None:
                self.game_id = db.create_game(conn)
            db.record_move(conn, self.game_id, self.move_number, player, row, col)
            if result is not None:
                db.finish_game(conn, self.game_id, result, self.move_number)

        if result is None:
            self.render(self.board, last_move=(row, col))
            self.status.configure(text=f"{player_for_move(self.move_number + 1)} to move")
        else:
            self.finished = True
            self.render(self.board, highlight=winning_line(self.board))
            self.status.configure(text=describe_result(result))
        self.refresh_history()

    # History

    def refresh_history(self) -> None:
        with db.connect(self.db_path) as conn:
            games = db.list_games(conn)
        self.history.delete(0, "end")
        self.history_ids = []
        for game in games:
            if game["winner"] is not None:
                outcome = "draw" if game["winner"] == DRAW else f"winner {game['winner']}"
            elif game["id"] == self.game_id and not self.finished:
                outcome = "in progress"
            else:
                outcome = "abandoned"
            line = (
                f"#{game['id']:<4} {format_timestamp(game['created_at'])}  "
                f"{outcome:<11} ({game['total_moves']} moves)"
            )
            self.history.insert("end", line)
            self.history_ids.append(game["id"])
        if self.mode == "replay" and self.replay_game_id in self.history_ids:
            self.history.selection_set(self.history_ids.index(self.replay_game_id))

    def on_history_select(self, _event: tk.Event) -> None:
        selection = self.history.curselection()
        if selection:
            self.load_replay(self.history_ids[selection[0]])

    # Replay mode

    def load_replay(self, game_id: int) -> None:
        self.stop_autoplay()
        with db.connect(self.db_path) as conn:
            moves = db.get_moves(conn, game_id)
        self.mode = "replay"
        self.replay_game_id = game_id
        self.replay_moves = [(m["row"], m["col"], m["player"]) for m in moves]
        self.set_replay_controls(True)
        self.replay_seek(0)

    def replay_seek(self, index: int) -> None:
        self.replay_index = max(0, min(index, len(self.replay_moves)))
        board = board_from_moves(self.replay_moves, self.replay_index)
        total = len(self.replay_moves)
        text = f"Replay #{self.replay_game_id}: move {self.replay_index}/{total}"

        result = game_result(board)
        if result is not None:
            self.render(board, highlight=winning_line(board))
            text += f" - {describe_result(result)}"
        elif self.replay_index > 0:
            row, col, _ = self.replay_moves[self.replay_index - 1]
            self.render(board, last_move=(row, col))
        else:
            self.render(board)
        self.status.configure(text=text)

    def replay_step(self, delta: int) -> None:
        self.replay_seek(self.replay_index + delta)

    def toggle_autoplay(self) -> None:
        if self.autoplay_job is not None:
            self.stop_autoplay()
            return
        if self.replay_index >= len(self.replay_moves):
            self.replay_seek(0)
        self.play_button.configure(text="Pause")
        self.autoplay_job = self.after(AUTOPLAY_MS, self._autoplay_tick)

    def _autoplay_tick(self) -> None:
        self.replay_step(1)
        if self.replay_index >= len(self.replay_moves):
            self.stop_autoplay()
        else:
            self.autoplay_job = self.after(AUTOPLAY_MS, self._autoplay_tick)

    def stop_autoplay(self) -> None:
        if self.autoplay_job is not None:
            self.after_cancel(self.autoplay_job)
            self.autoplay_job = None
        self.play_button.configure(text="Play")
