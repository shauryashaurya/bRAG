# Entry point: prepares the database and launches the Tkinter window.

import tkinter as tk

from game.database import DEFAULT_DB_PATH, init_db
from ui.app import App


def main() -> None:
    init_db(DEFAULT_DB_PATH)
    root = tk.Tk()
    root.title("Tic-Tac-Toe")
    root.resizable(False, False)
    App(root, DEFAULT_DB_PATH).pack()
    root.mainloop()


if __name__ == "__main__":
    main()
