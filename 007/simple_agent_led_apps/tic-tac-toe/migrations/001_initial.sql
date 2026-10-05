-- UP
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    winner TEXT,
    total_moves INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS moves (
    id INTEGER PRIMARY KEY,
    game_id INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    move_number INTEGER NOT NULL,
    player TEXT NOT NULL,
    row INTEGER NOT NULL,
    col INTEGER NOT NULL,
    UNIQUE (game_id, move_number)
);

CREATE INDEX IF NOT EXISTS idx_moves_game_id ON moves(game_id);

-- DOWN
DROP INDEX IF EXISTS idx_moves_game_id;
DROP TABLE IF EXISTS moves;
DROP TABLE IF EXISTS games;
