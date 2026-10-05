# Applies pending migrations. Usage: python scripts/migrate.py [db_path]

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.database import DEFAULT_DB_PATH, init_db  # noqa: E402


def main(argv: list[str]) -> None:
    db_path = argv[1] if len(argv) > 1 else DEFAULT_DB_PATH
    version = init_db(db_path)
    print(f"{db_path}: schema version {version}")


if __name__ == "__main__":
    main(sys.argv)
