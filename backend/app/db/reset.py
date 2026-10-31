"""Drop and recreate the SQLite database, then seed it."""

from pathlib import Path

from sqlalchemy.engine import make_url

from app.core.config import get_settings
from app.db.seed import main as seed_main
from app.db.session import engine

settings = get_settings()


def main() -> None:
    database_url = make_url(settings.database_url)
    if database_url.get_backend_name() != "sqlite":
        raise RuntimeError("reset is only supported for the SQLite demo database")

    database = database_url.database
    if not database or database == ":memory:":
        raise RuntimeError("reset requires a file-backed SQLite database")

    engine.dispose()
    base_path = Path(database).expanduser().resolve()
    for suffix in ("", "-wal", "-shm"):
        Path(f"{base_path}{suffix}").unlink(missing_ok=True)

    print(f"Removed {base_path}")
    seed_main()


if __name__ == "__main__":
    main()
