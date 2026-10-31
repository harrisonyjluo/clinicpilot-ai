"""SQLAlchemy engine and session wiring."""

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()
_database_url = make_url(settings.database_url)
_is_sqlite = _database_url.get_backend_name() == "sqlite"

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    future=True,
)
SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


@event.listens_for(engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    if _is_sqlite:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def ensure_sqlite_directory() -> None:
    """Create the parent directory for a file-backed SQLite database."""

    if not _is_sqlite:
        return
    database = _database_url.database
    if not database or database == ":memory:":
        return
    Path(database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    """Create all tables and upgrade compatible older schemas in place."""

    ensure_sqlite_directory()
    from app.db.migrations import upgrade_existing_schema
    from app.models import Base

    Base.metadata.create_all(bind=engine)
    upgrade_existing_schema(engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields one session per request."""

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
