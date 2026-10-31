"""Database liveness query used by the health endpoint."""

from sqlalchemy import text
from sqlalchemy.orm import Session


class HealthRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def ping(self) -> None:
        """Run a trivial query; raise SQLAlchemyError if the database is down."""

        self._session.execute(text("SELECT 1"))
