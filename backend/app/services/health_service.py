"""Health check use case."""

from sqlalchemy.exc import SQLAlchemyError

from app.repositories.health_repository import HealthRepository


class HealthService:
    def __init__(self, repository: HealthRepository) -> None:
        self._repository = repository

    def database_is_ready(self) -> bool:
        try:
            self._repository.ping()
        except SQLAlchemyError:
            return False
        return True
