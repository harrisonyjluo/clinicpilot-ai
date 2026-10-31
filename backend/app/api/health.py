"""Health endpoint. Does not depend on the LLM configuration."""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.repositories.health_repository import HealthRepository
from app.schemas.health import HealthResponse
from app.services.health_service import HealthService

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get(
    "/api/health",
    response_model=HealthResponse,
    responses={503: {"description": "Database is unavailable"}},
    summary="Liveness and database readiness",
)
def health(db: Session = Depends(get_db)):
    service = HealthService(HealthRepository(db))
    if not service.database_is_ready():
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "database": "error",
                "app": settings.app_name,
                "version": settings.app_version,
            },
        )
    return HealthResponse(
        status="ok",
        database="ok",
        app=settings.app_name,
        version=settings.app_version,
    )
