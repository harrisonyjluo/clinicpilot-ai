"""FastAPI application factory for ClinicPilot AI."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.bookings import router as bookings_router
from app.api.chat import router as chat_router
from app.api.errors import register_exception_handlers
from app.api.health import router as health_router
from app.api.patients import router as patients_router
from app.api.summarize import router as summarize_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.middleware import request_context_middleware
from app.db.session import init_db

settings = get_settings()
configure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Clinic assistant MVP: RAG, intent routing, and agentic booking.",
    lifespan=lifespan,
)
# CORS is registered last so it wraps the request middleware: a 500 produced
# by request_context_middleware still reaches the browser with CORS headers.
app.middleware("http")(request_context_middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
register_exception_handlers(app)

app.include_router(health_router)
app.include_router(bookings_router)
app.include_router(patients_router)
app.include_router(chat_router)
app.include_router(summarize_router)


@app.get("/", include_in_schema=False)
def root() -> dict[str, object]:
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": [
            "GET /api/bookings",
            "POST /api/bookings",
            "GET /api/bookings/{booking_id}/summary",
            "POST /api/bookings/{booking_id}/summary",
            "POST /api/chat",
            "POST /api/chat/stream",
            "GET /api/patients/{patient_id}/context",
            "POST /api/summarize",
        ],
    }
