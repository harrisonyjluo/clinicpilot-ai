"""Shared test fixtures.

The database URL is set before importing any app module so the engine binds to
a temporary SQLite file. The default suite never calls a real LLM.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import delete

TEST_DIR = Path(tempfile.mkdtemp(prefix="clinicpilot-tests-"))
TEST_DB = TEST_DIR / "test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB}"
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ["ANTHROPIC_API_KEY"] = ""

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()
test_settings = get_settings()
test_settings.anthropic_api_key = None

from fastapi.testclient import TestClient  # noqa: E402

from app.db.seed import seed_database  # noqa: E402
from app.db.session import SessionLocal, engine, init_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models import PatientBooking, PatientTreatmentHistory  # noqa: E402

SEEDED_BOOKING_IDS = ("B1001", "B1002", "B1003", "B1004")
SEEDED_HISTORY_IDS = ("H9001", "H9002")


@pytest.fixture(scope="session", autouse=True)
def seeded_database():
    init_db()
    with SessionLocal() as session:
        seed_database(session)
        session.commit()
    yield
    engine.dispose()
    shutil.rmtree(TEST_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def clean_runtime_bookings(seeded_database):
    """Remove bookings created by tests; keep the seeded demo state."""

    with SessionLocal() as session:
        session.execute(
            delete(PatientTreatmentHistory).where(
                PatientTreatmentHistory.record_id.notin_(SEEDED_HISTORY_IDS)
            )
        )
        session.execute(
            delete(PatientBooking).where(
                PatientBooking.booking_id.notin_(SEEDED_BOOKING_IDS)
            )
        )
        session.commit()
    yield


@pytest.fixture()
def db_session(seeded_database):
    with SessionLocal() as session:
        yield session


@pytest.fixture()
def client(seeded_database):
    with TestClient(app) as test_client:
        yield test_client
