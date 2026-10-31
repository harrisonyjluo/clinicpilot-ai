"""Container deployment assets: safe defaults and a one-command demo path."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_dockerfiles_build_the_backend_and_frontend_without_secrets():
    backend = (ROOT / "backend" / "Dockerfile").read_text(encoding="utf-8")
    frontend = (ROOT / "frontend" / "Dockerfile").read_text(encoding="utf-8")
    entrypoint = (ROOT / "backend" / "docker-entrypoint.sh").read_text(
        encoding="utf-8"
    )

    assert "FROM python:3.11-slim" in backend
    assert "USER clinicpilot" in backend
    assert "HEALTHCHECK" in backend
    assert "app.db.seed" in entrypoint
    assert "uvicorn app.main:app" in entrypoint
    assert "FROM node:20-alpine" in frontend
    assert "nginx:alpine" in frontend
    assert "VITE_PATIENT_ID" in frontend
    assert "ANTHROPIC_API_KEY" not in backend
    assert "ANTHROPIC_API_KEY" not in frontend


def test_compose_wires_health_checked_services_and_a_persistent_volume():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "backend:" in compose
    assert "frontend:" in compose
    assert "8000:8000" in compose
    assert "5173:80" in compose
    assert "clinicpilot-data" in compose
    assert "condition: service_healthy" in compose
    assert "env_file:" in compose
    assert "ANTHROPIC_API_KEY=" not in compose


def test_dockerignore_excludes_local_data_secrets_and_dependencies():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    for entry in (".env", "backend/data", "node_modules", ".venv", "dist"):
        assert entry in ignored


def test_tech_doc_documents_stateless_and_treatment_history_summary_paths():
    tech_doc = (ROOT / "TECH_DOC.md").read_text(encoding="utf-8")

    assert "/api/summarize" in tech_doc
    assert "writes nothing" in tech_doc.lower()
    assert "/api/bookings/{booking_id}/summary" in tech_doc
    assert "never persisted" in tech_doc.lower()
    assert "patient_treatment_history" in tech_doc
    assert "there is no separate `booking_summaries` table" in tech_doc.lower()
    assert "record_id" in tech_doc
    assert "visit_date" in tech_doc
