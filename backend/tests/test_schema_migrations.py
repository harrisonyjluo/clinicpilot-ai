"""Existing databases are upgraded in place when the history schema evolves."""

from sqlalchemy import create_engine, inspect, text

from app.db import session as session_module

LEGACY_HISTORY_DDL = """
CREATE TABLE patient_treatment_history (
    record_id VARCHAR(10) NOT NULL,
    patient_id VARCHAR(10) NOT NULL,
    doctor_id VARCHAR(10) NOT NULL,
    visit_date DATE NOT NULL,
    chief_complaint TEXT NOT NULL,
    diagnosis TEXT NOT NULL,
    prescribed_medication TEXT NOT NULL,
    doctor_notes TEXT NOT NULL,
    PRIMARY KEY (record_id)
)
"""

SUMMARY_COLUMNS = {
    "booking_id",
    "follow_up",
    "red_flags",
    "confidence",
    "needs_review",
    "created_at",
}


def test_init_db_adds_summary_columns_to_an_existing_history_table(
    tmp_path,
    monkeypatch,
):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'legacy.db'}",
        future=True,
    )
    with engine.begin() as connection:
        connection.execute(text(LEGACY_HISTORY_DDL))
        connection.execute(
            text(
                "INSERT INTO patient_treatment_history ("
                "record_id, patient_id, doctor_id, visit_date, "
                "chief_complaint, diagnosis, prescribed_medication, doctor_notes"
                ") VALUES ("
                "'H9002', 'P001', 'D101', '2026-09-30', "
                "'annual checkup', 'Healthy', 'none', 'keep exercising'"
                ")"
            )
        )

    monkeypatch.setattr(session_module, "engine", engine)

    session_module.init_db()

    columns = {
        column["name"]
        for column in inspect(engine).get_columns("patient_treatment_history")
    }
    assert SUMMARY_COLUMNS <= columns

    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT record_id, chief_complaint, booking_id, "
                "red_flags, needs_review, created_at "
                "FROM patient_treatment_history WHERE record_id = 'H9002'"
            )
        ).mappings().one()
    assert row["chief_complaint"] == "annual checkup"
    assert row["booking_id"] is None
    assert row["red_flags"] in ("[]", None)
    assert row["needs_review"] in (0, False)
    assert row["created_at"] is not None

    session_module.init_db()

    assert {
        column["name"]
        for column in inspect(engine).get_columns("patient_treatment_history")
    } == columns
    engine.dispose()
