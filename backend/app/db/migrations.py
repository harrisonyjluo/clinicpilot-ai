"""Small, idempotent SQLite upgrades for databases created before a change.

The project intentionally avoids a migration framework for the demo. Adding a
column with ``create_all`` is not supported by SQLAlchemy, so existing databases
need explicit ``ALTER TABLE`` statements. Fresh databases already have every
column and these calls become no-ops.
"""

from sqlalchemy import Engine, inspect, text

_HISTORY_SUMMARY_COLUMNS = {
    "booking_id": (
        "VARCHAR(10) REFERENCES patient_bookings(booking_id) "
        "ON DELETE SET NULL"
    ),
    "follow_up": "TEXT",
    "red_flags": "JSON NOT NULL DEFAULT '[]'",
    "confidence": "FLOAT NOT NULL DEFAULT 0.0",
    "needs_review": "BOOLEAN NOT NULL DEFAULT 0",
    "created_at": "DATETIME",
}


def upgrade_existing_schema(bind: Engine) -> None:
    """Add the booking-summary columns to an older history table.

    SQLite is the supported runtime store. Other dialects are left to their
    deployment migration tooling.
    """

    if bind.dialect.name != "sqlite":
        return

    inspector = inspect(bind)
    if "patient_treatment_history" not in inspector.get_table_names():
        return

    existing = {
        column["name"]
        for column in inspector.get_columns("patient_treatment_history")
    }
    with bind.begin() as connection:
        for name, definition in _HISTORY_SUMMARY_COLUMNS.items():
            if name not in existing:
                connection.execute(
                    text(
                        "ALTER TABLE patient_treatment_history "
                        f"ADD COLUMN {name} {definition}"
                    )
                )

        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ix_history_booking_id "
                "ON patient_treatment_history (booking_id)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_history_patient_id "
                "ON patient_treatment_history (patient_id)"
            )
        )
        connection.execute(
            text(
                "UPDATE patient_treatment_history "
                "SET created_at = CURRENT_TIMESTAMP "
                "WHERE created_at IS NULL"
            )
        )
