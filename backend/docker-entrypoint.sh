#!/bin/sh
set -eu

# Seed only an empty database so container restarts never overwrite bookings.
if [ ! -f /app/data/clinicpilot.db ]; then
    python -m app.db.seed
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
