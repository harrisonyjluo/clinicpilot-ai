# ClinicPilot AI

A clinic assistant MVP with a conversational booking agent, a FastAPI backend, and a React UI. The target demo flow is:

**ask about clinics/doctors → get a grounded recommendation → check existing bookings → book through AI tool-calling → see the booking persisted and rendered in the UI.**

The React/Vite UI ships tabbed chat and live-bookings views on top of pure dense-vector RAG, LLM-only intent classification with deterministic validation, clinic-first adaptive slot filling through REST tools, doctor availability/treatment-duration validation, grounded advice generation, safety gates, the `/api/chat` agent loop, and SSE streaming through `/api/chat/stream`. Bonuses: `POST /api/summarize`, automatic booking summaries persisted as linked `patient_treatment_history` rows with a Summary button in the bookings table, and Docker Compose assets.

Verification: the backend default suite and frontend Vitest suite pass, and a 9-test live HTTP E2E suite has been run against a configured provider using a disposable database. Docker Compose assets are statically checked; run them with the commands below. Without a provider key the UI surfaces the backend's 503 instead of failing silently.

- Architecture and design decisions: [`TECH_DOC.md`](TECH_DOC.md)

---

## Stack

| Layer | Choice |
|---|---|
| Backend | Python 3.11, FastAPI, Pydantic v2 |
| Database | SQLite + SQLAlchemy 2.0 |
| AI | Anthropic `claude-opus-5-5` with adaptive thinking |
| Agent | LLM structured-output classifier (intent, slots, safety flags) + validated router + read-only tool loop over REST |
| RAG | Pure dense-vector retrieval: `fastembed` BGE-small-en-v1.5 (384-d) → persisted FAISS cosine index → cross-encoder relevance gate |
| Frontend | React 18 + Vite 8 + TypeScript 5, plain CSS, Vitest + Testing Library |
| Summarizer (bonus) | One `messages.create` call with tolerant JSON parsing; `POST /api/summarize` is stateless, while chat-created bookings persist a structured `patient_treatment_history` row linked by unique `booking_id` (never the raw transcript) |
| Containers (bonus) | Docker Compose: backend Dockerfile + nginx-served frontend build, named data volume, healthchecks |

The backend and AI core run in one FastAPI process. The AI module calls backend capabilities through real REST calls over loopback, not in-process imports.

Natural-language understanding is LLM-only. Every chat turn makes one classifier call that returns a structured `RouteAnalysis` object (intent, confidence, slots, unknown doctor/treatment names, safety flag). Deterministic code validates catalog `D###`/`T###` IDs and ISO date/time values, merges slots, runs the state machine, and persists. There is no keyword, regex, substring, or token matching over user messages.

---

## Prerequisites

- Python 3.10 or newer (3.11 recommended; developed on 3.11.18)
- `pip`
- Node.js 20+ and npm (developed on Node 26 / npm 11) for the frontend

## Backend setup

```bash
cd backend

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a local environment file and set the Anthropic key (chat returns 503 without it; bookings and health still work):

```bash
cp ../.env.example ../.env
# edit .env and set ANTHROPIC_API_KEY; never commit it
```

Run from `backend/` so the `app` package and the relative SQLite path resolve correctly.

---

## Environment variables

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `DATABASE_URL` | `sqlite:///<repo>/backend/data/clinicpilot.db` | No | SQLAlchemy database URL |
| `ANTHROPIC_API_KEY` | empty | Yes, for chat | Claude API key; chat returns 503 without it |
| `ANTHROPIC_BASE_URL` | empty | No | Base URL for an Anthropic-compatible provider endpoint; unset means `https://api.anthropic.com`. `.env` values are not exported to the process, so the adapter passes this setting to the SDK explicitly |
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | No | Model ID used by the agent |
| `BACKEND_BASE_URL` | `http://127.0.0.1:8000` | No | Loopback base URL used by the agent's REST tool client; must match the uvicorn port |
| `DEFAULT_PATIENT_ID` | empty | No | Patient used when `/api/chat` omits `patient_id`; unset means such requests return 400 `PATIENT_ID_REQUIRED` |
| `CORS_ORIGINS` | `http://localhost:5173` | No | Comma-separated allowed frontend origins |
| `LOG_LEVEL` | `INFO` | No | Application log level |
| `RAG_INDEX_DIR` | `backend/data/rag` | No | Persisted index directory |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | No | Local `fastembed` embedding model |
| `RAG_EMBEDDING_DIMENSION` | `384` | No | Embedding width; must match the index |
| `RAG_EMBEDDING_CACHE_DIR` | `backend/data/fastembed` | No | Model cache; Hugging Face caches also live under `backend/data/` |
| `RAG_MIN_SCORE` | `0.48` | No | Minimum cosine similarity for a retrieval hit |
| `RAG_RECOMMEND_MIN_SCORE` | `0.55` | No | Stricter minimum for a treatment recommendation |
| `RAG_RECOMMEND_MARGIN` | `0.03` | No | Minimum lead over the runner-up treatment |
| `RAG_MMR_LAMBDA` | `0.7` | No | Relevance/diversity balance for multi-result search |
| `RAG_CHUNK_MAX_CHARS` | `800` | No | Sentence-aware chunk size |
| `RAG_CHUNK_OVERLAP` | `120` | No | Shared context between consecutive chunks |
| `RAG_REBUILD_INDEX` | `false` | No | Force a rebuild on the next retriever construction |
| `RAG_RERANK_ENABLED` | `true` | No | Cross-encoder relevance gate |
| `RAG_RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | No | Local cross-encoder model |
| `RAG_RERANK_MIN_SCORE` | `-8.0` | No | Gate threshold; candidates below it are dropped |

The RAG settings are read from the same root `.env` or `backend/.env` as the other variables.

The app loads `.env` from the repository root and from `backend/` (the backend-level file wins if both exist). Defaults are safe for local development.

---

## Database: initialize, seed, reset

The six provided CSVs are copied to `mock-data/`. The seed uses a real CSV parser, upserts by natural key, and is safe to run repeatedly.

```bash
cd backend
source .venv/bin/activate

python -m app.db.seed      # create tables + upsert the six CSV datasets
python -m app.db.reset     # delete the SQLite file, recreate, and seed from scratch
```

Expected clean counts:

```text
clinics 2, doctors 4, treatments 5, patients 3,
patient_bookings 4, patient_treatment_history 2
```

`seed` never deletes bookings created later by the AI; `reset` is the way to return to the original demo state. The two seeded history rows (`H9001`, `H9002`) have no `booking_id`, so seeded bookings have no linked summary. A chat-created summary is appended to the same `patient_treatment_history` table as a new `H####` row whose unique `booking_id` points back to its booking; the raw transcript is never stored. An existing SQLite database is upgraded in place on startup with idempotent `ALTER TABLE` logic, so the schema change does not require deleting its data. The SQLite file is `backend/data/clinicpilot.db` and is gitignored.

### Build the dense-vector index

The first index build downloads the embedding model (`BAAI/bge-small-en-v1.5`, 384 dimensions). The first chat also downloads the cross-encoder relevance model (`Xenova/ms-marco-MiniLM-L-6-v2`). Their caches live under `backend/data/` and are gitignored.

```bash
cd backend
source .venv/bin/activate

python -m app.rag.build_index
```

Expected output: 11 documents/chunks across `clinic` 2, `doctor` 4, `treatment` 5, with the model name, dimension, and source fingerprint recorded in `backend/data/rag/manifest.json`.

The chat runtime loads that persisted index and rebuilds it automatically when it is missing, corrupt, built by a different model, or older than the source CSVs. `RAG_REBUILD_INDEX=true` forces a rebuild on the next start. Retrieval has no lexical fallback: the query and documents are embedded, L2-normalized, and ranked by cosine similarity, with MMR diversity, parent-document deduplication, and a cross-encoder relevance gate.

---

## Run the backend

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

- API root: <http://127.0.0.1:8000/>
- OpenAPI UI: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/api/health>

```bash
curl http://127.0.0.1:8000/api/health
```

```json
{"status":"ok","database":"ok","app":"ClinicPilot AI","version":"0.1.0"}
```

A `503` with `"status":"error"` means the database ping failed. Health does not call the LLM, so it works before an Anthropic key is configured.

Bookings and patient context:

```bash
# Seeded demo bookings (B1001, B1004)
curl "http://127.0.0.1:8000/api/bookings?patient_id=P001"

# Create one; the server assigns B1005
curl -X POST http://127.0.0.1:8000/api/bookings \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"P001","doctor_id":"D103","treatment_id":"T003","booking_date":"2026-12-01","booking_time":"10:00:00"}'

# Profile, allergies, bookings, and history for the chat agent
curl http://127.0.0.1:8000/api/patients/P001/context
```

Chat:

```bash
# First turn; reuse the returned conversation_id for follow-ups
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What are your Saturday hours?"}'

# Doctor-first booking: naming the doctor fills the clinic automatically
# (the reply asks for the treatment, never for a clinic)
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"I want to book with Dr. Sarah Jenkins.","conversation_id":"<id>"}'

# Treatment-anchored booking (same conversation_id in every turn)
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"I want to book a dermatology appointment next Tuesday at 10am.","conversation_id":"<id>"}'

# Confirm; the server writes one row
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Yes, please.","conversation_id":"<id>"}'
```

The booking flow is clinic-first and adaptive. A clinic may be named first, in
which case the reply recommends its doctors; a doctor name fills the clinic from
the doctor record; a treatment or symptom derives both the doctor and the
clinic. The slot question and the confirmation summary show the clinic's
operating hours. A date the clinic is closed, a time outside those hours, or a
treatment that cannot finish before closing is refused with the reason and the
time is cleared, so no invalid row is written. This validates opening hours;
overbooking and real-time slot inventory are out of scope.

Recommendation replies are generated with an explicit advice contract rather
than a thin search summary. They acknowledge the concern, ask at most two
clarifying questions when needed, name the clinic before the doctor and
treatment (with duration and price), explain why the option fits, include the
treatment preparation instructions, personalize allergy and emergency guidance,
and end with a concrete next step. The advice is grounded in retrieved CSV
records. When `ANTHROPIC_API_KEY` is absent, `POST /api/chat` still returns 503;
the orchestrator's deterministic advice fallback is exercised by the test suite.

Transcript summary (bonus):

```bash
curl -X POST http://127.0.0.1:8000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{"transcript":"Doctor: What brings you in today? Patient: Sore throat for three days. Doctor: Take penicillin and rest.","patient_id":"P001"}'
```

The response always contains `chief_complaint`, `diagnosis`, `medication`, and
`doctor_instructions`, plus `follow_up`, `red_flags`, `confidence`, and
`needs_review`. When `patient_id` is supplied it must exist, and known allergies
are added to `red_flags` with `needs_review: true`. `POST /api/summarize` itself
is stateless and writes nothing. A missing API key or provider failure returns
the same structured 503 codes as chat.

### Booking-linked treatment-history summaries

Every booking created through chat also gets a structured summary persisted
automatically as a `patient_treatment_history` row linked by its unique
`booking_id`. It is not stored in the bookings table and there is no separate
`booking_summaries` table. The raw transcript is **not** stored; only the
structured fields are persisted. Seeded history rows have no booking link, so
seeded bookings B1001 and B1004 have no conversation and their Summary dialog
says “No summary available”.

```bash
# Generate or replace the treatment-history summary for an existing booking
curl -X POST http://127.0.0.1:8000/api/bookings/B1005/summary \
  -H "Content-Type: application/json" \
  -d '{"transcript":"Doctor: What brings you in today? Patient: Sore throat for three days. Doctor: Take penicillin and rest."}'

# Read the linked history record (404 BOOKING_SUMMARY_NOT_FOUND when absent)
curl http://127.0.0.1:8000/api/bookings/B1005/summary
```

The response includes `record_id` and `visit_date` in addition to `booking_id`
and the four required fields. `POST` upserts, so retries update the same history
row rather than creating duplicates. The endpoint returns 404 for an unknown
booking, 422 for an empty/oversized transcript or an unknown field, and 503 when
the model is not configured. In the UI, each row in the bookings table has a
**Summary** button that opens an accessible dialog with the linked treatment
history record ID and visit date, the four required fields, follow-up, red
flags, a clinician-review notice, and loading/missing/error states. Summary
generation is isolated from booking creation: if the summary call fails, the
booking is still confirmed and visible.

`POST /api/chat` returns `{reply, intent, conversation_id, booking, tool_calls}`.
`POST /api/chat/stream` accepts the same payload and streams Server-Sent Events:
`{"type":"delta","delta":"..."}` frames followed by one
`{"type":"done","response":{...}}` frame carrying the same full response, or
one `{"type":"error","error":{"code","message"}}` frame if the provider or
orchestrator fails after the stream opens. The UI uses the streaming endpoint
and renders assistant text incrementally. Without `ANTHROPIC_API_KEY` both chat
endpoints return 503 `AI_NOT_CONFIGURED`; health, bookings, and patient context
still work. Provider failures also return a
structured 503: `AI_AUTHENTICATION_FAILED` (the key or base URL was rejected),
`AI_UPSTREAM_UNAVAILABLE` (the endpoint could not be reached), or
`AI_UPSTREAM_ERROR` (another provider error), or `AI_RESPONSE_INVALID` (the
classifier's JSON could not be parsed; fences, surrounding prose, and a missing
`confidence` are already tolerated). `patient_id` is required on
bookings and optional on chat; omitted chat `patient_id` resolves from
`DEFAULT_PATIENT_ID`, and chat returns 400 `PATIENT_ID_REQUIRED` when neither is
present. The API accepts any existing patient ID (unknown patients return 404),
while the demo UI exposes no patient switcher.

The agent reaches `/api/patients/...` and `/api/bookings` through its own
loopback REST API. If you run uvicorn on another port, set
`BACKEND_BASE_URL` to match; otherwise chat tool calls fail. The write path is
code-gated: the model only receives read-only tools, and booking creation runs
after explicit confirmation and read-back verification.

---

## Tests

The backend suite uses FastAPI's `TestClient` / `httpx.ASGITransport` and a temporary SQLite database; the default run never calls a real LLM.

```bash
cd backend
source .venv/bin/activate
pytest -q
```

Run a single file or test:

```bash
pytest tests/test_health.py -q
pytest tests/test_seed.py::test_seed_counts_match_the_dataset -q
pytest tests/test_bookings.py -q
```

Current coverage (222 tests passing, 20 opt-in tests deselected: live 1, model 10, e2e 9):

- `tests/test_seed.py` — exact seed counts, idempotent reseed, P001 profile and bookings B1001/B1004, all doctors/treatments, history H9002.
- `tests/test_health.py` — health 200, health path in OpenAPI, root discovery payload.
- `tests/test_bookings.py` — seeded list, required `patient_id`, any seeded patient, create/persist/read-back, idempotent `client_request_id`, unknown doctor/treatment/patient, structured validation errors.
- `tests/test_patients.py` — P001/P002 profile, allergy list, bookings, history; 404 and 422 paths.
- `tests/test_chat.py` — whitespace-only 422, any/omitted `patient_id` reaching the 503 configuration check, missing API key 503.
- `tests/test_chat_service.py` — `DEFAULT_PATIENT_ID` fills an omitted chat patient, an explicit ID wins, and no ID plus no default returns 400 `PATIENT_ID_REQUIRED`.
- `tests/test_config.py` — blank `DEFAULT_PATIENT_ID` is treated as unset, and a configured ID is pattern-validated.
- `tests/test_api_contract.py` — API routes in OpenAPI, `X-Request-ID` generation/echo, structured 404.
- `tests/test_rag.py` — CSV fidelity, semantic retrieval outcomes, Saturday hours, no-match behavior.
- `tests/test_semantic_retrieval.py` — paraphrase ranking, metadata filters, MMR diversity, recommendation thresholds and margin, no-match behavior.
- `tests/test_chunking.py` — sentence-aware chunking, overlap, long-sentence windows, parent links.
- `tests/test_embeddings.py` — normalization, query-instruction handling, dimension validation.
- `tests/test_vector_store.py` — persisted FAISS round trip, cosine ranking, per-kind indexes, metadata filters, MMR, parent deduplication, model/dimension mismatch rejection.
- `tests/test_rag_runtime.py` — build/persist/reload, stale-source rebuild, configured thresholds, injected reranker.
- `tests/test_reranking.py` — relevance gate drops weak vector hits while vector scores stay authoritative.
- `tests/test_rag_evaluation.py` — golden-set metrics (recall@k, MRR, nDCG, no-match).
- `tests/test_build_index.py` — CLI summary and manifest for the 11 indexed records.
- `tests/test_chat_service.py` — retriever factory is constructed once and reused.
- `tests/test_conversation_state.py` — slot merge/order, state store identity/reset, patient change reset.
- `tests/test_intent_router.py` — scripted-classifier dispatch for every intent, catalog ID and ISO date/time validation, unknown doctor/treatment names, merged missing slots, and normalized safety flags; no text-matching expectations.
- `tests/test_llm_adapter.py` — classifier JSON text parsing (fenced JSON and surrounding prose tolerated, missing confidence defaulted), unusable output → 503 `AI_RESPONSE_INVALID`, provider error translation, base URL handling, and streamed text deltas forwarded while the final tool-call turn is preserved.
- `tests/test_agent_tools.py` — REST profile/bookings reads, exactly one write, idempotent replay, structured errors, grounded search, strict schemas.
- `tests/test_orchestrator.py` — emergency/allergy/injection short-circuits, grounded tool loop, booking lookup, clinic-first adaptive slot filling, doctor-first clinic auto-fill, clinic-hours and treatment-duration rejection, one-write guarantee, adversarial write rejection, the explicit advice-generation contract, the grounded no-LLM advice fallback, summary-failure isolation after a confirmed booking, incremental LLM delta forwarding, and deterministic replies emitted once.
- `tests/test_availability.py` — operating-hours parsing: weekday ranges, Saturday windows, closed days, end-boundary exclusion, and 24-hour notation.
- `tests/test_chat_booking_flow.py` — five-turn `/api/chat` flow persists B1005, automatically writes exactly one summary for it, and `GET /api/bookings` returns the booking; unknown patient 404.
- `tests/test_chat_stream.py` — `/api/chat/stream` emits incremental deltas then one terminal `done` response, turns a post-open orchestration failure into one structured `error` frame, and returns JSON 503/422 before the stream opens for a missing key or empty message.
- `tests/test_summarize.py` — required four fields, tolerant model-output parsing, allergy red flags, unknown patient 404, validation errors, and the stateless endpoint.
- `tests/test_booking_summaries.py` — persistence as `patient_treatment_history`, linked `record_id`/`visit_date`, upsert without duplication, review rules, missing key/unknown booking, seeded booking with no summary, no raw-transcript column, no `booking_summaries` table, and GET/POST HTTP contracts.
- `tests/test_schema_migrations.py` — in-place upgrade of an older history table: adds summary columns/indexes, preserves existing rows, backfills `created_at`, and stays idempotent on a second startup.
- `tests/test_deployment_files.py` — static checks that the Dockerfiles, Compose file, and entrypoint exist and stay consistent with the documented ports and commands; also checks the documented stateless summarize endpoint and treatment-history summary path.
- `tests/test_architecture.py` — agent/RAG import-boundary guards and a no-regex rule for `router.py`/`orchestrator.py`.
- `tests/test_e2e_chat.py` — opt-in live HTTP E2E (9 tests): the reported four-turn transcript with no repeated reply, Chinese slot filling → one persisted booking visible in `GET /api/bookings`, doctor-first clinic auto-fill, closed-day and duration-overrun refusals, emergency and allergy refusal, the five-stage demo flow, and the transcript summarizer.
- `tests/test_live_smoke.py` — opt-in real Anthropic call; excluded by default.
- `tests/test_model_rag.py` — opt-in real-embedding semantic tests and golden-set thresholds; excluded by default.

The default suite is offline (`pytest.ini` deselects the `live`, `model`, and `e2e` markers) and uses a deterministic test embedder plus a temporary SQLite database. To run the opt-in suites:

```bash
# Real Anthropic call; requires ANTHROPIC_API_KEY
RUN_LIVE_TESTS=1 pytest -m live -q

# Real local embedding + cross-encoder models; first run downloads them
RUN_MODEL_TESTS=1 pytest -m model -q

# Live HTTP E2E against a running backend (start uvicorn first)
RUN_E2E_TESTS=1 E2E_BASE_URL=http://127.0.0.1:8000 pytest -m e2e -q

# Golden-set retrieval report; exits non-zero below the requested thresholds
python -m app.rag.evaluate --fail-under-recall 1.0 --fail-under-no-match 1.0
```

The evaluated report includes `recall_at_k`, `mrr`, `ndcg_at_k`, `no_match_accuracy`, and per-case ranks. The current 11-case set passes with perfect recall and no-match accuracy.

---

## Repository layout

```text
backend/
  app/
    api/            HTTP routes (health, bookings, patients, chat, errors)
    core/           settings, errors, logging, request middleware
    db/             engine/session, seed, reset
    models/         six SQLAlchemy models (including patient_treatment_history)
    repositories/   SQL only (health ping, booking/patient queries)
    schemas/        Pydantic request/response models
    services/       use cases (health, bookings, chat seam)
    agent/          LLM classifier, validated router, state, REST tools, orchestrator, availability
    rag/            CSV knowledge base, chunking, BGE embeddings, FAISS store, vector retriever, cross-encoder gate, build/evaluate CLIs
  tests/            pytest suite
  Dockerfile
  docker-entrypoint.sh
  requirements.txt
  pytest.ini
frontend/           React 18 + Vite + TypeScript UI and Vitest suite
  Dockerfile
  nginx.conf
docker-compose.yml  backend + frontend, named SQLite volume, healthchecks
mock-data/          the six seed CSVs
TECH_DOC.md
```

---

## API status

Implemented now:

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Liveness + database readiness |
| GET | `/api/bookings?patient_id=P001` | List a patient's bookings (`patient_id` required) |
| POST | `/api/bookings` | Create a booking (201); idempotent with `client_request_id` |
| GET | `/api/patients/{patient_id}/context` | Profile, allergies, bookings, and history |
| POST | `/api/chat` | RAG answer, recommendation, booking lookup, or slot-filling booking; returns intent, tool calls, and the persisted booking |
| POST | `/api/chat/stream` | Same payload as `/api/chat`, streamed as SSE `delta` frames plus one terminal `done` (full response) or `error` frame |
| POST | `/api/summarize` | Structured medical summary from a raw transcript (bonus); optional `patient_id` adds allergy red flags; stateless |
| GET | `/api/bookings/{booking_id}/summary` | Treatment-history record linked to one booking, including `record_id` and `visit_date`; 404 `BOOKING_SUMMARY_NOT_FOUND` when absent |
| POST | `/api/bookings/{booking_id}/summary` | Generate a summary and upsert the linked `patient_treatment_history` row from `{"transcript": "..."}`; called automatically after a chat-created booking |
| GET | `/` | Service discovery payload |
| GET | `/docs` | OpenAPI UI |

No additional knowledge-search endpoint is planned; `search_knowledge` is a
local tool over the persisted dense-vector RAG index.

---

## Frontend

The React/Vite app lives in `frontend/`. It presents two tabs — **Chat** and
**Bookings** — and switches between them without a page reload. The selected
tab fills the full content width.

- A chat panel for the configured patient, with the informational-demo
  disclaimer, incremental assistant text streamed from `/api/chat/stream`, a
  typing indicator, an error banner that keeps the failed message for retry,
  and `conversation_id` reuse across turns. The composer grows with the text,
  sends with **Enter**, inserts a newline with **Shift+Enter**, and leaves IME
  composition untouched. The transcript scrolls to the newest message after
  every send and reply.
- A bookings panel that loads `GET /api/bookings?patient_id=<configured>`, shows
  the booking table, last-updated time, empty/loading/error states, and a manual
  **Refresh** button. Each row has a **Summary** button that opens the linked
  `patient_treatment_history` record in an accessible dialog, including its
  record ID and visit date, or explains that no summary is available for a
  seeded booking. Long appointment lists scroll vertically
  inside a viewport-bounded, keyboard-focusable table region with a sticky
  header, so the page itself does not grow without limit.
- Automatic refresh of the bookings panel when a chat response contains a
  persisted booking, so a new booking appears without a page reload.

### Run the frontend

Start the backend first (port 8000), then:

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Open <http://localhost:5173>. `VITE_PATIENT_ID` is required and selects the
patient whose chat and bookings the UI loads; no patient record is built into
the frontend. `VITE_API_BASE_URL` defaults to `http://localhost:8000`. The
backend's default `CORS_ORIGINS` already allows `http://localhost:5173`.

The UI asks `POST /api/chat/stream` and `GET /api/bookings` for that configured
patient ID and renders only what the API returns. Set `VITE_PATIENT_ID` to any
patient that exists in the seeded database.

### Frontend tests and build

```bash
cd frontend
npm test           # 42 Vitest tests (jsdom + Testing Library)
npm run build      # tsc --noEmit && vite build
```

The tests cover the API client (URL/patient scoping, JSON payloads, structured
`ApiError` for 503 and an unreachable backend, booking-linked summary fetch,
SSE frames split across chunks, mid-stream `error` events, and a missing `done`
event), chat behavior (configured patient ID with no built-in demo identity,
streamed deltas rendered into one assistant message, multi-turn
`conversation_id`, booking confirmation callback, visible backend error,
whitespace-only guard, Enter-to-send with an IME-composition guard), the
auto-growing composer and scroll-to-latest transcript, bookings behavior (initial
load, manual refresh, refresh token, empty and error states, keyboard-focusable
scroll region, Summary button, loading/ready/missing/error dialog states, the
linked history record ID and visit date, and close), and App-level integration
(tab switching with exactly one visible panel,
product copy without internal development labels, and a mocked chat booking that
causes a second bookings fetch and renders B1005).

Automated coverage includes a 9-test live HTTP E2E suite that exercises the same
API calls the UI makes. The demo section above lists the manual browser flow
(hours → recommendation → next appointment → multi-turn booking → new row).
Without provider configuration, `POST /api/chat` and
`POST /api/chat/stream` return 503 `AI_NOT_CONFIGURED` and the chat panel
renders that error visibly.

---

## Docker Compose (bonus)

`docker-compose.yml` builds both services: the FastAPI backend on port 8000 and
an nginx-served production frontend build on port 5173. The backend entrypoint
seeds the database only when `/app/data/clinicpilot.db` does not exist, so a
restart never overwrites bookings. SQLite, the persisted RAG index, and the
embedding cache live in the named `clinicpilot-data` volume. The frontend waits
for the backend healthcheck.

```bash
cp .env.example .env          # set ANTHROPIC_API_KEY and DEFAULT_PATIENT_ID
docker compose up --build
curl http://127.0.0.1:8000/api/health
open http://localhost:5173    # or browse to it manually
```

**Verification boundary:** the Docker CLI is unavailable in the environment
where this was built, so the images were never built and this stack was never
run. `tests/test_deployment_files.py` only checks the files statically. Run the
commands above on a Docker-capable machine before treating containerized
deployment as verified.

---

## Demo patient

The demo configuration sets `DEFAULT_PATIENT_ID=P001` in the backend `.env` and
`VITE_PATIENT_ID=P001` in the frontend `.env`; the backend resolves the profile,
allergies, bookings, and history from the database. No patient record is
hardcoded in application code. The seeded demo patient is **P001 John Doe**
(DOB 1988-05-14), allergies **Penicillin, Peanuts**, with existing bookings
**B1001** (2026-11-05 09:00) and **B1004** (2026-11-15 11:00). There is no
authentication and the UI has no patient switcher. The REST API accepts any existing patient ID by explicit user request; every agent tool call is scoped to the conversation's patient. Seeded bookings B1001/B1004 have no linked treatment-history summary; a summary is generated automatically and appended to `patient_treatment_history` when a new appointment is booked through chat.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'app'` | Run commands from `backend/`, with the virtualenv active |
| `Address already in use` | Stop the other process or run `uvicorn ... --port 8001` |
| Health returns 503 | Re-run `python -m app.db.seed`; check `DATABASE_URL` and file permissions |
| Seed count mismatch | Run `python -m app.db.reset`; the counts should be 2/4/5/3/4/2 |
| Chat returns 503 `AI_NOT_CONFIGURED` | Set `ANTHROPIC_API_KEY` in the root `.env` or `backend/.env`, then restart uvicorn |
| Chat returns 503 `AI_AUTHENTICATION_FAILED` | The provider rejected the request (HTTP 401/403). Check that `ANTHROPIC_API_KEY` belongs to the provider at `ANTHROPIC_BASE_URL`, and that `ANTHROPIC_MODEL` exists there |
| Chat returns 503 `AI_UPSTREAM_UNAVAILABLE` | The configured provider endpoint is unreachable; check network access and `ANTHROPIC_BASE_URL` |
| Chat returns 503 `AI_UPSTREAM_ERROR` | The provider rejected an SDK parameter or model name; check `ANTHROPIC_MODEL` and provider compatibility with `messages.create`, `thinking`, and `output_config` |
| Chat returns 503 `AI_RESPONSE_INVALID` | The classifier output could not be parsed as a JSON object with an `intent`. The adapter already tolerates code fences, surrounding prose, and a missing `confidence`; check the `classifier_response_invalid` log line (message text is not logged) and the provider's JSON mode |
| Assistant repeats the same booking question | The classifier is not receiving the active booking stage/last question or is returning `new_booking` for unrelated messages. Run `pytest tests/test_intent_router.py tests/test_orchestrator.py`, then the live E2E regression |
| Booking on a Sunday/closed day or after hours is refused | Intended: doctor availability is the clinic's `operating_hours` from `clinics.csv`. The reply names the weekday and hours and asks for another date/time |
| Booking near closing time is refused | Intended: the treatment's `duration_minutes` would run past closing. The reply states the duration and closing time and asks for an earlier start |
| Doctor-first booking never asks which clinic | Intended: a doctor belongs to exactly one clinic in the dataset, so the clinic is derived from the doctor and shown in the slot summary and confirmation |
| Chat tool calls fail or time out | Ensure `BACKEND_BASE_URL` matches the running uvicorn host/port (default `http://127.0.0.1:8000`) |
| Booking confirmation creates two rows | Should not happen: retries reuse a stable `client_request_id`. Run `python -m app.db.reset` to return to the demo state and report the request IDs |
| Summary dialog says “No summary available” | Expected for seeded B1001/B1004 and any booking without a linked `patient_treatment_history` row. Summaries are attached only when the agent creates the booking; you can POST a transcript to `/api/bookings/{id}/summary` manually |
| Summary POST returns 503 `AI_NOT_CONFIGURED` | Set `ANTHROPIC_API_KEY`; booking creation itself does not depend on summary generation, so the booking remains valid |
| Existing database is missing summary columns | Restart the backend: `init_db()` applies the idempotent SQLite upgrade automatically. Do not delete the database; existing bookings and history rows are preserved |
| First chat or index build is slow | The embedding and reranker models are downloading; later runs use the `backend/data/` cache |
| `RAG index model mismatch` or stale results | Run `python -m app.rag.build_index` (or set `RAG_REBUILD_INDEX=true`) to rebuild for the configured model and CSVs |

If you start the server inside a sandboxed shell and see `error while attempting to bind ... Operation not permitted`, local port binding is disabled by that sandbox. Run the server in a normal terminal (or allow local binding in the sandbox settings).
