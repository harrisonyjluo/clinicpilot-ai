# ClinicPilot AI — Technical Design

This document describes the implemented ClinicPilot AI system: the backend API,
persistence, pure dense-vector RAG, LLM-only intent classification with
deterministic validation, clinic-first adaptive slot filling, grounded advice
generation, REST tool loop, availability validation, safety, the React UI, the
transcript-summarizer bonus, booking-linked summaries persisted as patient
treatment history, and Docker Compose assets. The backend default suite and the
frontend Vitest suite pass, and a 9-test live HTTP E2E suite has been run against
a configured provider using a disposable database. Docker Compose assets are
statically checked; run them with the commands in README.md.

---

## 1. Overview

ClinicPilot AI is a clinic assistant MVP. A patient chats with an assistant that can:

1. Answer questions about clinics, doctors, and treatments from the provided CSVs.
2. Recommend a treatment and doctor from a described symptom.
3. Look up the fixed demo patient's existing appointments.
4. Collect missing booking parameters over multiple turns.
5. Create a booking through a REST tool call and show it immediately in the UI.

The demo configuration sets `DEFAULT_PATIENT_ID=P001` (backend) and
`VITE_PATIENT_ID=P001` (frontend). The seeded profile is **P001 John Doe**
(DOB 1988-05-14), with allergies **Penicillin, Peanuts**, existing bookings
**B1001** and **B1004**, and history record **H9002**. No patient record is
hardcoded in application code; there is no authentication and no patient
switching.

Evaluation priority, highest first: clean architecture, intent routing,
AI orchestration/tool-calling, database persistence, then UI styling.

---

## 2. Architecture

### 2.1 Chosen shape: modular monolith

One FastAPI process hosts the HTTP API, application services, persistence, RAG
index, and the agent. The frontend is a separate React/Vite app. SQLite is the
only database.

```text
React/Vite (5173)
      │  REST
      ▼
FastAPI (8000)
  ├─ api/           HTTP adapters, validation, status codes
  ├─ services/      use cases and transaction boundaries
  ├─ repositories/  SQLAlchemy queries
  ├─ models/        ORM mappings
  ├─ agent/         LLM classifier, validated router, state machine, prompts, tool loop
  ├─ rag/           CSV knowledge base, embeddings, FAISS index, retriever
  └─ SQLite (backend/data/clinicpilot.db)
      ▲
      │  httpx loopback REST
      └─ agent/tools.py (BackendClient)
```

The AI core is a **logical service** (`agent/` + `rag/`) physically deployed
inside the same process. This is intentional: it removes a second service,
second port, second set of environment variables, and a class of live-demo
failure modes, while preserving the boundary through a real HTTP tool client.

### 2.2 Dependency rule

```text
api → services → repositories → models/db
services → agent, rag
agent → tool_client → HTTP → api
rag → mock-data (read); rag → backend/data/rag (own index cache)
```

- `api/` must not contain SQL or call the LLM.
- `services/` must not contain raw SQL or HTTP.
- `repositories/` must not contain business rules.
- `agent/` must not import `repositories`, `models`, `services`, or `db`.
- `rag/` must not write the database, call an LLM, or make HTTP calls.
- `schemas/` must not leak ORM objects.

The runtime path `chat → agent → HTTP → api → service → repository` is a
loopback, not an import cycle. An architecture test scans `agent/` imports and
fails if the boundary is broken.

### 2.3 Why this shape

The assignment evaluates clean architecture, intent routing, tool-calling, and
persistence. A single process with enforced layers makes all four visible and
testable. A two-service split would add operational work without adding a
required behavior. The suggested `ai-service/` directory is a suggestion, not a
requirement; `agent/` + `rag/` are the logical AI service and can be extracted
later without changing interfaces.

Alternatives considered:

| Option | Why not chosen |
|---|---|
| Two services + LangGraph + REST | More moving parts, contract drift, two ports, higher live-demo risk; no required behavior gained |
| MCP-first with a separate MCP server | Extra process/transport for four tools; better as a future optional wrapper |
| NestJS backend | Adds a second language/runtime; FastAPI shares Python with the AI core |
| External vector database / LangChain / microservices | The local exact FAISS index is sufficient at this corpus size; a server-based vector DB adds cost and failure modes without improving the evaluation |

---

## 3. Stack decisions

| Area | Choice | Rationale |
|---|---|---|
| Language | Python 3.11 (assignment requires 3.10+) | One language for API and AI |
| Backend | FastAPI | Native Pydantic validation, OpenAPI, async support |
| Persistence | SQLAlchemy 2.0 + SQLite | Zero-setup local DB; sufficient for a single-user demo |
| Validation | Pydantic v2 + pydantic-settings | Request/response schemas and typed settings |
| LLM | Anthropic `claude-opus-5-5` | Strong tool use; current default |
| Agent | LLM structured-output classifier + validated router + manual tool loop | Natural-language understanding is the LLM's job; deterministic code validates catalog IDs, merges slots, and enforces the state machine |
| RAG | Pure dense-vector retrieval (BGE-small-en-v1.5 + FAISS + cross-encoder gate) | Handles paraphrase/symptom language the sparse CSVs do not contain; local, reproducible, no external vector DB |
| Tool protocol | REST via `httpx` | Meets the brief; MCP can wrap the same tools later |
| Frontend | React 18 + Vite + TypeScript, plain CSS | Functional demo, no SSR/routing need |

Configuration comes from environment variables. No secrets are committed.

---

## 4. Intent recognition (LLM-only NLU)

Taxonomy:

| Intent | Example | Route |
|---|---|---|
| `emergency` | "I have chest pain" | Safety response; no booking |
| `booking_query` | "When is my next appointment?" | Booking lookup tool |
| `new_booking` | "I want to book a dermatologist" | Slot filling → create booking |
| `booking_confirmation` | "Yes, confirm" | Confirm pending booking |
| `treatment_recommendation` | "I have a rash" | RAG recommendation |
| `clinic_info` | "What are your Saturday hours?" | RAG clinic lookup |
| `smalltalk` | "Hello" | Greeting + capabilities |
| `out_of_scope` | "What's the weather?" | Polite refusal |

**Every turn makes one classifier call.** `AnthropicIntentClassifier`
(`agent/llm.py`) sends the user message plus the current booking stage,
collected slots, last question, recent history, patient allergies, today's date,
and the full clinic/doctor/treatment catalog. It asks for a single JSON object
with the intent, confidence, clinic/doctor/treatment slot values, free-text
unknown names, notes, and a safety flag. The model resolves relative
expressions ("tomorrow", "明天", "next Tuesday", "下午3点") to ISO dates/times
itself.

`agent/router.py` is a validator, not a text interpreter: it accepts catalog
`C###`/`D###`/`T###` IDs only, parses ISO date/time strings, reports unknown
clinic, doctor, or treatment names, merges validated slots into conversation
state, and computes missing slots. It contains no keyword, regex, substring, or
token matching over user messages; an architecture test enforces that.

The provider contract is plain JSON text rather than strict JSON-schema
enforcement, because the configured Anthropic-compatible provider endpoint does
not enforce schema defaults. The adapter tolerates a missing `confidence`, code
fences, and surrounding prose; only `intent` is required. An unparseable
response becomes `503 AI_RESPONSE_INVALID`, never an unhandled 500.

Priority order is encoded in the classifier prompt:

1. Emergency.
2. Active slot-filling state: decide whether the message is a slot answer, a
   change, an unrelated question, or a greeting; do not swallow it as a slot
   answer.
3. Pending confirmation (affirmative only in `CONFIRMING`).
4. Booking query.
5. New booking.
6. Recommendation / clinic info.
7. Smalltalk / out of scope.

Invalid IDs such as `D999` are cleared and reported with the real catalog
options; invalid dates such as `2026-02-30` stay missing and are asked again.

### 4.1 Clinic-first adaptive slot filling (2026-10-31)

Clinic is a conversation preference in `ConversationState`, not a fifth
persisted booking slot. `patient_bookings` has no `clinic_id`, and the
`create_booking` schema is unchanged. The clinic is derived from the chosen
doctor and shown before the doctor in recommendations and confirmations.

The deterministic orchestrator chooses the next question adaptively:

1. **Doctor known** — derive the clinic from the doctor. If exactly one
   treatment is compatible with the specialty, infer it; otherwise list the
   compatible treatments and ask. D101 now surfaces T001 *and* T004 instead of
   silently choosing T001.
2. **Treatment/symptom known** — derive the doctor and clinic; present the
   clinic first, then the doctor.
3. **Clinic known** — list only that clinic's doctors and ask which one.
4. **No anchor** — ask which clinic first, with names, addresses, and
   specialties.

The resulting order is doctor → treatment → date → time after the anchor, with
one question per turn. If an explicit clinic conflicts with a chosen doctor,
the clinic is kept and the doctor is cleared; the reply explains the conflict.
The final confirmation and booking reply both read clinic → doctor → treatment
→ date/time.

### 4.2 Doctor availability and treatment duration (2026-10-31)

The dataset has no per-doctor schedule. A doctor's bookable time is therefore
bounded by the operating hours of the clinic where they practice, parsed from
`clinics.csv` by `agent/availability.py` (deterministic catalog parsing, not
NLU). The time question states those hours. Before confirming, the orchestrator
rejects:

- a weekday on which the clinic is closed (for example Sunset on Sunday);
- a start time outside the clinic's operating window;
- a start time whose treatment `duration_minutes` would run past closing
  (for example a 30-minute dermatology consultation at 4:45 PM before a 5:00 PM
  close).

The invalid date/time slot is cleared, the reply explains the actual window,
and the user is asked again. This is not real-time slot inventory or
overbooking detection, which remains out of scope.

State machine (`agent/state.py`):

```text
IDLE
  └─ new_booking ──► COLLECTING {doctor, treatment, date, time}
                     clinic preference derived from doctor; not persisted
                         └─ all four slots ──► CONFIRMING
                                                  ├─ yes ──► REST write ──► IDLE
                                                  └─ changes ──► COLLECTING
```

A booking is written only from `CONFIRMING` after an explicit affirmative.
Missing slots produce one deterministic question at a time; no write is
possible from `COLLECTING`. Conversation state is in memory, keyed by
`conversation_id`, and a backend restart resets it. This is a documented demo
limitation.

---

## 5. AI orchestration and tool calling

The agent defines five strict tools, but only the three read-only tools are
offered to the LLM:

| Tool | Transport | Purpose |
|---|---|---|
| `search_knowledge` | Local `rag/` retriever | Clinic/doctor/treatment retrieval and recommendation evidence |
| `get_patient_profile` | `GET /api/patients/{patient_id}/context` | Profile, allergies, bookings, history |
| `get_patient_bookings` | `GET /api/bookings?patient_id=...` | Existing appointments |
| `create_booking` | `POST /api/bookings` | Persist after explicit confirmation; executed by the application, never by the model loop |
| `save_booking_summary` | `POST /api/bookings/{booking_id}/summary` | Persist the structured summary as a `patient_treatment_history` row linked to the booking; application-executed, not offered to the model |

Doctor search results are enriched with the doctor's clinic object so the model
can present the clinic before the doctor. The booking write schema remains
`doctor_id`, `treatment_id`, `booking_date`, `booking_time`, and optional
`notes`; there is no `clinic_id`.

Tool schemas use `strict: true`, `additionalProperties: false`, and explicit
`required` fields. The model is `claude-opus-5-5` with adaptive thinking and
`output_config.effort = "medium"`; forced tool choice is not used. The loop
executes read-only `tool_use` blocks, returns all tool results in one user
message, preserves thinking blocks, and caps at four iterations.

Guardrails are code-level, not prompt-only:

- The LLM receives only `search_knowledge`, `get_patient_profile`, and
  `get_patient_bookings`. Even if it emits `create_booking`, the orchestrator
  rejects it as "not available" and writes nothing.
- The application calls `create_booking` only from `CONFIRMING` with all four
  slots and an explicit affirmative.
- Patient identity is taken from conversation state, never from model input;
  the chat service resolves `patient_id` from the request or `DEFAULT_PATIENT_ID`
  and rejects requests where neither is present. REST remains patient-generic by
  user request.
- The classifier flags allergy, emergency, and out-of-scope input; the
  orchestrator then short-circuits with a deterministic safety reply before any
  reply-model call or write. Prompt-injection resistance is enforced by the
  read-only tool allowlist.
- Invalid doctor/treatment IDs and doctor/treatment mismatches are rejected
  before a write.
- Success is claimed only from a tool result; after creation the agent calls
  `GET /api/bookings` to verify the row exists.
- A stable `client_request_id` makes confirmation retries idempotent and
  guarantees at most one row.

REST is the primary integration. An optional MCP wrapper was not implemented;
if added later, it must reuse the same REST client and validation.

### 5.1 Advice generation (2026-11-01)

The assignment pairs RAG retrieval with advice generation. Recommendation turns
therefore run a dedicated path instead of a generic “search and summarize”:

1. `_handle_advice` pre-retrieves the top knowledge records for the user's
   message and places them in the reply prompt as the source of truth. The model
   may call `search_knowledge` again only for a genuinely different query.
2. `agent/advice.py` defines `ADVICE_GENERATION_GUIDELINES`, a seven-part
   contract: acknowledge the concern; ask at most two clarifying questions when
   the description is ambiguous; recommend clinic → doctor → treatment with
   duration and price; explain why the option fits the described need; give the
   treatment's pre-treatment preparation; personalize allergy and emergency
   guidance from the patient record and clinic data; and end with a concrete
   next step (book, call, or urgent care).
3. Replies are grounded: no invented clinic, doctor, treatment, fee, hour, or
   policy. The clinic is always named before the doctor, consistent with the
   clinic-first booking flow.
4. When no LLM is configured, `_format_advice_fallback` produces a deterministic
   reply from the same retrieved records. It includes the recommendation,
   duration/price, reason, preparation instructions, allergy warning, emergency
   instructions, follow-up advice, disclaimer, and a booking offer, so the
   demo still gives substantive advice without a key.
5. `clinic_info` keeps its plain grounded Q&A prompt; the advice contract is not
   forced onto hours/policy questions.

Covered by `test_recommendation_uses_an_explicit_advice_generation_contract`,
`test_clinic_info_does_not_use_the_advice_contract`,
`test_advice_fallback_is_grounded_when_no_llm_is_configured`, and
`test_treatment_search_results_expose_preparation_instructions`.

### 5.2 Streaming orchestration (2026-11-01)

The non-streaming `handle(request)` path is unchanged. The streaming path adds
an optional async `on_text` sink:

1. `ChatService.prepare` validates configuration and resolves the patient. The
   HTTP route calls it before opening the SSE response.
2. `ChatService.stream` runs `_run(request, on_text=emit)` in an
   `asyncio.Task`; `emit` pushes `{"type": "delta", ...}` events onto a queue
   that the response generator drains. A consumer disconnect cancels and awaits
   the producer.
3. `ChatOrchestrator.handle` forwards `on_text` into the booking-query, advice,
   and generic LLM branches. `_run_llm_loop` prefers
   `AnthropicLLM.stream_complete` and falls back to `complete()` plus one
   `on_text(turn.text)` when the client has no streaming method.
4. The orchestrator records streamed text and reconciles it with the
   authoritative reply: if the reply starts with the visible streamed prefix it
   emits only the remainder; otherwise the terminal `done` response replaces the
   rendered message. Deterministic/template replies go through the same sink as
   one delta.
5. The producer ends with one `done` event carrying
   `response.model_dump(mode="json")`; `AppError` becomes one `error` event with
   its code/message, and an unexpected exception becomes `CHAT_STREAM_FAILED`
   after being logged.

Covered by `tests/test_chat_stream.py`, the streaming cases in
`tests/test_orchestrator.py` and `tests/test_llm_adapter.py`, and the frontend
SSE client tests.

---

## 6. RAG design (dense-vector retrieval)

Corpus: 20 CSV rows — clinics 2, doctors 4, treatments 5, patients 3, bookings 4,
history 2. The index covers the 11 clinic/doctor/treatment documents. Patient,
booking, and history data are never embedded; they are served by the REST
patient/booking tools.

Retrieval is **pure dense-vector**. `app/rag/retriever.py` contains no BM25,
TF-IDF, token-overlap, keyword, regex-category, or synonym-string scoring. A
query is embedded once and ranked by cosine similarity. Catalog `D###`/`T###`
IDs and ISO date/time values are validated in `agent/router.py` from the
classifier's structured output; they are not retrieval signals.

### 6.1 Indexing pipeline

`python -m app.rag.build_index` performs a reproducible build:

1. **Load** `clinics.csv`, `doctors.csv`, and `treatments.csv` with a real CSV
   parser; validate columns and preserve quoted commas and numeric types.
2. **Build one parent `KnowledgeDocument` per record** (11 total) with `kind`,
   `id`, `title`, payload, and retrieval text.
3. **Write natural-language retrieval text** with safe semantic enrichment:
   category meaning and common patient phrasing, never invented fees, hours,
   doctors, or policies. Enrichment is index-only; returned payloads are the
   original CSV fields.
4. **Chunk** with sentence-aware windows (`RAG_CHUNK_MAX_CHARS=800`,
   `RAG_CHUNK_OVERLAP=120`). Short records produce one chunk; every chunk keeps
   `parent_id` so retrieval can return the whole record.
5. **Embed passages** with `fastembed` and `BAAI/bge-small-en-v1.5`, 384
   dimensions. BGE's query instruction
   (`Represent this sentence for searching relevant passages: `) is prepended
   to queries only, never to passages.
6. **Normalize** every vector to unit L2 norm and store `float32`.
7. **Persist** the chunks, vectors, per-kind FAISS indexes, and a manifest.

The runtime uses `load_or_build_store`: it loads a compatible persisted index
and rebuilds automatically when the index is missing, corrupt, built by a
different model or dimension, or older than the source CSVs (source
fingerprint). `RAG_REBUILD_INDEX=true` forces a rebuild.

### 6.2 Vector storage

```text
backend/data/rag/
  manifest.json        # schema version, model, dimension, metric, chunk count,
                       # source fingerprint, per-kind counts
  chunks.json          # chunk text + parent metadata + payload
  vectors.npz          # normalized float32 vectors per kind
  index_clinic.faiss   # exact inner-product index
  index_doctor.faiss
  index_treatment.faiss
```

One `faiss.IndexFlatIP` per kind gives exact cosine search (inner product on
normalized vectors) and avoids an over-fetch heuristic when filtering by kind.
The `VectorStore` seam allows `IndexHNSWFlat` or IVF-PQ at larger scale; the
retriever code does not change. Chunks and vectors are persisted separately
from the binary index, so a reload does not re-parse CSV or re-embed. Writes
use temporary files plus atomic rename. `backend/data/` is gitignored.

### 6.3 Similarity search

1. Embed and normalize the query once (with the BGE query instruction).
2. Search the requested kind, or all three, for
   `max(limit * 5, limit + 8)` candidates.
3. Clip inner-product scores to `[-1, 1]` cosine similarity; drop candidates
   below `RAG_MIN_SCORE` (default `0.48`).
4. Apply only structured metadata filters supplied by a validated tool
   argument (`kind`, `category`, `specialty`).
5. Keep the best chunk per parent document, so a multi-chunk record appears
   once.
6. Apply MMR (`RAG_MMR_LAMBDA=0.7`) for diversity when returning several
   results.
7. Run the **cross-encoder relevance gate** (see below).
8. Return `{kind, id, title, score, matched_on, payload}`; `matched_on` is
   `("vector",)`, never a keyword claim. No candidate above threshold means an
   explicit empty list.

### 6.4 Cross-encoder relevance gate

Dense bi-encoders optimize recall and can rank an off-topic query above the
absolute cosine threshold. After vector ranking, `FastEmbedReranker` scores
each `(query, chunk)` pair with
`Xenova/ms-marco-MiniLM-L-6-v2` and drops candidates below
`RAG_RERANK_MIN_SCORE` (default `-8.0`).

This is a **gate, not a re-ranker**: the cross-encoder decides relevance, while
the vector cosine score still determines the result order and the
recommendation margin. That preserves calibrated cosine thresholds and avoids
letting a second model silently reorder results. The gate runs on the small
top-k set, so latency stays bounded. The models download on first use and are
cached under `backend/data/`.

### 6.5 Recommendation

`recommend(query)` searches treatment documents only, takes the top two, and
requires:

- `top_score >= RAG_RECOMMEND_MIN_SCORE` (default `0.55`), and
- `top_score - runner_up_score >= RAG_RECOMMEND_MARGIN` (default `0.03`) when a
  runner-up exists, and
- the top hit to survive the cross-encoder gate.

It then resolves the doctor and clinic through the structured
`category → specialty → doctor → clinic` joins. Those joins are deterministic
data relationships, not retrieval. Weak or ambiguous evidence returns `None`
and the agent asks a clarifying question instead of guessing.

### 6.6 Golden-set outcomes

The 11-case golden set in `rag/evaluate.py` is the regression contract for
model, chunking, threshold, and index changes. Representative outcomes:

| User language | Expected semantic result | Doctor | Clinic |
|---|---|---|---|
| my face is breaking out | Dermatology / T003 | D103 | C002 |
| I twisted my ankle playing soccer | Orthopedics / T005 | D104 | C001 |
| my little one needs their shots | Pediatrics / T002 | D102 | C001 |
| I need a flu jab | Immunization / T004 | D101 | C001 |
| I want my yearly physical | Preventative Care / T001 | D101 | C001 |
| Saturday hours / cancellation policy | C001 / C002 clinic fields | — | C001/C002 |
| weather, rent a car, tax advice | no result | — | — |

These are expected retrieval outcomes, not keyword rules. Evaluation reports
`recall@k`, `MRR`, `nDCG@k`, and no-match accuracy:

```bash
python -m app.rag.evaluate --fail-under-recall 1.0 --fail-under-no-match 1.0
RUN_MODEL_TESTS=1 pytest -m model -q
```

Current result with the real models: recall@3 1.0, MRR 1.0, nDCG@3 1.0,
no-match accuracy 1.0, all 11 cases passing. The default pytest suite uses a
deterministic topic embedder and never downloads a model.

No external vector database is used. The exact local FAISS index is sufficient
for 11 vectors and keeps the demo self-contained; HNSW/IVF or a server-based
vector store remains a measured scaling option behind `VectorStore`.

---

## 7. Database schema

SQLite file: `backend/data/clinicpilot.db` (gitignored). SQLAlchemy 2.0 models
use `Base.metadata.create_all`; no Alembic in the initial build. SQLite foreign
keys are enabled with `PRAGMA foreign_keys=ON`.

| Table | Primary key | Columns | Foreign keys |
|---|---|---|---|
| `clinics` | `clinic_id` | clinic_name, address, phone, operating_hours, cancellation_policy, emergency_instructions | — |
| `doctors` | `doctor_id` | clinic_id, full_name, specialty, languages_spoken, consultation_fee_usd, bio | clinic_id → clinics |
| `treatments` | `treatment_id` | treatment_name, category, duration_minutes, price_usd, pre_treatment_instructions | — |
| `patients` | `patient_id` | first_name, last_name, dob, gender, phone, email, known_allergies, emergency_contact | — |
| `patient_bookings` | `booking_id` | patient_id, doctor_id, treatment_id, booking_date, booking_time, status, notes, created_at, client_request_id | patient_id → patients; doctor_id → doctors; treatment_id → treatments |
| `patient_treatment_history` | `record_id` | patient_id, doctor_id, booking_id (nullable, unique), visit_date, chief_complaint, diagnosis, prescribed_medication, doctor_notes, follow_up, red_flags (JSON list), confidence, needs_review, created_at | patient_id → patients; doctor_id → doctors; booking_id → patient_bookings (`ON DELETE SET NULL`) |

Indexes:

- `patient_bookings(patient_id, booking_date)`
- `doctors(clinic_id)`
- `patient_treatment_history(patient_id)`
- `patient_treatment_history(booking_id)` — unique

`patient_treatment_history` is the system of record for both seeded clinical
history and summaries generated from chat-created bookings. The four required
summary fields map onto its existing clinical columns:

| Summary field | History column | Storage |
|---|---|---|
| `chief_complaint` | `chief_complaint` | text |
| `diagnosis` | `diagnosis` | text |
| `medication` | `prescribed_medication` | newline-separated text; read back as a list |
| `doctor_instructions` | `doctor_notes` | newline-separated text; read back as a list |
| `follow_up` | `follow_up` | text or NULL |
| `red_flags` | `red_flags` | JSON list |
| `confidence` | `confidence` | float |
| `needs_review` | `needs_review` | boolean |

AI-created rows receive `H` + a zero-padded sequence from the current maximum
(`H9003` after the seeded `H9001`/`H9002`), adapt `patient_id` and `doctor_id`
from the booking, and set `visit_date` to the booking date. The nullable unique
`booking_id` links the history record to its booking while leaving the two
seeded history rows unlinked. There is no separate `booking_summaries` table;
the obsolete model was removed. There is no transcript column, and the raw
conversation is never persisted or logged.

Fresh databases get the full schema from `Base.metadata.create_all`. For an
existing SQLite database, `init_db()` runs an idempotent upgrade that adds the
six summary columns, creates both history indexes, and backfills `created_at`
without deleting existing rows. Seeded bookings have no linked summary and the
UI reports “No summary available”. Deleting a booking sets `booking_id` to NULL
rather than deleting the clinical history row.

Seed:

- Source: `mock-data/*.csv` (the six provided seed CSVs).
- Command: `python -m app.db.seed` from `backend/`.
- Idempotent upsert by natural key; user-created bookings are never deleted.
- Expected clean counts: clinics 2, doctors 4, treatments 5, patients 3,
  patient_bookings 4, patient_treatment_history 2.
- Demo reset: `python -m app.db.reset` deletes the SQLite file and re-seeds.

Booking persistence rules:

- Generate `booking_id` as `B` + zero-padded sequence (`B1005`, `B1006`, ...)
  from the max existing ID inside the transaction.
- Default status `Confirmed`; `created_at` is UTC now.
- Validate patient/doctor/treatment existence before insert; foreign keys are
  the final guard.
- One transaction per booking.
- Optional `client_request_id` is a nullable unique column for idempotent
  retries. SQLite permits multiple NULLs.

Implemented files: `backend/app/models/`, `backend/app/db/session.py`,
`backend/app/db/seed.py`, `backend/app/db/reset.py`.

---

## 8. API contract

### Implemented

| Method | Path | Description | Response |
|---|---|---|---|
| GET | `/api/health` | Liveness + database readiness; no LLM dependency | 200 with `status`, `database`, `app`, `version`; 503 if the DB ping fails |
| GET | `/api/bookings` | List a patient's bookings, ordered by date/time ascending | 200 `{patient_id, count, bookings[]}`; the `patient_id` query parameter is required |
| POST | `/api/bookings` | Create a booking for any existing patient; called by the agent's `create_booking` tool | 201 `BookingRead`; replayed `client_request_id` → 200 with the same row |
| GET | `/api/patients/{patient_id}/context` | Profile, allergy list, bookings, and treatment history | 200; the agent requests P001 context |
| POST | `/api/chat` | Message + optional `patient_id`/`conversation_id` | 200 `{reply, intent, conversation_id, booking?, tool_calls[]}`; 503 when `ANTHROPIC_API_KEY` is missing; 404 for an unknown patient |
| POST | `/api/chat/stream` | Same payload as `/api/chat`; replies as `text/event-stream` | 200 SSE: `delta` frames, then exactly one `done` frame carrying the full `ChatResponse`, or one `error` frame; configuration/patient errors return JSON 503/400 before the stream opens |
| POST | `/api/summarize` | Raw doctor–patient transcript + optional `patient_id`; stateless | 200 `TranscriptSummary`; 400 `PATIENT_ID_REQUIRED`; 404 `PATIENT_NOT_FOUND`; 422 invalid transcript; 503 `AI_NOT_CONFIGURED`/`AI_RESPONSE_INVALID` |
| GET | `/api/bookings/{booking_id}/summary` | Read the treatment-history record linked to a booking | 200 `BookingSummaryRead` with `record_id`, `booking_id`, and `visit_date`; 404 `BOOKING_NOT_FOUND` or `BOOKING_SUMMARY_NOT_FOUND` |
| POST | `/api/bookings/{booking_id}/summary` | Generate a summary and upsert the linked `patient_treatment_history` row from `{"transcript": "..."}`; called by the agent after a chat-created booking | 200 `BookingSummaryRead` with `record_id`, `booking_id`, and `visit_date`; 404 unknown booking; 422 empty/oversized transcript or unknown field; 503 `AI_NOT_CONFIGURED`/`AI_RESPONSE_INVALID` |
| GET | `/` | Discovery payload for the demo | name, version, docs, health, endpoints |
| GET | `/docs` | OpenAPI UI | HTML |
| GET | `/openapi.json` | Machine-readable contract | OpenAPI 3.1 JSON |

Patient scope: the API accepts any existing patient ID. Unknown patients
return 404 on booking creation, patient context, and chat; malformed IDs return
422. Chat resolves `patient_id` from the request or `DEFAULT_PATIENT_ID` and
returns 400 `PATIENT_ID_REQUIRED` when neither exists; every tool call is scoped
to the resolved conversation patient. The UI exposes no patient switcher; there
is no authentication.

Booking IDs are generated server-side as `B` + a zero-padded sequence from the
current maximum (`B1005` after seed). Dates use `YYYY-MM-DD`; times use
`HH:MM:SS`. `client_request_id` is optional and unique; a repeated request
returns the original booking with 200 instead of inserting a duplicate.

### Chat streaming contract

`POST /api/chat/stream` preserves the non-streaming payload and terminal shape.
Each frame is `data: <JSON>\n\n`:

```json
{"type": "delta", "delta": "Our Saturday hours are "}
{"type": "done", "response": {"reply": "...", "intent": "...", "conversation_id": "...", "booking": null, "tool_calls": []}}
{"type": "error", "error": {"code": "AI_UPSTREAM_UNAVAILABLE", "message": "..."}}
```

- LLM-generated answers stream the provider's real `text_delta` chunks. A
  deterministic/template reply is emitted as one delta; the client renders it
  with the same incremental message update path.
- The `done` payload is authoritative for `conversation_id`, `intent`,
  `booking`, and `tool_calls`; the frontend replaces the streamed text with
  `response.reply` at the end so tool-turn preambles cannot leave residue.
- Configuration and patient resolution happen before the response opens, so a
  missing API key returns JSON 503 `AI_NOT_CONFIGURED` and an unresolvable
  patient returns 400 `PATIENT_ID_REQUIRED`. Provider or orchestration failures
  after the stream opens become a single `error` frame.
- Responses set `Cache-Control: no-cache`, `Connection: keep-alive`, and
  `X-Accel-Buffering: no`; the consumer cancels the producer task if the client
  disconnects.
- The implementation lives in `backend/app/api/chat.py`
  (`StreamingResponse`), `backend/app/services/chat_service.py`
  (`prepare`/`stream`/`_run` queue bridge), and
  `backend/app/agent/llm.py::AnthropicLLM.stream_complete`. The orchestrator
  accepts an optional `on_text` sink and otherwise behaves identically.

`search_knowledge` runs against the local read-only index; it is not a REST
endpoint. All patient and booking reads/writes go through the REST API.

### Transcript summarizer (bonus)

`POST /api/summarize` accepts `{"transcript": "...", "patient_id": "P001"}`
(patient optional, resolved from `DEFAULT_PATIENT_ID`) and returns:

```json
{
  "chief_complaint": "...",
  "diagnosis": "...",
  "medication": ["..."],
  "doctor_instructions": ["..."],
  "follow_up": "...",
  "red_flags": ["..."],
  "confidence": 0.0,
  "needs_review": true
}
```

The four assignment-required fields are always present; missing information is
`"not stated"` or an empty list. The prompt receives the patient's actual
allergy list, requires exact medication names/dosages, and flags conflicts.
`needs_review` is forced true for any red flag, confidence below 0.7, or a
transcript shorter than 40 characters. The summarizer uses `messages.create`
with the shared tolerant JSON parser because the configured provider endpoint
does not reliably enforce schema output; invalid output maps to
`503 AI_RESPONSE_INVALID`. Implementation: `api/summarize.py`,
`services/summary_service.py`, `agent/summarizer.py`, `schemas/summary.py`.

`POST /api/summarize` remains stateless: it returns the structured object and
writes nothing. Booking-linked persistence uses the dedicated endpoints below.

### Booking-linked treatment-history summaries (corrected 2026-11-01)

The user corrected the persistence target: a summary belongs in the patient's
treatment history, not in a booking-owned table. The API keeps its booking-facing
routes, but both now read and write `patient_treatment_history`:

- `POST /api/bookings/{booking_id}/summary` accepts
  `{"transcript": "..."}` (1–20,000 characters, extra fields forbidden),
  resolves the booking and its patient, passes the patient's real allergy list
  to the summarizer, applies the same review rules, and creates or updates the
  history row whose unique `booking_id` matches. A retry updates that row.
- `GET /api/bookings/{booking_id}/summary` returns `BookingSummaryRead` — the
  four required fields plus `follow_up`, `red_flags`, `confidence`,
  `needs_review`, `record_id`, `booking_id`, `visit_date`, and `created_at`.
- The record's `record_id` identifies the history row (`H9003` after seeded
  history), `patient_id`/`doctor_id` come from the booking, and `visit_date` is
  the booking date. The four summary fields map to the clinical columns listed
  in §7; `medication` and `doctor_instructions` are stored as newline-separated
  text and returned as lists.
- The raw transcript is not stored: the history table has no transcript column,
  and the service never writes the request text to the database or logs.
- Seeded history rows (`H9001`, `H9002`) have no `booking_id`. Seeded bookings
  B1001/B1004 (and any booking without a linked history row) return
  `404 BOOKING_SUMMARY_NOT_FOUND`; the UI renders “No summary available”.
- There is no `booking_summaries` table or ORM model. An existing SQLite
  database is upgraded in place, preserving its rows.

**Automatic generation after a chat booking.** After
`booking_confirmation` creates a booking, the orchestrator builds a transcript
from the in-memory conversation (`Patient:` / `Clinic assistant:` lines, capped
to 20,000 characters from the end) and calls `save_booking_summary` through the
REST tool client. The resulting summary is saved automatically as a linked
`patient_treatment_history` row. This keeps the “agent never touches the ORM”
boundary. The call is isolated: a `ToolError` or unexpected exception is logged
and added to the chat trace, but the already-persisted booking still succeeds
and is returned. The summary tool is not offered to the LLM, so the model cannot
trigger it or fabricate a summary.

Implementation: `models/history.py`,
`repositories/booking_summary_repository.py`,
`services/booking_summary_service.py`, `schemas/summary.py`,
`db/migrations.py`, `api/bookings.py`, `agent/tools.py`,
`agent/orchestrator.py`.

### Error contract

Implemented as `{"error": {"code": "...", "message": "...", "details": ...}}`
for application errors, Pydantic validation failures, unknown routes, and
unexpected exceptions. Response headers echo or generate `X-Request-ID`;
request logs contain method, path, status, and duration but never message text
or patient data.

---

## 9. Safety

- Chat resolves the patient from the request or `DEFAULT_PATIENT_ID`. Every tool
  call is scoped to the conversation patient; no other patient's data is sent to
  the LLM. The REST API itself accepts any existing patient ID by explicit user
  request.
- No diagnosis; informational guidance and clinician referral.
- Ground every clinical claim in CSV rows or tool results.
- The classifier flags allergy conflicts before recommendation/booking. The
  orchestrator refuses with a deterministic safety reply: no reply-model call
  and no write. P002's `"None"` allergy value is normalized to an empty list.
- The transcript summarizer receives the patient's allergy list and flags any
  medication conflict; it never diagnoses beyond the transcript and requires
  clinician review. `POST /api/summarize` is stateless. Per-booking summaries
  persist only the structured object, never the raw transcript, and the UI
  labels them AI-generated and requiring clinician review.
- Clinic-hours and treatment-duration validation is operational scheduling
  safety, not clinical advice. The dataset has no per-doctor schedule, so the
  doctor's bookable window is the clinic's operating hours.
- The classifier flags emergency language; the orchestrator routes to
  911/emergency-room guidance with no reply-model call and no booking in the
  same turn. The classifier call itself still happens, because emergency
  detection is natural-language understanding.
- Prompt injection: user text is data; the read-only tool allowlist rejects
  `create_booking` even if the model emits it; no system-prompt disclosure.
- Invalid IDs and incompatible doctor/treatment pairs are rejected before any
  write.
- No raw patient text in logs by default.
- The UI states that this is an informational demo, not medical advice, and
  repeats emergency guidance for urgent symptoms.

Safety decisions come from the classifier's structured `safety_flag` and are
enforced in `agent/orchestrator.py`; there are no keyword or regex safety rules
over user messages. Covered by the router, orchestrator, chat-flow, and live E2E
tests.

---

## 10. Frontend

### 10.1 Shape and data flow

```text
App (refreshToken, activeTab)
 ├─ tablist: Chat | Bookings
 ├─ ChatPanel ── streamChat() ──► POST /api/chat/stream (SSE deltas + done)
 │    └─ booking present → onBookingCreated() → refreshToken += 1
 └─ BookingsPanel ── getBookings() ──► GET /api/bookings?patient_id=<configured>
      ├─ BookingTable ── Summary button ──► onViewSummary(booking)
      └─ BookingSummaryDialog ── getBookingSummary() ──► GET /api/bookings/{id}/summary
```

`App` owns the cross-panel state: a monotonically increasing `refreshToken` and
the selected tab. `ChatPanel` increments the token only when a chat response
contains a persisted booking. `BookingsPanel` refetches on mount, on manual
**Refresh**, and whenever `refreshToken` changes, so a booking created through
the agent appears without a page reload. Both panels stay mounted; the inactive
one is hidden with the `hidden` attribute, so chat history and the bookings
refresh effect survive tab switches. The selected panel fills the content width.

### 10.2 Components

- `ChatPanel` — receives the configured `patientId` as a prop, generic welcome
  message, persistent disclaimer, message list, typing indicator, composer.
  It consumes `/api/chat/stream`, appends each delta to one assistant message,
  replaces that message with the authoritative `done.response.reply`, keeps
  `conversation_id` across turns, sends through `AbortController`, keeps the
  failed message in the transcript, restores it to the input, and renders the
  backend error in a `role="alert"` banner.
- `MessageList` / `MessageInput` — a labelled `role="log"` transcript that scrolls
  to the newest entry whenever a message is added or sending starts, and a
  full-width textarea that auto-grows up to 160px. Enter sends, Shift+Enter
  inserts a newline, and Enter during IME composition is ignored; the Send
  button remains a fallback and stays disabled while the input is empty.
- `BookingsPanel` / `BookingTable` — loading, error, empty, and table states;
  booking count and last-updated time; formatted date/time/price; a Summary
  button on every row with an accessible per-booking label. The table sits in a
  labelled, keyboard-focusable `region` capped to the viewport height, so a long
  list scrolls vertically inside the panel with a sticky column header instead
  of stretching the page.
- `BookingSummaryDialog` — `role="dialog"` with `aria-modal` and a labelled
  heading; fetches the summary when opened, aborts stale requests, focuses the
  Close button, closes on Escape or backdrop click, and renders loading, ready,
  missing (`BOOKING_SUMMARY_NOT_FOUND` → “No summary available…”), and error
  states. The ready state shows the linked treatment-history `record_id` and
  `visit_date`, chief complaint, diagnosis, medication, doctor instructions,
  follow-up, red flags, a clinician-review notice, and a non-medical-advice
  disclaimer.
- `api/client.ts` — one `request<T>` helper that maps transport failures and
  structured backend errors (`{error: {code, message, details}}`) to a typed
  `ApiError`; aborts are rethrown untouched. `getBookingSummary()` hits the
  booking-linked summary endpoint.
- `types.ts` — request/response types mirroring the FastAPI schemas.

The UI deliberately has no router, global store, or component library. The
brief deprioritizes styling; plain CSS provides a readable tabbed, full-width
layout that works on narrow screens.

### 10.3 Configuration and security

- `VITE_PATIENT_ID` is required; `main.tsx` passes it to `App`, which passes it
  to both panels. It selects which patient the UI loads; no name, booking, or
  other patient record is built into the bundle. If it is missing, the app
  renders a configuration error instead of panels.
- `VITE_API_BASE_URL` defaults to `http://localhost:8000`; only `VITE_*`
  variables are exposed to the bundle.
- No API key, token, or backend secret is referenced in frontend code. The
  Anthropic key stays in the backend environment; the browser only sees
  structured API responses.
- The backend's default `CORS_ORIGINS` includes `http://localhost:5173`, the
  Vite dev origin. Other ports require updating the backend setting.
- Accessibility: labelled input, `role="log"` transcript with `aria-live`,
  `role="alert"` errors, `role="status"` loading/typing indicators,
  `tablist`/`tab`/`tabpanel` roles with `aria-selected` and `aria-controls`,
  table headers and caption, labelled Summary buttons, and a modal summary
  dialog with `role="dialog"`, `aria-modal`, and `aria-labelledby`.

### 10.4 Failure behavior

| Condition | UI result |
|---|---|
| Backend down | Error banner “Could not reach the backend. Is the API running?”; message kept for retry |
| `503 AI_NOT_CONFIGURED` | Error banner with the backend message; no silent failure |
| `503 AI_AUTHENTICATION_FAILED` / `AI_UPSTREAM_UNAVAILABLE` / `AI_UPSTREAM_ERROR` | Error banner with the backend message; message kept for retry |
| `503 AI_RESPONSE_INVALID` | Error banner explaining the assistant could not understand the provider's response; message kept for retry |
| Unexpected `500` | Structured error banner. CORS now wraps the request middleware, so the browser receives the backend message instead of reporting a transport failure |
| Aborted request (unmount) | Ignored; no state update |
| Empty bookings | “No bookings found.” |
| Booking response | Confirmation message, then automatic bookings refetch |
| Summary request in flight | Dialog shows “Loading summary…”; closing aborts the request |
| `404 BOOKING_SUMMARY_NOT_FOUND` | Dialog explains that summaries are generated automatically for chat bookings |
| Summary request failure | Dialog shows the backend/transport error in a `role="alert"` banner |

### 10.5 Frontend tests

`npm test` runs 42 Vitest tests in jsdom with Testing Library:

- `api/client.test.ts` — patient-scoped URL, JSON chat payload, structured
  `ApiError` for 503, transport failure mapped to status 0, per-booking summary
  URL and response parsing, SSE frames split across chunks resolved into
  deltas plus the terminal response, mid-stream `error` events mapped to
  `ApiError`, and a stream ending without `done` rejected as
  `CHAT_STREAM_INCOMPLETE`.
- `components/ChatPanel.test.tsx` — configured patient context without a
  built-in identity, supplied `P777` reaching the chat payload, disclaimer,
  send and append, streamed deltas rendered into exactly one assistant
  message, Enter-to-send, `conversation_id` reuse, booking callback, visible
  error with retry, empty input guard.
- `components/MessageInput.test.tsx` — Enter sends, Shift+Enter inserts a
  newline, IME composition is ignored, whitespace-only input is blocked, and
  the textarea grows to a 160px cap.
- `components/MessageList.test.tsx` — the transcript scrolls to its bottom when
  messages change and when the typing indicator starts.
- `components/BookingsPanel.test.tsx` — configured patient in request, header,
  and table caption; initial rows, manual refresh, refresh token, empty state,
  error banner, and the focusable "Bookings table" scroll region; a Summary
  button per row; loading, ready, missing, and error dialog states; the four
  required summary fields; the linked `record_id`/`visit_date`; and closing the
  dialog.
- `App.test.tsx` — a mocked booking response triggers a second `getBookings`
  and renders B1005 in the table; both panels receive the configured patient
  ID; tabs switch with exactly one visible panel; product copy contains no
  internal development labels or backend jargon.

`npm run build` runs `tsc --noEmit` first, so the production build is also the
strict type check.

---

## 11. Testing

Test layers:

| Layer | Scope | Network/LLM? |
|---|---|---|
| Contract/seed | Counts, idempotency, P001 context, relationships | No |
| API | Health, bookings CRUD/idempotency, patient context, chat contract and booking flow, errors, request IDs, OpenAPI | No |
| Unit | RAG vector search/chunking/MMR, router, state machine, safety | No |
| Integration | Chat with a scripted fake LLM | No |
| Agent | Tool loop, guardrails, read-back | No |
| Live smoke | One real Anthropic call, opt-in `-m live` | Yes |
| Live HTTP E2E | Real uvicorn, real loopback REST tools, real SQLite, real provider; opt-in `-m e2e` | Yes |
| Frontend | 42 Vitest + Testing Library tests in jsdom, mocked API client and SSE stream; `tsc --noEmit` + Vite build | No |
| End-to-end UI | Automated in jsdom with a mocked chat response; the live browser walkthrough is a manual check and was not run in this environment | Mocked |
| Container files | Static Dockerfile/Compose assertions; no runtime container test was run in this environment | No |

All implementation work follows the mandatory red-green-refactor workflow:
run and show a failing test before implementation, write the minimum code to
pass, then refactor with the suite green. Tests use
Arrange-Act-Assert and behavior-describing names; the default suite never calls
a real LLM.

Implemented tests:

- `tests/test_seed.py` — exact counts 2/4/5/3/4/2; second seed unchanged;
  P001 allergies/bookings; all treatments/doctors queryable; H9002 present.
- `tests/test_health.py` — 200 health, 503 database-failure path, OpenAPI path,
  root discovery, and summary routes in the discovery endpoint list.
- `tests/test_bookings.py` — seeded list, required `patient_id`, any seeded
  patient, create → B1005 and immediate GET visibility, idempotent replay,
  unknown doctor/treatment/patient with no write, structured 422.
- `tests/test_patients.py` — P001 and P002 profiles, `["Penicillin", "Peanuts"]`,
  B1001/B1004, H9002 with doctor name; 404/422 paths.
- `tests/test_chat.py` — empty-message 422, any/omitted `patient_id` reaching
  the 503 configuration check, missing API key 503 with `ANTHROPIC_API_KEY` in
  the message.
- `tests/test_api_contract.py` — all API routes in OpenAPI, generated and
  echoed `X-Request-ID`, structured 404.
- `tests/test_rag.py` — CSV fidelity, all seed IDs, semantic retrieval outcomes,
  Saturday hours, Sunset ranking, no-match behavior, typed join values.
- `tests/test_semantic_retrieval.py` — paraphrase ranking, metadata filters,
  MMR diversity, recommendation threshold/margin, no-match behavior.
- `tests/test_chunking.py` — sentence-aware chunking, overlap, long-sentence
  windows, parent links.
- `tests/test_embeddings.py` — normalization, query-instruction handling,
  dimension validation.
- `tests/test_vector_store.py` — persisted FAISS round trip, cosine ranking,
  per-kind indexes, metadata filters, MMR, parent deduplication, model and
  dimension mismatch rejection.
- `tests/test_rag_runtime.py` — build/persist/reload, stale-source rebuild,
  configured thresholds, injected reranker.
- `tests/test_reranking.py` — the cross-encoder gate drops weak vector hits
  while vector scores remain authoritative.
- `tests/test_rag_evaluation.py` — recall@k, MRR, nDCG, and no-match metrics.
- `tests/test_build_index.py` — build CLI summary and manifest for 11 records.
- `tests/test_chat_service.py` — the configured retriever is built once and
  reused.
- `tests/test_model_rag.py` — opt-in real-model semantic and golden-set checks
  (`RUN_MODEL_TESTS=1 pytest -m model`), excluded by default.
- `tests/test_conversation_state.py` — slot merge semantics, required order,
  store identity/reset, patient change reset.
- `tests/test_intent_router.py` — scripted-classifier dispatch for every intent,
  catalog ID and ISO date/time validation, unknown doctor/treatment names,
  merged missing slots, and normalized safety flags; no text-matching
  expectations.
- `tests/test_llm_adapter.py` — classifier JSON text parsing (fenced JSON and
  surrounding prose tolerated, missing confidence defaulted), unusable output →
  503 `AI_RESPONSE_INVALID`, provider error translation, base URL handling, and
  streaming text deltas forwarded while the final tool-call turn is preserved.
- `tests/test_agent_tools.py` — REST profile/booking reads, exactly one row,
  idempotent replay, structured errors, grounded search, strict schemas.
- `tests/test_orchestrator.py` — emergency/allergy/injection short-circuits,
  grounded tool loop, booking lookup, clinic-first adaptive slot filling,
  doctor-first clinic derivation without a clinic question, full slot-filling
  flow, one-write guarantee, adversarial `create_booking` rejection, invalid
  doctor, availability prompts/refusals, treatment-duration overrun, the
  explicit advice-generation contract, the clinic-Q&A path staying separate,
  the grounded no-LLM advice fallback, preparation instructions in treatment
  results, summary-failure isolation after a confirmed booking, LLM reply
  deltas forwarded incrementally, and deterministic replies emitted once.
- `tests/test_availability.py` — parses `Mon-Fri`/`Sat-Sun` ranges, open/closed
  days, start-inclusive/end-exclusive boundaries, 24-hour notation, and the
  closing time of the window containing a start time.
- `tests/test_chat_booking_flow.py` — async `/api/chat` through five turns,
  persisted B1005 visible via `/api/bookings`, exactly one automatic summary
  write with the booking conversation in the transcript, and unknown patient
  → 404.
- `tests/test_chat_stream.py` — `/api/chat/stream` emits incremental deltas
  then one `done` frame with the full response, maps an orchestration failure
  to one structured `error` frame after the stream opens, rejects a missing
  API key with JSON 503 before opening the stream, and rejects an empty
  message with 422.
- `tests/test_chat_service.py` — `DEFAULT_PATIENT_ID` fills an omitted chat
  patient, an explicit ID wins, and no ID plus no default → 400.
- `tests/test_config.py` — blank `DEFAULT_PATIENT_ID` is treated as unset and a
  configured ID is pattern-validated.
- `tests/test_architecture.py` — agent cannot import persistence/services/API;
  RAG cannot import HTTP, LLM, or persistence; `router.py` and `orchestrator.py`
  import no regular-expression module.
- `tests/test_summarize.py` — 10 tests: four required fields, allergy
  propagation, red-flag/low-confidence/short-transcript review, missing patient
  and key, prompt contract, invalid output → 503, route contract, 422 bounds.
- `tests/test_booking_summaries.py` — 12 tests: no `booking_summaries` table,
  persistence as `patient_treatment_history`, linked `record_id`/`visit_date`,
  upsert without duplication, review rules, missing API key, unknown booking,
  seeded booking with no summary, no raw-transcript column, and HTTP GET/POST
  success, 404, and 422 contracts.
- `tests/test_schema_migrations.py` — one test: an older history table gains the
  six summary columns and indexes in place, keeps its existing row, backfills
  `created_at`, and remains unchanged when `init_db()` runs twice.
- `tests/test_deployment_files.py` — static Docker/Compose assertions: base
  images, non-root user, healthchecks, ports, named volume, health dependency,
  `.dockerignore`, and no baked secret; also asserts this document records the
  stateless summarize endpoint and the booking-linked treatment-history
  persistence path.
- `tests/test_e2e_chat.py` — opt-in live HTTP E2E (9 tests): the reported
  four-turn transcript with no repeated reply, Chinese/English multi-turn
  booking, the full five-stage demo, doctor-first clinic derivation,
  closed-day refusal, treatment-duration refusal, transcript summarizer,
  emergency, and allergy refusal.
- `tests/test_live_smoke.py` — opt-in real Anthropic call; excluded by default.

Full backend suite: 222 passed, 20 opt-in tests deselected (live 1, model 10,
e2e 9); 242 tests collected in total. The default suite uses a temporary SQLite
database and a deterministic test embedder; it never calls a real LLM or
downloads a model. Frontend: 42 Vitest tests passed and `npm run build`
(`tsc --noEmit` + Vite) passed.

Live HTTP E2E (2026-10-31, provider configured through `ANTHROPIC_BASE_URL`):
`RUN_E2E_TESTS=1 E2E_BASE_URL=http://127.0.0.1:8002 pytest -m e2e -q` → 9
passed against an isolated, freshly seeded `backend/data/e2e-live.db`. This
covered the five-stage demo (clinic hours → grounded recommendation → existing
booking → confirmation → persisted row visible in `GET /api/bookings`), the
reported non-repetition transcript, doctor-first clinic derivation, closed-day
and treatment-duration refusals, the transcript summarizer, emergency, and
allergy refusal. The browser walkthrough is a manual check and was not run in
this environment.

Acceptance evidence: Saturday hours (C001 9:00 AM–1:00 PM); skin rash →
T003/D103/C002; next appointment → B1001; missing slots → questions with no
write; complete booking → exactly one B1005 row and immediate GET visibility;
peanut allergy → safe refusal; chest pain → 911 guidance; invalid D999 → no
write; injection/out-of-scope → refusal; no API key → 503.

Additional acceptance evidence: doctor-first "Dr. Sarah Jenkins" →
Metro Care auto-filled, asks treatment, no clinic question; treatment-first
dermatology → Sunset → Elena → Dermatology Consultation, clinic first; D101
without a treatment asks T001 vs T004; Sunday at Sunset → closed-day refusal;
4:45 PM 30-minute dermatology → duration-overrun refusal; time question and
confirmation show the clinic operating hours; all tested offline and in live
E2E, with the successful booking verified through `GET /api/bookings`.

Commands:

```bash
cd backend
source .venv/bin/activate
pytest
RUN_LIVE_TESTS=1 pytest -m live -q    # opt-in; requires ANTHROPIC_API_KEY
RUN_MODEL_TESTS=1 pytest -m model -q  # opt-in; downloads the local models
RUN_E2E_TESTS=1 pytest -m e2e -q      # opt-in; needs a running backend at E2E_BASE_URL
pytest tests/test_summarize.py -q
pytest tests/test_booking_summaries.py -q
pytest tests/test_schema_migrations.py -q
pytest tests/test_chat_booking_flow.py -q
python -m app.rag.build_index
python -m app.rag.evaluate --fail-under-recall 1.0 --fail-under-no-match 1.0
```

Coverage is requirement-driven, not a percentage target. Critical branches are
router priority, slot transitions, booking validation/idempotency, tool
guardrails, safety, and seed idempotency.

---

## 12. Limitations and future work

- Conversation state is in memory; restarting the backend resets it.
- Chat resolves its patient from `DEFAULT_PATIENT_ID`; the UI sends the patient
  from `VITE_PATIENT_ID`. Neither hardcodes a patient record, and neither
  exposes a patient switcher. The REST API accepts any existing patient ID.
  There is no authentication.
- No overbooking or real-time slot-inventory logic (explicitly out of scope).
  Availability validation is limited to clinic operating hours plus treatment
  duration fit; the dataset has no per-doctor schedule or appointment calendar.
- No multi-tenancy, migrations, or horizontal scaling.
- RAG uses an exact local FAISS index; HNSW/IVF or a server-based vector store
  is a documented scaling seam, not a demo requirement. The cross-encoder adds
  a small local-model latency cost.
- The React UI is implemented and unit/integration-tested in jsdom. The live
  HTTP E2E flow passed against the configured provider; the full browser
  walkthrough is a manual check and was not run in this environment. Without
  provider configuration the UI surfaces `AI_NOT_CONFIGURED`.
- Bonus features implemented: transcript summarizer (stateless and persisted
  per booking), advice generation, and Docker Compose. Seeded bookings B1001 and
  B1004 have no conversation and therefore no summary; only chat-created
  bookings are summarized automatically. Because the raw transcript is not
  retained, a later regeneration needs a newly supplied transcript.
  Not implemented: Langfuse/OpenTelemetry tracing, MCP wrapper, Kubernetes
  manifests (Compose satisfies the assignment's "Compose or `/k8s`" choice).
- The Docker assets are statically tested; `docker compose up --build` is the
  final runtime check on a Docker host. It was not run in this environment.

---

## 13. Observability and containers

Request-scoped logging records method, path, status, request ID, and duration,
plus structured error responses. Raw chat text and patient data are never
logged.

Implemented containers (bonus): `backend/Dockerfile` (Python 3.11 slim,
`libgomp1` for FAISS, non-root user, healthcheck), `backend/docker-entrypoint.sh`
(seed only when the SQLite file is absent, then uvicorn), `frontend/Dockerfile`
(Node 20 build → nginx), `frontend/nginx.conf`, `docker-compose.yml` (backend
8000, frontend 5173:80, `env_file`, healthcheck-gated dependency, named
`clinicpilot-data` volume), and `.dockerignore`. `tests/test_deployment_files.py`
asserts the static contract. The images were not built and Compose was not
started in this environment; `docker compose up --build` on a Docker host is the
final runtime check. SQLite requires a single replica and a persistent volume.

Not implemented: JSON-formatted spans/traces carrying conversation ID, intent,
tool names, token usage, or error codes; Langfuse/OpenTelemetry adapters. The
existing request-scoped logs remain the observability baseline.
