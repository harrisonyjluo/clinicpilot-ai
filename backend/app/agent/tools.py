"""Agent tool layer.

Every read and write the agent performs against clinic data goes through the
public REST API over HTTP, never through the ORM or service layer directly.
The patient identity is injected by the orchestrator from conversation state,
so the model cannot choose or override whose data it touches.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx
from pydantic import ValidationError

from app.rag.index import KnowledgeBase
from app.rag.retriever import KnowledgeRetriever, Recommendation, SearchResult
from app.schemas.booking import BookingCreate

READ_ONLY_TOOL_NAMES = frozenset(
    {"search_knowledge", "get_patient_profile", "get_patient_bookings"}
)


class ToolError(Exception):
    """An expected tool failure that can be reported back to the agent."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "TOOL_ERROR",
        status_code: int = 502,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details


class BackendClient:
    """Thin HTTP client for the clinic REST API."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ToolError(
                "The clinic backend could not be reached.",
                code="BACKEND_UNAVAILABLE",
                status_code=502,
            ) from exc

        if response.status_code >= 400:
            code = "BACKEND_ERROR"
            message = f"Backend returned HTTP {response.status_code}."
            details: Any | None = None
            try:
                envelope = response.json()
                error = envelope.get("error", {})
                code = error.get("code", code)
                message = error.get("message", message)
                details = error.get("details")
            except ValueError:
                pass
            raise ToolError(
                message,
                code=code,
                status_code=response.status_code,
                details=details,
            )

        try:
            return response.json()
        except ValueError as exc:
            raise ToolError(
                "The clinic backend returned a non-JSON response.",
                code="INVALID_BACKEND_RESPONSE",
                status_code=502,
            ) from exc

    async def get_patient_profile(self, patient_id: str) -> dict[str, Any]:
        return await self._request(
            "GET",
            f"/api/patients/{patient_id}/context",
        )

    async def get_patient_bookings(self, patient_id: str) -> dict[str, Any]:
        return await self._request(
            "GET",
            "/api/bookings",
            params={"patient_id": patient_id},
        )

    async def create_booking(self, payload: BookingCreate) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/api/bookings",
            json=payload.model_dump(mode="json"),
        )

    async def create_booking_summary(
        self,
        booking_id: str,
        transcript: str,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/bookings/{booking_id}/summary",
            json={"transcript": transcript},
        )

    async def verify_booking(
        self,
        booking_id: str,
        patient_id: str,
    ) -> dict[str, Any] | None:
        listing = await self.get_patient_bookings(patient_id)
        for booking in listing.get("bookings", []):
            if booking.get("booking_id") == booking_id:
                return booking
        return None


def _serialize_result(
    result: SearchResult,
    knowledge: KnowledgeBase,
) -> dict[str, Any]:
    serialized = {
        "kind": result.kind,
        "id": result.id,
        "title": result.title,
        "score": result.score,
        "matched_on": list(result.matched_on),
        "record": result.payload,
    }
    if result.kind == "doctor":
        clinic = knowledge.get_clinic(result.payload.get("clinic_id"))
        if clinic is not None:
            serialized["clinic"] = clinic
    if result.kind == "treatment":
        serialized["pre_treatment_instructions"] = result.payload.get(
            "pre_treatment_instructions"
        )
    return serialized


def _serialize_recommendation(
    recommendation: Recommendation | None,
) -> dict[str, Any] | None:
    if recommendation is None:
        return None
    return {
        "treatment": recommendation.treatment,
        "doctor": recommendation.doctor,
        "clinic": recommendation.clinic,
        "reason": recommendation.reason,
    }


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search_knowledge",
        "description": (
            "Search the clinic knowledge base for clinics, doctors, treatments, "
            "opening hours, policies, and treatment recommendations. Use this for "
            "any question about services or for grounding a recommendation. Only "
            "report facts returned by this tool; never invent clinic details."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 500,
                    "description": "The patient's question or described need.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_patient_profile",
        "description": (
            "Fetch the current patient's profile, allergies, emergency contact, "
            "existing bookings, and treatment history from the clinic backend. "
            "The patient is identified by the conversation; do not guess an ID."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_patient_bookings",
        "description": (
            "Fetch the current patient's bookings from the clinic backend. Use "
            "this for any question about existing or upcoming appointments."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "create_booking",
        "description": (
            "Persist a confirmed appointment for the current patient through the "
            "clinic backend. Only call this after the patient has explicitly "
            "confirmed every detail. The patient identity and idempotency key are "
            "supplied by the application, not the model."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "doctor_id": {
                    "type": "string",
                    "pattern": r"^D\d{3}$",
                    "description": "Doctor ID from the knowledge base.",
                },
                "treatment_id": {
                    "type": "string",
                    "pattern": r"^T\d{3}$",
                    "description": "Treatment ID from the knowledge base.",
                },
                "booking_date": {
                    "type": "string",
                    "pattern": r"^\d{4}-\d{2}-\d{2}$",
                    "description": "Appointment date in YYYY-MM-DD format.",
                },
                "booking_time": {
                    "type": "string",
                    "pattern": r"^\d{2}:\d{2}(:\d{2})?$",
                    "description": "Appointment time in HH:MM or HH:MM:SS format.",
                },
                "notes": {
                    "type": ["string", "null"],
                    "maxLength": 500,
                    "description": "Optional visit notes; null when not provided.",
                },
            },
            "required": [
                "doctor_id",
                "treatment_id",
                "booking_date",
                "booking_time",
                "notes",
            ],
        },
    },
]


class AgentTools:
    """Executes tool calls and enforces patient scoping and payload validation."""

    def __init__(
        self,
        knowledge: KnowledgeBase,
        retriever: KnowledgeRetriever,
        backend: BackendClient,
    ) -> None:
        self.knowledge = knowledge
        self.retriever = retriever
        self.backend = backend

    @staticmethod
    def definitions(read_only: bool = False) -> list[dict[str, Any]]:
        if read_only:
            return [
                definition
                for definition in TOOL_DEFINITIONS
                if definition["name"] in READ_ONLY_TOOL_NAMES
            ]
        return list(TOOL_DEFINITIONS)

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        *,
        patient_id: str,
        client_request_id: str | None = None,
    ) -> dict[str, Any]:
        arguments = arguments or {}

        if name == "search_knowledge":
            return self._search(arguments)
        if name == "get_patient_profile":
            return await self.backend.get_patient_profile(patient_id)
        if name == "get_patient_bookings":
            return await self.backend.get_patient_bookings(patient_id)
        if name == "create_booking":
            return await self._create_booking(
                arguments,
                patient_id=patient_id,
                client_request_id=client_request_id,
            )

        raise ToolError(
            f"Unknown tool: {name}",
            code="UNKNOWN_TOOL",
            status_code=400,
        )

    def _search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError(
                "search_knowledge requires a non-empty query.",
                code="VALIDATION_ERROR",
                status_code=422,
            )

        query = query.strip()
        grouped = self.retriever.search_all(query, limit_per_kind=2)
        results: list[SearchResult] = []
        seen: set[tuple[str, str]] = set()
        for bucket in grouped.values():
            for result in bucket:
                key = (result.kind, result.id)
                if key not in seen:
                    seen.add(key)
                    results.append(result)
        results.sort(key=lambda result: (-result.score, result.kind, result.id))

        return {
            "query": query,
            "results": [
                _serialize_result(result, self.knowledge)
                for result in results
            ],
            "recommendation": _serialize_recommendation(
                self.retriever.recommend(query)
            ),
        }

    async def _create_booking(
        self,
        arguments: dict[str, Any],
        *,
        patient_id: str,
        client_request_id: str | None,
    ) -> dict[str, Any]:
        payload_data = {
            "patient_id": patient_id,
            "doctor_id": arguments.get("doctor_id"),
            "treatment_id": arguments.get("treatment_id"),
            "booking_date": arguments.get("booking_date"),
            "booking_time": arguments.get("booking_time"),
            "notes": arguments.get("notes"),
            "client_request_id": client_request_id or str(uuid.uuid4()),
        }
        try:
            payload = BookingCreate.model_validate(payload_data)
        except ValidationError as exc:
            raise ToolError(
                "The booking tool arguments are invalid.",
                code="VALIDATION_ERROR",
                status_code=422,
                details=exc.errors(include_url=False),
            ) from exc

        return await self.backend.create_booking(payload)

    async def verify_booking(
        self,
        booking_id: str,
        *,
        patient_id: str,
    ) -> dict[str, Any] | None:
        return await self.backend.verify_booking(booking_id, patient_id)

    async def save_booking_summary(
        self,
        booking_id: str,
        transcript: str,
    ) -> dict[str, Any]:
        """Persist a structured summary through the REST API.

        This is an application-driven tool, not a model-selectable one, so it
        never appears in the LLM tool definitions.
        """

        if not booking_id or not transcript.strip():
            raise ToolError(
                "A booking ID and transcript are required.",
                code="VALIDATION_ERROR",
                status_code=422,
            )
        return await self.backend.create_booking_summary(
            booking_id,
            transcript,
        )
