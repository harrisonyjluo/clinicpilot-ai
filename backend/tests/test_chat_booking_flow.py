"""End-to-end HTTP flow: chat slot filling persists a booking visible in REST."""

from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from app.agent.orchestrator import ChatOrchestrator
from app.agent.router import Intent, IntentRouter, RouteAnalysis
from app.agent.state import ConversationStore
from app.agent.tools import AgentTools, BackendClient
from app.api.dependencies import get_booking_summary_service, get_chat_service
from app.core.config import Settings
from app.core.errors import NotFoundError
from app.main import app
from app.rag.index import KnowledgeBase
from app.schemas.summary import BookingSummaryRead
from app.services.chat_service import ChatService
from tests.rag_support import build_test_retriever


class StubClassifier:
    def __init__(self, routes: dict[str, RouteAnalysis]) -> None:
        self.routes = routes

    async def classify(self, message, state, *, today):
        if message not in self.routes:
            raise AssertionError(
                f"Unexpected message reached the classifier: {message!r}"
            )
        return self.routes[message]


def analysis(intent: Intent, **overrides) -> RouteAnalysis:
    values = {"intent": intent, "confidence": 0.9}
    values.update(overrides)
    return RouteAnalysis(**values)


class StubBookingSummaryService:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.saved: dict[str, BookingSummaryRead] = {}

    async def generate(self, booking_id: str, request):
        self.calls.append(
            {"booking_id": booking_id, "transcript": request.transcript}
        )
        result = BookingSummaryRead(
            record_id="H9003",
            booking_id=booking_id,
            visit_date=date(2026, 11, 3),
            chief_complaint="skin rash on the arm",
            diagnosis="not stated",
            medication=[],
            doctor_instructions=["keep the area clean"],
            follow_up=None,
            red_flags=[],
            confidence=0.8,
            needs_review=True,
            created_at=datetime.now(timezone.utc),
        )
        self.saved[booking_id] = result
        return result

    def get(self, booking_id: str) -> BookingSummaryRead:
        if booking_id not in self.saved:
            raise NotFoundError(
                f"No summary exists for booking {booking_id}.",
                code="BOOKING_SUMMARY_NOT_FOUND",
            )
        return self.saved[booking_id]


@asynccontextmanager
async def chat_test_app():
    api_transport = httpx.ASGITransport(app=app)
    tool_transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=api_transport,
        base_url="http://testserver",
    ) as api_client, httpx.AsyncClient(
        transport=tool_transport,
        base_url="http://testserver",
    ) as tool_client:
        knowledge = KnowledgeBase.load_default()
        retriever = build_test_retriever(knowledge)
        classifier = StubClassifier(
            {
                "I want to book an appointment.": analysis(
                    Intent.NEW_BOOKING
                ),
                "Dermatology": analysis(
                    Intent.NEW_BOOKING,
                    treatment_id="T003",
                ),
                "next Tuesday": analysis(
                    Intent.NEW_BOOKING,
                    booking_date="2026-11-03",
                ),
                "10am": analysis(
                    Intent.NEW_BOOKING,
                    booking_time="10:00",
                ),
                "Yes, please.": analysis(Intent.BOOKING_CONFIRMATION),
            }
        )
        orchestrator = ChatOrchestrator(
            knowledge=knowledge,
            router=IntentRouter(knowledge, classifier),
            tools=AgentTools(
                knowledge,
                retriever,
                BackendClient(tool_client),
            ),
            store=ConversationStore(),
            llm=None,
            today=lambda: date(2026, 10, 31),
        )
        service = ChatService(
            settings=Settings(
                _env_file=None,
                anthropic_api_key="test-key",
                default_patient_id="P001",
            ),
            orchestrator=orchestrator,
        )
        summary_service = StubBookingSummaryService()
        app.dependency_overrides[get_chat_service] = lambda: service
        app.dependency_overrides[get_booking_summary_service] = (
            lambda: summary_service
        )
        try:
            yield SimpleNamespace(
                client=api_client,
                summaries=summary_service,
            )
        finally:
            app.dependency_overrides.pop(get_chat_service, None)
            app.dependency_overrides.pop(get_booking_summary_service, None)


@pytest.mark.asyncio
async def test_chat_booking_flow_persists_one_row_and_lists_it():
    async with chat_test_app() as ctx:
        api_client = ctx.client
        first = await api_client.post(
            "/api/chat",
            json={"message": "I want to book an appointment."},
        )
        assert first.status_code == 200
        first_body = first.json()
        conversation_id = first_body["conversation_id"]
        assert first_body["intent"] == "new_booking"
        assert first_body["booking"] is None
        assert "clinic" in first_body["reply"].casefold()
        assert "Sunset Wellness Center" in first_body["reply"]

        for message in ("Dermatology", "next Tuesday"):
            response = await api_client.post(
                "/api/chat",
                json={
                    "message": message,
                    "conversation_id": conversation_id,
                },
            )
            assert response.status_code == 200
            assert response.json()["booking"] is None

        ready = await api_client.post(
            "/api/chat",
            json={
                "message": "10am",
                "conversation_id": conversation_id,
            },
        )
        assert ready.status_code == 200
        ready_body = ready.json()
        assert ready_body["booking"] is None
        assert "2026-11-03" in ready_body["reply"]
        assert "10:00 AM" in ready_body["reply"]
        assert ready_body["reply"].index(
            "Sunset Wellness Center"
        ) < ready_body["reply"].index(
            "Dr. Elena Rostova"
        ) < ready_body["reply"].index("Dermatology Consultation")

        confirmed = await api_client.post(
            "/api/chat",
            json={
                "message": "Yes, please.",
                "conversation_id": conversation_id,
            },
        )
        assert confirmed.status_code == 200
        confirmed_body = confirmed.json()
        assert confirmed_body["booking"]["booking_id"] == "B1005"
        assert confirmed_body["booking"]["patient_id"] == "P001"
        assert "confirmed" in confirmed_body["reply"].casefold()
        assert "create_booking" in {
            call["name"] for call in confirmed_body["tool_calls"]
        }

        listing = await api_client.get(
            "/api/bookings",
            params={"patient_id": "P001"},
        )
        assert listing.status_code == 200
        listing_body = listing.json()
        assert listing_body["count"] == 3
        assert "B1005" in {
            booking["booking_id"] for booking in listing_body["bookings"]
        }

        assert len(ctx.summaries.calls) == 1
        summary_call = ctx.summaries.calls[0]
        assert summary_call["booking_id"] == "B1005"
        assert "Dermatology" in summary_call["transcript"]
        assert "Yes, please." in summary_call["transcript"]
        assert "Sunset Wellness Center" in summary_call["transcript"]

        summary = await api_client.get("/api/bookings/B1005/summary")
        assert summary.status_code == 200
        assert summary.json()["chief_complaint"] == "skin rash on the arm"


@pytest.mark.asyncio
async def test_chat_unknown_patient_returns_structured_404():
    async with chat_test_app() as ctx:
        response = await ctx.client.post(
            "/api/chat",
            json={"message": "Hello", "patient_id": "P999"},
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "PATIENT_NOT_FOUND"
