"""Chat orchestration seam: configuration checks and runtime wiring."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx

from app.agent.llm import AnthropicIntentClassifier, AnthropicLLM
from app.agent.orchestrator import ChatOrchestrator
from app.agent.router import IntentRouter
from app.agent.state import ConversationStore
from app.agent.tools import AgentTools, BackendClient
from app.core.config import Settings, get_settings
from app.core.errors import (
    AppError,
    BadRequestError,
    ServiceUnavailableError,
)
from app.rag.index import KnowledgeBase
from app.rag.retriever import KnowledgeRetriever
from app.rag.runtime import create_configured_retriever
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger("clinicpilot.chat_service")

TextDeltaSink = Callable[[str], Awaitable[None]]


class ChatService:
    def __init__(
        self,
        settings: Settings | None = None,
        orchestrator: ChatOrchestrator | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._orchestrator = orchestrator
        self._store = ConversationStore()
        self._knowledge: KnowledgeBase | None = None
        self._retriever: KnowledgeRetriever | None = None
        self._llm: AnthropicLLM | None = None
        self._classifier: AnthropicIntentClassifier | None = None

    def prepare(self, request: ChatRequest) -> ChatRequest:
        """Validate configuration and resolve the patient before streaming."""

        if not self._settings.anthropic_api_key:
            raise ServiceUnavailableError(
                "ANTHROPIC_API_KEY is not set. Configure it to enable chat.",
                code="AI_NOT_CONFIGURED",
            )

        patient_id = request.patient_id or self._settings.default_patient_id
        if patient_id is None:
            raise BadRequestError(
                "patient_id is required when DEFAULT_PATIENT_ID is not configured.",
                code="PATIENT_ID_REQUIRED",
            )
        if patient_id != request.patient_id:
            return request.model_copy(update={"patient_id": patient_id})
        return request

    async def handle(self, request: ChatRequest) -> ChatResponse:
        return await self._run(self.prepare(request))

    async def stream(
        self,
        request: ChatRequest,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield chat SSE events: incremental text, then one terminal payload."""

        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def emit(delta: str) -> None:
            await queue.put({"type": "delta", "delta": delta})

        async def produce() -> None:
            try:
                response = await self._run(request, on_text=emit)
                await queue.put(
                    {
                        "type": "done",
                        "response": response.model_dump(mode="json"),
                    }
                )
            except AppError as exc:
                await queue.put(
                    {
                        "type": "error",
                        "error": {
                            "code": exc.code,
                            "message": exc.message,
                        },
                    }
                )
            except Exception:
                logger.exception("chat_stream_failed")
                await queue.put(
                    {
                        "type": "error",
                        "error": {
                            "code": "CHAT_STREAM_FAILED",
                            "message": (
                                "The chat stream failed unexpectedly. "
                                "Please try again."
                            ),
                        },
                    }
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(produce())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _run(
        self,
        request: ChatRequest,
        *,
        on_text: TextDeltaSink | None = None,
    ) -> ChatResponse:
        if self._orchestrator is not None:
            if on_text is None:
                return await self._orchestrator.handle(request)
            return await self._orchestrator.handle(
                request,
                on_text=on_text,
            )

        async with httpx.AsyncClient(
            base_url=self._settings.backend_base_url,
            timeout=10.0,
        ) as client:
            orchestrator = self._build_orchestrator(client)
            if on_text is None:
                return await orchestrator.handle(request)
            return await orchestrator.handle(request, on_text=on_text)

    def _ensure_retriever(self) -> KnowledgeRetriever:
        if self._retriever is None:
            self._knowledge = KnowledgeBase.load_default()
            self._retriever = create_configured_retriever(
                self._settings,
                knowledge=self._knowledge,
            )
        return self._retriever

    def _build_orchestrator(
        self,
        client: httpx.AsyncClient,
    ) -> ChatOrchestrator:
        retriever = self._ensure_retriever()
        knowledge = self._knowledge
        if knowledge is None:  # pragma: no cover - _ensure_retriever sets it
            raise ServiceUnavailableError(
                "Clinic knowledge base is unavailable.",
                code="KNOWLEDGE_BASE_UNAVAILABLE",
            )
        if self._llm is None:
            self._llm = AnthropicLLM(self._settings)
        if self._classifier is None:
            self._classifier = AnthropicIntentClassifier(
                self._settings,
                knowledge,
            )

        router = IntentRouter(
            knowledge,
            classifier=self._classifier,
        )
        tools = AgentTools(
            self._knowledge,
            retriever,
            BackendClient(client),
        )
        return ChatOrchestrator(
            knowledge=self._knowledge,
            router=router,
            tools=tools,
            store=self._store,
            llm=self._llm,
        )
