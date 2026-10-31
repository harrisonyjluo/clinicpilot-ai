"""Chat endpoints: the stable JSON contract and the SSE streaming variant."""

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_chat_service
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api/chat", tags=["chat"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(event: dict) -> str:
    """Serialise one SSE frame; the payload is always a JSON object."""

    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a chat message to the clinic assistant",
    responses={
        422: {"description": "Payload validation failed"},
        503: {"description": "The LLM is not configured or unavailable"},
    },
)
async def chat(
    payload: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    return await service.handle(payload)


@router.post(
    "/stream",
    summary="Stream a chat reply as Server-Sent Events",
    responses={
        200: {
            "description": (
                "SSE frames: delta events, then one done or error event"
            ),
            "content": {"text/event-stream": {}},
        },
        422: {"description": "Payload validation failed"},
        503: {"description": "The LLM is not configured or unavailable"},
    },
)
async def chat_stream(
    payload: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    # Prepare before opening the stream: configuration and patient errors must
    # surface as ordinary JSON responses with a real HTTP status code.
    prepared = service.prepare(payload)

    async def frames():
        async for event in service.stream(prepared):
            yield _sse(event)

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
