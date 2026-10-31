"""Request-scoped logging and request ID middleware."""

import logging
import time
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from app.schemas.error import ErrorDetail, ErrorResponse

logger = logging.getLogger("clinicpilot.request")
REQUEST_ID_HEADER = "X-Request-ID"


def _internal_error_response(request_id: str) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(
            code="INTERNAL_ERROR",
            message="An unexpected error occurred.",
        )
    )
    return JSONResponse(
        status_code=500,
        content=body.model_dump(mode="json"),
        headers={REQUEST_ID_HEADER: request_id},
    )


async def request_context_middleware(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    """Attach a request ID, log the request outcome, and never log patient text."""

    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.exception(
            "request_failed method=%s path=%s request_id=%s duration_ms=%s",
            request.method,
            request.url.path,
            request_id,
            duration_ms,
        )
        return _internal_error_response(request_id)

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers[REQUEST_ID_HEADER] = request_id
    logger.info(
        "request_completed method=%s path=%s status=%s request_id=%s duration_ms=%s",
        request.method,
        request.url.path,
        response.status_code,
        request_id,
        duration_ms,
    )
    return response
