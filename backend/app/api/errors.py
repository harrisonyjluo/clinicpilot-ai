"""Translate exceptions into the shared structured error contract."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import AppError
from app.core.middleware import REQUEST_ID_HEADER
from app.schemas.error import ErrorDetail, ErrorResponse

logger = logging.getLogger("clinicpilot.errors")

_HTTP_CODES = {
    400: "BAD_REQUEST",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
}


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details=None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(code=code, message=message, details=details)
    )
    request_id = getattr(request.state, "request_id", None)
    headers = {REQUEST_ID_HEADER: request_id} if request_id else None
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return _error_response(
            request,
            exc.status_code,
            exc.code,
            exc.message,
            exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = [
            {
                "loc": [str(part) for part in error.get("loc", ())],
                "msg": error.get("msg", "Invalid value"),
                "type": error.get("type", "value_error"),
            }
            for error in exc.errors()
        ]
        return _error_response(
            request,
            422,
            "VALIDATION_ERROR",
            "The request payload is invalid.",
            details,
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "HTTP_ERROR")
        return _error_response(
            request,
            exc.status_code,
            code,
            str(exc.detail),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception(
            "unhandled_error method=%s path=%s request_id=%s",
            request.method,
            request.url.path,
            getattr(request.state, "request_id", None),
        )
        return _error_response(
            request,
            500,
            "INTERNAL_ERROR",
            "An unexpected error occurred.",
        )
