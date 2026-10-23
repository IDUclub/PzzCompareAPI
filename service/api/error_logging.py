"""Log the reason behind 4xx responses.

Uvicorn's access log only records the status code, so a 422 from a request
check (e.g. "urban_api returned no functional zones") left no trace in
``GET /logs``. These handlers log the ``detail`` and then defer to FastAPI's
default handlers, so the response the client gets is unchanged.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("service.http_errors")

# Statuses that carry no useful detail and would only add noise.
_SKIPPED_STATUSES = {404, 405}
_MAX_DETAIL_CHARS = 2000


def _detail_text(detail: Any) -> str:
    if isinstance(detail, str):
        text = detail
    else:
        text = json.dumps(detail, ensure_ascii=False, default=str)
    if len(text) > _MAX_DETAIL_CHARS:
        text = text[:_MAX_DETAIL_CHARS] + "…"
    return text


def _validation_errors_for_log(exc: RequestValidationError) -> list[dict[str, Any]]:
    # Drop "input": it echoes the request payload (geometries, user files).
    return [
        {key: value for key, value in error.items() if key in ("type", "loc", "msg")}
        for error in exc.errors()
    ]


def _log_client_error(request: Request, status_code: int, detail: Any) -> None:
    logger.warning(
        "%s %s -> %s: %s",
        request.method,
        request.url.path,
        status_code,
        _detail_text(detail),
    )


async def _logging_http_exception_handler(
    request: Request, exc: StarletteHTTPException
):
    if 400 <= exc.status_code < 500 and exc.status_code not in _SKIPPED_STATUSES:
        _log_client_error(request, exc.status_code, exc.detail)
    return await http_exception_handler(request, exc)


async def _logging_validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    _log_client_error(request, 422, _validation_errors_for_log(exc))
    return await request_validation_exception_handler(request, exc)


def register_client_error_logging(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, _logging_http_exception_handler)
    app.add_exception_handler(
        RequestValidationError, _logging_validation_exception_handler
    )
