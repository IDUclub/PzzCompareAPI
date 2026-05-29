"""Centralised translation of upstream errors into MCP / JSON-RPC errors.

Without this, every tool would need its own ``try/except ApiError`` boilerplate.
With it, tools just call the API client and any non-2xx response surfaces
to the LLM as a structured ``McpError`` with a sensible code and message.

JSON-RPC error code conventions used here:
- ``-32602`` Invalid params (client-side issue: 4xx from upstream)
- ``-32603`` Internal error (server-side issue: 5xx, network)
- ``-32001`` Application error (anything else)

See https://www.jsonrpc.org/specification#error_object.
"""
from __future__ import annotations

from functools import wraps
from typing import Any, Awaitable, Callable

import httpx
from mcp import ErrorData, McpError

from .api_client import ApiError


def _format_body(body: Any) -> str:
    """Best-effort short message from an arbitrary REST body."""
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
    if isinstance(body, str):
        return body.strip() or "(empty body)"
    return repr(body)


def _api_error_to_mcp(exc: ApiError) -> McpError:
    """Map an HTTP-level error from upstream into a JSON-RPC McpError."""
    if exc.status in (401, 403):
        return McpError(ErrorData(
            code=-32002,
            message=(
                "AUTH_TOKEN_EXPIRED: the user's authorization token was rejected "
                f"(HTTP {exc.status}). It is likely expired or invalid. Ask the "
                "frontend/user to provide a fresh Bearer token, then retry the "
                "same call. Do not retry with the old token."
            ),
        ))
    if 400 <= exc.status < 500:
        code = -32602
    elif 500 <= exc.status < 600:
        code = -32603
    else:
        code = -32001
    message = f"upstream API {exc.status}: {_format_body(exc.body)}"
    return McpError(ErrorData(code=code, message=message))


def map_errors(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Decorator: wrap an async MCP tool so upstream errors surface cleanly.

    Catches:
    - ``ApiError`` from our HTTP client → translated by status range.
    - ``httpx.HTTPError`` (connection, timeout) → -32603 Internal error.
    - Anything else → re-raised unchanged (let fastmcp report it).
    """

    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except ApiError as exc:
            raise _api_error_to_mcp(exc) from exc
        except httpx.HTTPError as exc:
            raise McpError(
                ErrorData(code=-32603, message=f"upstream unreachable: {exc}")
            ) from exc

    return wrapper
