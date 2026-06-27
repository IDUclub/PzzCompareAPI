"""Async HTTP client for the IDUclub ChatStorage service.

ChatStorage is a tiny FastAPI + MongoDB service that persists per-user
assistant chat history. The user is derived on its side from the Bearer
token (``sub`` claim) — we never pass ``user_id`` ourselves, we just
forward the incoming ``Authorization: Bearer ...`` header verbatim, the
same way ``urban_api_client`` does.

Contract (see IDUclub/ChatStorage docs/frontend-chat-history.md):

- ``POST /api/v1/chat_history/create_chat`` — body ``{title, scenario_id,
  project_id, metadata}`` → ``ChatSummary`` (201).
- ``POST /api/v1/chat_history/{chat_id}/message`` — body ``{role, content,
  metadata}`` (simple text) or ``{role, parts, metadata}`` (explicit parts)
  → ``Message`` (201).
- ``GET /api/v1/chat_history/{chat_id}`` — full ``Chat`` with messages.

Mirrors the style of ``urban_api_client.UrbanApiClient``: one instance per
request, used as an async context manager, auth header passed per call.
"""
from __future__ import annotations

from typing import Any

import httpx

_API_PREFIX = "/api/v1/chat_history"


class ChatStorageError(RuntimeError):
    """Non-2xx response from ChatStorage."""

    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self.body = body
        super().__init__(f"chat_storage returned {status}: {body!r}")


class ChatStorageClient:
    """Thin async wrapper. One instance per request — auth header is per-call."""

    def __init__(self, base_url: str, timeout_seconds: float = 10.0) -> None:
        if not base_url:
            raise RuntimeError(
                "chat_storage_base_url is not configured. Set "
                "CHAT_STORAGE_BASE_URL to enable chat history persistence."
            )
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds)

    async def __aenter__(self) -> "ChatStorageClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._client.aclose()

    @staticmethod
    def _auth_headers(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    async def create_chat(
        self,
        token: str,
        *,
        title: str | None = None,
        scenario_id: str | int | None = None,
        project_id: str | int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an empty chat and return its ``ChatSummary`` (incl. ``chat_id``).

        The owning user is taken from the token on the ChatStorage side.
        """
        payload = {
            "title": title,
            "scenario_id": scenario_id,
            "project_id": project_id,
            "metadata": metadata or {},
        }
        # Drop keys ChatStorage treats as "unset" so it applies its own
        # defaults instead of storing explicit nulls.
        payload = {k: v for k, v in payload.items() if v is not None}
        resp = await self._client.post(
            f"{_API_PREFIX}/create_chat",
            json=payload,
            headers=self._auth_headers(token),
        )
        return self._json_or_raise(resp)

    async def add_message(
        self,
        token: str,
        chat_id: str,
        *,
        role: str,
        content: str | None = None,
        parts: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a message to a chat and return the stored ``Message``.

        Provide either ``content`` (simple text message) or ``parts`` (explicit
        multi-part message) — ChatStorage requires exactly one of them.
        """
        payload: dict[str, Any] = {"role": role, "metadata": metadata or {}}
        if parts is not None:
            payload["parts"] = parts
        else:
            payload["content"] = content
        resp = await self._client.post(
            f"{_API_PREFIX}/{chat_id}/message",
            json=payload,
            headers=self._auth_headers(token),
        )
        return self._json_or_raise(resp)

    async def get_chat(self, token: str, chat_id: str) -> dict[str, Any]:
        """Return the full chat (``ChatSummary`` + ordered ``messages``)."""
        resp = await self._client.get(
            f"{_API_PREFIX}/{chat_id}",
            headers=self._auth_headers(token),
        )
        return self._json_or_raise(resp)

    @staticmethod
    def _json_or_raise(resp: httpx.Response) -> Any:
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except ValueError:
                body = resp.text
            raise ChatStorageError(resp.status_code, body)
        return resp.json()
