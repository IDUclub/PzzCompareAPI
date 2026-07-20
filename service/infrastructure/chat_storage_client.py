"""Async HTTP client for the IDUclub ChatStorage service.

ChatStorage is a tiny FastAPI + MongoDB service that persists per-user
assistant chat history.

Authentication is **machine-to-machine**: every request carries OUR service
account's Keycloak token (client_credentials, obtained via ``idu-service-auth``)
plus an ``X-User-Id`` header naming the end user. ChatStorage recognizes the
trusted service account and stores/reads history under the supplied
``X-User-Id`` (the user's Keycloak ``sub``). We use the service token — not the
user's — because the user's token expires during a long computation, by which
point we still need to persist the chat.

Contract (see IDUclub/ChatStorage docs/frontend-chat-history.md):

- ``POST /api/v1/chat_history/create_chat`` — body ``{title, scenario_id,
  project_id, metadata}`` → ``ChatSummary`` (201).
- ``POST /api/v1/chat_history/{chat_id}/message`` — body ``{role, content,
  metadata}`` (simple text) or ``{role, parts, metadata}`` (explicit parts)
  → ``Message`` (201).
- ``GET /api/v1/chat_history/{chat_id}`` — full ``Chat`` with messages.

Mirrors the style of ``urban_api_client.UrbanApiClient``: one instance per
request, used as an async context manager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from idu_service_auth import KeycloakTokenClient

_API_PREFIX = "/api/v1/chat_history"


class ChatStorageError(RuntimeError):
    """Non-2xx response (or transport error) from ChatStorage."""

    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self.body = body
        super().__init__(f"chat_storage returned {status}: {body!r}")


class ChatStorageClient:
    """Thin async wrapper. One instance per request.

    Auth is per-call: the service bearer token (fetched from ``token_client``,
    which refreshes it) plus the end-user id in ``X-User-Id``.
    """

    def __init__(
        self,
        base_url: str,
        token_client: "KeycloakTokenClient",
        *,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not base_url:
            raise RuntimeError(
                "chat_storage_base_url is not configured. Set "
                "CHAT_STORAGE_BASE_URL to enable chat history persistence."
            )
        if token_client is None:
            raise RuntimeError("A Keycloak service token client is required.")
        self._token_client = token_client
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds)

    async def __aenter__(self) -> "ChatStorageClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._client.aclose()

    async def _auth_headers(self, user_id: str) -> dict[str, str]:
        """Service bearer token + the end-user id for ChatStorage."""
        # Import lazily so the module stays importable (and unit-testable with a
        # fake token client) without the optional auth library installed.
        try:
            from idu_service_auth import KeycloakAuthError
        except ImportError:
            KeycloakAuthError = ()  # nothing to catch when the lib is absent

        try:
            headers = dict(await self._token_client.get_authorization_headers())
        except KeycloakAuthError as exc:  # Keycloak unavailable / bad credentials
            raise ChatStorageError(0, f"service token unavailable: {exc}") from exc
        headers["X-User-Id"] = str(user_id)
        return headers

    async def create_chat(
        self,
        user_id: str,
        *,
        title: str | None = None,
        scenario_id: str | int | None = None,
        project_id: str | int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an empty chat and return its ``ChatSummary`` (incl. ``chat_id``).

        The chat is owned by ``user_id`` (sent as ``X-User-Id``), not by the
        service account whose token authenticates the call.
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
            headers=await self._auth_headers(user_id),
        )
        return self._json_or_raise(resp)

    async def add_message(
        self,
        user_id: str,
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
            headers=await self._auth_headers(user_id),
        )
        return self._json_or_raise(resp)

    async def get_chat(self, user_id: str, chat_id: str) -> dict[str, Any]:
        """Return the full chat (``ChatSummary`` + ordered ``messages``)."""
        resp = await self._client.get(
            f"{_API_PREFIX}/{chat_id}",
            headers=await self._auth_headers(user_id),
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
