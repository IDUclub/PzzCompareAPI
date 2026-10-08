"""Async HTTP client for the zone regulations NormGraph reads from ПЗЗ documents.

NormGraph (IDUclub/NormGraph) parses the land-use and development rules uploaded to
IDU_DVD into zones: permitted uses (ВРИ codes per section) and limit parameters
(height, floors, coverage, …). Each ПЗЗ is tagged with the Urban API territory it
belongs to.

- ``GET /regulations/documents?territory_id=…`` — ПЗЗ documents of those territories.
- ``GET /regulations/zones?doc_id=…`` — the zones of one document.

NormGraph accepts only service-account tokens, so every request carries OUR
Keycloak token (client_credentials via ``idu-service-auth``), never the user's.
One instance per request, used as an async context manager, like
``UrbanApiClient``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from idu_service_auth import KeycloakTokenClient

# NormGraph caps a zone listing at 2000; a ПЗЗ has a few hundred zones at most.
_ZONES_LIMIT = 2000


class NormGraphError(RuntimeError):
    """Non-2xx response (or a missing service token) from NormGraph."""

    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self.body = body
        super().__init__(f"normgraph returned {status}: {body!r}")


class NormGraphClient:
    def __init__(
        self,
        base_url: str,
        token_client: "KeycloakTokenClient",
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not base_url:
            raise RuntimeError(
                "normgraph_base_url is not configured. Set NORMGRAPH_BASE_URL "
                "to check scenarios against the ПЗЗ of their territory."
            )
        if token_client is None:
            raise RuntimeError("A Keycloak service token client is required.")
        self._token_client = token_client
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds)

    async def __aenter__(self) -> "NormGraphClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._client.aclose()

    async def _auth_headers(self) -> dict[str, str]:
        try:
            from idu_service_auth import KeycloakAuthError
        except ImportError:
            KeycloakAuthError = ()  # nothing to catch when the lib is absent

        try:
            return dict(await self._token_client.get_authorization_headers())
        except KeycloakAuthError as exc:
            raise NormGraphError(0, f"service token unavailable: {exc}") from exc

    async def regulation_documents(
        self, territory_ids: list[int]
    ) -> list[dict[str, Any]]:
        """ПЗЗ documents tagged with any of ``territory_ids``, with their zone counts."""
        resp = await self._client.get(
            "/regulations/documents",
            params=[("territory_id", t) for t in territory_ids],
            headers=await self._auth_headers(),
        )
        return self._json_or_raise(resp)

    async def zones(self, doc_id: str) -> list[dict[str, Any]]:
        """The zones of one ПЗЗ document: uses, parameters and the document itself."""
        resp = await self._client.get(
            "/regulations/zones",
            params={"doc_id": doc_id, "limit": _ZONES_LIMIT},
            headers=await self._auth_headers(),
        )
        return self._json_or_raise(resp).get("zones") or []

    @staticmethod
    def _json_or_raise(resp: httpx.Response) -> Any:
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except ValueError:
                body = resp.text
            raise NormGraphError(resp.status_code, body)
        return resp.json()
