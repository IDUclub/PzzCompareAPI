"""Shared Bearer-token extraction + verification dependency.

Lives in its own module so both ``scenarios`` and ``classifier`` routers can
depend on it without an import cycle (``scenarios`` already imports helpers
from ``classifier``).

Two dependencies are provided:

- ``verify_token`` returns the raw bearer string after verification. Endpoints
  that only forward the *user* token downstream (e.g. to urban_api) keep using
  it unchanged.
- ``get_current_user`` returns an :class:`AuthUser` (raw token + ``user_id``).
  The chat flow needs the ``user_id`` (Keycloak ``sub``): chat history is
  persisted to ChatStorage with OUR service-account token, and the end user is
  named via the ``X-User-Id`` header. The ``user_id`` is captured here, at
  request start, while the user's token is still fresh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..auth.exceptions import AuthError
from ..dependencies import get_auth_client

http_bearer = HTTPBearer()
optional_http_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthUser:
    """Authenticated caller derived from the incoming Keycloak token."""

    token: str
    """Raw bearer string, forwarded downstream (e.g. to urban_api)."""

    user_id: str
    """Keycloak ``sub`` claim — the end-user id (``""`` if unavailable)."""

    claims: dict[str, Any] = field(default_factory=dict)


def _get_token_from_header(credentials: HTTPAuthorizationCredentials) -> str:
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    token = credentials.credentials
    if not token:
        raise HTTPException(
            status_code=400,
            detail="Token is missing in the authorization header",
        )
    return token


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer),
) -> str:
    """Extract the Bearer token and verify it (Keycloak JWT) when enabled.

    When ``AUTH_VERIFY`` is false the token is accepted as-is (urban_api
    validates it downstream). When true, the signature + claims are checked
    against the realm JWKS; a rejected token yields 401 so the caller can
    refresh it.
    """
    token = _get_token_from_header(credentials)
    auth_client = get_auth_client()
    if auth_client.config.verify:
        try:
            await auth_client.get_user_from_token(token)
        except AuthError as exc:
            raise HTTPException(status_code=401, detail=exc.detail) from exc
    return token


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer),
) -> AuthUser:
    """Verify the incoming Keycloak token and return the caller identity.

    Used by the chat endpoints, which need both the raw token and the user's
    ``sub``. When ``AUTH_VERIFY`` is true the signature + claims are checked
    (a rejected token → 401). When false the claims are still decoded (without
    signature verification) so ``user_id`` is available for ChatStorage — the
    same behaviour as the ChatStorage service itself.
    """
    token = _get_token_from_header(credentials)
    auth_client = get_auth_client()
    try:
        claims = await auth_client.process_token(token)
    except AuthError as exc:
        # In verify mode a bad token is rejected. With verification disabled a
        # non-JWT can't be decoded — accept it but without a user id (chat
        # persistence is then skipped for this request).
        if auth_client.config.verify:
            raise HTTPException(status_code=401, detail=exc.detail) from exc
        claims = {}
    return AuthUser(
        token=token,
        user_id=str(claims.get("sub") or ""),
        claims=claims,
    )


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(optional_http_bearer),
) -> AuthUser | None:
    """Identify the caller when a token is present, without demanding one.

    The task-submission endpoints have always accepted anonymous calls; requiring a
    token there now would break existing clients. They still need an identity when one
    is offered, to check that an ``upload_id`` belongs to the caller.
    """
    if credentials is None:
        return None
    return await get_current_user(credentials)
