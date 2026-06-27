"""Shared Bearer-token extraction + verification dependency.

Lives in its own module so both ``scenarios`` and ``classifier`` routers can
depend on it without an import cycle (``scenarios`` already imports helpers
from ``classifier``).
"""
from __future__ import annotations

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..auth.exceptions import AuthError
from ..dependencies import get_auth_client

http_bearer = HTTPBearer()


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
