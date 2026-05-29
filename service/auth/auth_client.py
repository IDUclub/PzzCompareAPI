"""Keycloak JWT verification client (JWKS signature + claims check).

Adapted from IDUclub ChatStorage's ``AuthenticationClient`` — same flow,
but uses ``httpx`` (already a dependency) instead of ``aiohttp``.

Flow: fetch the realm's JWKS (public keys, cached), match the token's
``kid``, verify the RS256 signature + issuer (and optionally audience),
return the decoded payload. When ``config.verify`` is false, claims are
decoded WITHOUT signature verification (dev/test).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from cachetools import TTLCache
from jose import JWTError, jwt
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from .auth_config import AuthConfig
from .exceptions import (
    AuthDecodeError,
    InvalidAudienceError,
    InvalidTokenSignatureError,
    TokenExpiredError,
)

logger = logging.getLogger("service.auth")

JWKS_CACHE_KEY = "jwks"


class AuthenticationClient:
    """Validate JWT tokens and extract the user id (``sub``) from the payload."""

    RETRIES = 3

    def __init__(self, config: AuthConfig) -> None:
        self.config = config
        self._jwks_cache: TTLCache = TTLCache(maxsize=1, ttl=config.jwks_cache_ttl)
        self._user_cache: TTLCache = TTLCache(
            maxsize=config.user_cache_size, ttl=config.user_cache_ttl
        )
        self._lock = asyncio.Lock()

    @retry(
        stop=stop_after_attempt(RETRIES),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(httpx.TransportError),
    )
    async def _fetch_jwks(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            resp = await client.get(self.config.jwks_url)
            resp.raise_for_status()
            return resp.json()

    async def get_jwks(self) -> dict[str, Any]:
        """Return the cached JWKS document ({"keys": [JWK, ...]})."""
        if JWKS_CACHE_KEY in self._jwks_cache:
            return self._jwks_cache[JWKS_CACHE_KEY]
        async with self._lock:
            if JWKS_CACHE_KEY in self._jwks_cache:
                return self._jwks_cache[JWKS_CACHE_KEY]
            document = await self._fetch_jwks()
            self._jwks_cache[JWKS_CACHE_KEY] = document
            return document

    async def _verify_jwt(self, token: str) -> dict[str, Any]:
        """Full JWT verification (signature + claims)."""
        try:
            jwks_document = await self.get_jwks()

            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            if not kid:
                raise InvalidTokenSignatureError()

            key = next(
                (k for k in jwks_document.get("keys", []) if k.get("kid") == kid),
                None,
            )
            if not key:
                raise InvalidTokenSignatureError()

            payload = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                issuer=self.config.server_url,
                options={"verify_aud": False},
            )

            if self.config.verify_aud:
                audiences = payload.get("aud", [])
                if isinstance(audiences, str):
                    audiences = [audiences]
                elif audiences is None:
                    audiences = []
                if not any(a in self.config.valid_audiences for a in audiences):
                    raise InvalidAudienceError()

            return payload
        except JWTError as exc:
            if "expired" in str(exc).lower():
                raise TokenExpiredError() from exc
            raise InvalidTokenSignatureError() from exc
        except (InvalidTokenSignatureError, InvalidAudienceError, TokenExpiredError):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("token verification failed")
            raise AuthDecodeError() from exc

    async def process_token(self, token: str) -> dict[str, Any]:
        """Verify (or, when disabled, just decode) and return the claims."""
        if self.config.verify:
            return await self._verify_jwt(token)
        try:
            return jwt.get_unverified_claims(token)
        except Exception as exc:  # noqa: BLE001
            raise AuthDecodeError() from exc

    async def get_user_from_token(self, token: str) -> int | str | None:
        """Validate the token and return the user id (``sub`` claim)."""
        cached = self._user_cache.get(token)
        if cached is not None:
            return cached
        payload = await self.process_token(token)
        user_id = payload.get("sub")
        self._user_cache[token] = user_id
        return user_id
