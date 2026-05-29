"""Auth-layer exceptions. Mapped to HTTP 401 at the API dependency."""
from __future__ import annotations


class AuthError(Exception):
    """Base class for token-verification failures."""

    detail = "Authentication failed"


class TokenExpiredError(AuthError):
    detail = "Token has expired"


class InvalidTokenSignatureError(AuthError):
    detail = "Invalid token signature"


class InvalidAudienceError(AuthError):
    detail = "Invalid token audience"


class AuthDecodeError(AuthError):
    detail = "Could not decode token"
