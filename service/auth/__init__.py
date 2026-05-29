"""Bearer-token verification (Keycloak JWT via JWKS).

Mirrors the IDUclub convention (see ChatStorage `app/common/auth`): verify
the user's JWT signature + claims against the realm's JWKS before trusting
it. Verification is opt-in via ``AUTH_VERIFY`` so dev/test environments
without a configured realm keep working.
"""
from .auth_config import AuthConfig, build_auth_config
from .auth_client import AuthenticationClient
from .exceptions import (
    AuthDecodeError,
    AuthError,
    InvalidAudienceError,
    InvalidTokenSignatureError,
    TokenExpiredError,
)

__all__ = [
    "AuthConfig",
    "build_auth_config",
    "AuthenticationClient",
    "AuthError",
    "AuthDecodeError",
    "InvalidAudienceError",
    "InvalidTokenSignatureError",
    "TokenExpiredError",
]
