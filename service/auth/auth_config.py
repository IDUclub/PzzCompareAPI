"""Auth configuration — Keycloak realm + verification toggles.

Mirrors IDUclub ChatStorage's ``AuthConfig`` so the convention is familiar
across services.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from service.settings import Settings


@dataclass
class AuthConfig:
    """Authentication and token-validation settings."""

    verify: bool  # проверять ли токен вообще
    server_url: str  # https://.../realms/<realm>
    client_id: str = ""
    verify_aud: bool = True
    valid_audiences: list[str] = field(default_factory=list)
    user_cache_ttl: int = 300  # TTL кеша пользователей (сек)
    user_cache_size: int = 10_000  # размер кеша пользователей
    jwks_cache_ttl: int = 600  # TTL кеша JWKS (сек)
    timeout: int = 5

    def __post_init__(self) -> None:
        if self.server_url and not self.server_url.startswith("http"):
            self.server_url = "http://" + self.server_url
        self.server_url = self.server_url.rstrip("/")

    @property
    def jwks_url(self) -> str:
        return f"{self.server_url}/protocol/openid-connect/certs"

    @property
    def token_url(self) -> str:
        return f"{self.server_url}/protocol/openid-connect/token"


def build_auth_config(settings: "Settings") -> AuthConfig:
    """Build an AuthConfig from application settings."""
    raw_aud = (settings.auth_valid_audiences or "").strip()
    audiences = [a.strip() for a in raw_aud.split(",") if a.strip()]
    return AuthConfig(
        verify=settings.auth_verify,
        server_url=settings.auth_server_url,
        client_id=settings.auth_client_id,
        verify_aud=settings.auth_verify_aud,
        valid_audiences=audiences,
        user_cache_ttl=settings.auth_user_cache_ttl,
        user_cache_size=settings.auth_user_cache_size,
        jwks_cache_ttl=settings.auth_jwks_cache_ttl,
        timeout=settings.auth_timeout_seconds,
    )
