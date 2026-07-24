from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Generator

from fastapi import Depends, FastAPI, Request
from sqlalchemy.orm import Session

from .auth import AuthenticationClient, build_auth_config
from .db import session_scope

if TYPE_CHECKING:
    from idu_service_auth import KeycloakTokenClient, KeycloakTokenConfig

logger = logging.getLogger("service.auth")
from .domain.ports.config_repository import ConfigRepository
from .domain.ports.event_repository import EventRepository
from .domain.ports.task_repository import TaskRepository
from .infrastructure.chat_storage_client import ChatStorageClient
from .infrastructure.ollama_chat_client import OllamaChatClient
from .infrastructure.repositories.sqlalchemy_config_repository import (
    SqlAlchemyConfigRepository,
)
from .infrastructure.repositories.sqlalchemy_event_repository import (
    SqlAlchemyEventRepository,
)
from .infrastructure.repositories.sqlalchemy_task_repository import (
    SqlAlchemyTaskRepository,
)
from .settings import Settings, get_settings


@lru_cache(maxsize=1)
def get_auth_client() -> AuthenticationClient:
    """Process-singleton auth client built from settings (JWKS cache shared)."""
    return AuthenticationClient(build_auth_config(get_settings()))


# ── Keycloak service token (outbound M2M auth) ───────────────────────────────
#
# ChatStorage is called with OUR service-account (client_credentials) token, not
# the user's token: the user's token expires during long computations, so by the
# time we persist the chat it would be rejected. The service token is obtained
# with the ``idu-service-auth`` library and shared for the whole process (created
# in the FastAPI lifespan, see ``service/app.py``); the end user is identified to
# ChatStorage via an ``X-User-Id`` header.

_service_token_client: "KeycloakTokenClient | None" = None


def keycloak_service_configured(settings: Settings | None = None) -> bool:
    """True when the Keycloak service-account credentials are fully configured."""
    settings = settings or get_settings()
    return all(
        (
            settings.keycloak_url,
            settings.keycloak_realm,
            settings.keycloak_client_id,
            settings.keycloak_client_secret,
        )
    )


def build_keycloak_token_config(
    settings: Settings | None = None,
) -> "KeycloakTokenConfig":
    """Build the ``idu-service-auth`` config from application settings."""
    from idu_service_auth import KeycloakTokenConfig

    settings = settings or get_settings()
    return KeycloakTokenConfig(
        auth_server_url=settings.keycloak_url,
        realm=settings.keycloak_realm,
        client_id=settings.keycloak_client_id,
        client_secret=settings.keycloak_client_secret,
        scope=settings.keycloak_scope or None,
        background_refresh=True,
    )


def set_service_token_client(client: "KeycloakTokenClient | None") -> None:
    """Register the process-wide service token client (called from lifespan)."""
    global _service_token_client
    _service_token_client = client


def get_service_token_client() -> "KeycloakTokenClient | None":
    """Return the process-wide service token client, or None if not initialized."""
    return _service_token_client


def build_chat_storage_client(
    settings: Settings | None = None,
) -> ChatStorageClient | None:
    """Build a fresh ChatStorage client, or None when persistence is unavailable.

    Not a singleton: the client holds an ``httpx.AsyncClient`` bound to the
    request's event loop, mirroring how ``UrbanApiClient`` is instantiated per
    request. ChatStorage is called with the process-wide Keycloak service token
    client (M2M). Returns None when either the ChatStorage URL or the service
    token client is unavailable — persistence is then disabled rather than
    falling back to the (possibly expired) user token.
    """
    settings = settings or get_settings()
    if not settings.chat_storage_base_url:
        return None
    token_client = get_service_token_client()
    if token_client is None:
        logger.warning(
            "CHAT_STORAGE_BASE_URL is set but the Keycloak service token client "
            "is not initialized; chat history persistence is disabled."
        )
        return None
    return ChatStorageClient(
        base_url=settings.chat_storage_base_url,
        token_client=token_client,
        timeout_seconds=settings.chat_storage_timeout_seconds,
    )


def build_ollama_chat_client(settings: Settings | None = None) -> OllamaChatClient:
    """Build a fresh streaming Ollama chat client. Caller owns it via ``async with``.

    Mirrors gMART: one Ollama host (``ollama_base_url``); the model is chosen
    per request, with ``chat_model`` (or ``generate_model``) as the default.
    """
    settings = settings or get_settings()
    return OllamaChatClient(
        base_url=settings.ollama_base_url,
        default_model=settings.chat_model or settings.generate_model,
        timeout_seconds=settings.chat_request_timeout_seconds,
        temperature=settings.chat_temperature,
    )


def init_dependencies(app: FastAPI) -> None:
    """Store app-wide singletons in FastAPI state on startup."""
    app.state.settings = get_settings()


def get_app_settings(request: Request) -> Settings:
    """Return the effective settings, including any live runtime overrides.

    Delegates to ``get_settings`` (TTL-gated override sync + cached build) rather
    than the startup snapshot in ``app.state.settings``, so config changed via the
    admin API takes effect on subsequent requests without a redeploy.
    """
    return get_settings()


def get_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session for the duration of a single request.

    FastAPI caches this dependency within a request, so all repos injected
    into the same handler share the same session and transaction.
    """
    with session_scope() as session:
        yield session


def get_task_repo(session: Session = Depends(get_db)) -> TaskRepository:
    return SqlAlchemyTaskRepository(session)


def get_event_repo(session: Session = Depends(get_db)) -> EventRepository:
    return SqlAlchemyEventRepository(session)


def get_config_repo(session: Session = Depends(get_db)) -> ConfigRepository:
    return SqlAlchemyConfigRepository(session)
