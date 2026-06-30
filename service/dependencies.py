from __future__ import annotations

from functools import lru_cache
from typing import Generator

from fastapi import Depends, FastAPI, Request
from sqlalchemy.orm import Session

from .auth import AuthenticationClient, build_auth_config
from .db import session_scope
from .domain.ports.config_repository import ConfigRepository
from .domain.ports.event_repository import EventRepository
from .domain.ports.task_repository import TaskRepository
from .infrastructure.chat_storage_client import ChatStorageClient
from .infrastructure.ollama_chat_client import OllamaChatClient
from .infrastructure.repositories.sqlalchemy_config_repository import SqlAlchemyConfigRepository
from .infrastructure.repositories.sqlalchemy_event_repository import SqlAlchemyEventRepository
from .infrastructure.repositories.sqlalchemy_task_repository import SqlAlchemyTaskRepository
from .settings import Settings, get_settings


@lru_cache(maxsize=1)
def get_auth_client() -> AuthenticationClient:
    """Process-singleton auth client built from settings (JWKS cache shared)."""
    return AuthenticationClient(build_auth_config(get_settings()))


def build_chat_storage_client(settings: Settings | None = None) -> ChatStorageClient:
    """Build a fresh ChatStorage client. Caller owns it via ``async with``.

    Not a singleton: the client holds an ``httpx.AsyncClient`` bound to the
    request's event loop, mirroring how ``UrbanApiClient`` is instantiated
    per request.
    """
    settings = settings or get_settings()
    return ChatStorageClient(
        base_url=settings.chat_storage_base_url,
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
