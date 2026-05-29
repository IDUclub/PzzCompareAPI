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
from .infrastructure.repositories.sqlalchemy_config_repository import SqlAlchemyConfigRepository
from .infrastructure.repositories.sqlalchemy_event_repository import SqlAlchemyEventRepository
from .infrastructure.repositories.sqlalchemy_task_repository import SqlAlchemyTaskRepository
from .settings import Settings, get_settings


@lru_cache(maxsize=1)
def get_auth_client() -> AuthenticationClient:
    """Process-singleton auth client built from settings (JWKS cache shared)."""
    return AuthenticationClient(build_auth_config(get_settings()))


def init_dependencies(app: FastAPI) -> None:
    """Store app-wide singletons in FastAPI state on startup."""
    app.state.settings = get_settings()


def get_app_settings(request: Request) -> Settings:
    """Return settings from application state (one instance per process)."""
    return request.app.state.settings


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
