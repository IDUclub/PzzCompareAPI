"""FastAPI application root.

Endpoint logic lives in ``service/api/*`` routers, grouped by concern:
``system``, ``classifier`` (task submission), ``tasks`` (task management).
This module only wires the app: startup lifecycle, dependency init,
config defaults, and ``include_router`` calls.
"""
import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager

from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import admin_config, classifier, scenarios, system, tasks
from .api.utils import api_log
from .db import session_scope
from .dependencies import (
    build_keycloak_token_config,
    init_dependencies,
    keycloak_service_configured,
    set_service_token_client,
)
from .infrastructure.repositories.sqlalchemy_config_repository import SqlAlchemyConfigRepository
from .log_sink import setup_redis_sink
from .logging_config import setup_logging
from .settings import get_settings

setup_logging()


def _run_migrations() -> None:
    """Apply pending Alembic migrations at startup.

    Using ``alembic upgrade head`` instead of ``Base.metadata.create_all``
    ensures schema changes are applied incrementally without manual ALTER
    TABLE statements or database drops. Alembic acquires an advisory lock,
    so multiple replicas starting simultaneously won't conflict.
    """
    import pathlib
    project_root = pathlib.Path(__file__).resolve().parent.parent
    alembic_cfg = AlembicConfig(str(project_root / "alembic.ini"))
    alembic_command.upgrade(alembic_cfg, "head")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_redis_sink(redis_url=get_settings().redis_url)
    init_dependencies(app)
    if get_settings().run_migrations_on_startup:
        _run_migrations()
    with session_scope() as session:
        config_repo = SqlAlchemyConfigRepository(session)
        defaults = (
            ("priority_current_sum", "0", "int"),
            ("priority_max_sum", "20", "int"),
        )
        for name, value, py_type in defaults:
            if config_repo.get(name) is None:
                config_repo.set(name, value, py_type)

    async with AsyncExitStack() as stack:
        # Shared Keycloak service-token client for outbound M2M auth to
        # ChatStorage. Created once per process; background refresh keeps the
        # token fresh so chat history persists even after the user's own token
        # has expired mid-computation.
        if keycloak_service_configured():
            from idu_service_auth import KeycloakTokenClient

            token_client = await stack.enter_async_context(
                KeycloakTokenClient(build_keycloak_token_config())
            )
            set_service_token_client(token_client)
            logging.getLogger("service.auth").info(
                "Keycloak service token client initialized."
            )
        else:
            logging.getLogger("service.auth").warning(
                "Keycloak service credentials are not fully configured "
                "(KEYCLOAK_URL/REALM/CLIENT_ID/CLIENT_SECRET); ChatStorage "
                "history persistence is disabled."
            )
        api_log("startup", "finished")
        try:
            yield
        finally:
            set_service_token_client(None)
            api_log("shutdown", "finished")


app = FastAPI(title="PZZ Pipeline Background Service", lifespan=lifespan)

_cors_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "*")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(classifier.router)
app.include_router(scenarios.router)
app.include_router(tasks.router)
app.include_router(admin_config.router)
