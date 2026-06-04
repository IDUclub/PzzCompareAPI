"""FastAPI application root.

Endpoint logic lives in ``service/api/*`` routers, grouped by concern:
``system``, ``classifier`` (task submission), ``tasks`` (task management).
This module only wires the app: startup lifecycle, dependency init,
config defaults, and ``include_router`` calls.
"""
from contextlib import asynccontextmanager

from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from fastapi import FastAPI

from .api import classifier, scenarios, system, tasks
from .api.utils import api_log
from .db import session_scope
from .dependencies import init_dependencies
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
    api_log("startup", "finished")
    yield
    api_log("shutdown", "finished")


app = FastAPI(title="PZZ Pipeline Background Service", lifespan=lifespan)

app.include_router(system.router)
app.include_router(classifier.router)
app.include_router(scenarios.router)
app.include_router(tasks.router)
