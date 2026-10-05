"""Integration: the service against real PostgreSQL, Redis and MinIO.

Run with ``pytest -m integration``; each test self-skips when its service is not
configured or not reachable, so the unit run (``pytest -m "not integration"``)
and a plain ``pytest`` stay hermetic. The ``integration.yml`` workflow starts
the services and sets:

* ``PZZ_TEST_DATABASE_URL`` — an empty PostgreSQL database (``postgresql+psycopg://…``);
* ``PZZ_TEST_REDIS_URL`` — the Celery broker;
* ``PZZ_TEST_MINIO_ENDPOINT``, ``PZZ_TEST_MINIO_ACCESS_KEY``, ``PZZ_TEST_MINIO_SECRET_KEY``.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.integration


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.skip(f"{name} is not set")
    return value


@pytest.fixture
def pg_engine():
    from sqlalchemy import create_engine, text

    url = _env("PZZ_TEST_DATABASE_URL")
    engine = create_engine(url, future=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("select 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PostgreSQL unavailable: {exc}")
    yield engine
    engine.dispose()


def _alembic(monkeypatch, url: str, command: str, revision: str) -> None:
    from alembic import command as alembic_command
    from alembic.config import Config

    # alembic/env.py takes DATABASE_URL over alembic.ini; conftest points it at SQLite.
    monkeypatch.setenv("DATABASE_URL", url)
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    getattr(alembic_command, command)(config, revision)


def _uniqueness_form(diff) -> bool:
    """An index or unique-constraint diff (compared separately, by column sets)."""
    from sqlalchemy import UniqueConstraint

    op = diff[0] if isinstance(diff, tuple) else diff[0][0]
    if op in ("add_index", "remove_index"):
        return True
    return op in ("add_constraint", "remove_constraint") and isinstance(
        diff[1], UniqueConstraint
    )


def _unique_sets_in_db(engine) -> set[tuple[str, frozenset[str]]]:
    from sqlalchemy import inspect

    inspector = inspect(engine)
    found = set()
    for table in inspector.get_table_names():
        if table == "alembic_version":
            continue
        for constraint in inspector.get_unique_constraints(table):
            found.add((table, frozenset(constraint["column_names"])))
        for index in inspector.get_indexes(table):
            if index["unique"]:
                found.add((table, frozenset(index["column_names"])))
    return found


def _unique_sets_in_models(metadata) -> set[tuple[str, frozenset[str]]]:
    from sqlalchemy import UniqueConstraint

    found = set()
    for table in metadata.tables.values():
        for constraint in table.constraints:
            if isinstance(constraint, UniqueConstraint):
                found.add((table.name, frozenset(c.name for c in constraint.columns)))
        for index in table.indexes:
            if index.unique:
                found.add((table.name, frozenset(c.name for c in index.columns)))
        for column in table.columns:
            if column.unique:
                found.add((table.name, frozenset([column.name])))
    return found


def test_migrations_build_the_models_and_roll_back(pg_engine, monkeypatch):
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from sqlalchemy import inspect

    from service.models import Base

    url = pg_engine.url.render_as_string(hide_password=False)
    _alembic(monkeypatch, url, "downgrade", "base")
    _alembic(monkeypatch, url, "upgrade", "head")
    with pg_engine.connect() as connection:
        context = MigrationContext.configure(
            connection, opts={"compare_type": True}
        )
        drift = compare_metadata(context, Base.metadata)
    # Every model change ships with its migration. Uniqueness is compared by the column
    # sets it covers: the migrations declare a unique constraint beside a plain index
    # where the models declare a unique index — the same rule in another form.
    assert [d for d in drift if not _uniqueness_form(d)] == []
    assert _unique_sets_in_db(pg_engine) == _unique_sets_in_models(Base.metadata)

    _alembic(monkeypatch, url, "downgrade", "base")
    tables = set(inspect(pg_engine).get_table_names()) - {"alembic_version"}
    assert tables == set()
    _alembic(monkeypatch, url, "upgrade", "head")


def test_a_task_goes_through_its_states_in_postgres(pg_engine, monkeypatch):
    from sqlalchemy.orm import Session

    from service.domain.task_state import TaskStatus
    from service.infrastructure.repositories.sqlalchemy_task_repository import (
        SqlAlchemyTaskRepository,
    )

    _alembic(
        monkeypatch,
        pg_engine.url.render_as_string(hide_password=False),
        "upgrade",
        "head",
    )
    external_id = f"it-{uuid.uuid4().hex}"
    with Session(pg_engine) as session:
        repo = SqlAlchemyTaskRepository(session)
        task = repo.create(
            external_id=external_id,
            cadastral_data_path="minio://in/objects.geojson",
            pzz_zones_data_path="minio://in/zones.geojson",
            pzz_zone_vri_labels_path="data/pzz_zone_llm_labels_template.json",
            vri_classifier_path="data/rosreestr_vri_classifier_2024_12_24.json",
            cadastral_vri_col="vri_text",
        )
        session.commit()
        now = datetime.now(timezone.utc)
        repo.update_status(task.id, TaskStatus.running, started_at=now)
        repo.set_result(task.id, "minio://out/result.geojson")
        repo.update_status(task.id, TaskStatus.finished, finished_at=now)
        session.commit()

    with Session(pg_engine) as session:
        stored = SqlAlchemyTaskRepository(session).get_by_external_id(external_id)
        assert stored.status == TaskStatus.finished
        assert stored.result_path == "minio://out/result.geojson"
        assert stored.started_at is not None and stored.finished_at is not None
        session.delete(stored)
        session.commit()


def test_the_celery_broker_accepts_connections():
    from kombu import Connection

    url = _env("PZZ_TEST_REDIS_URL")
    with Connection(url, connect_timeout=5) as connection:
        try:
            connection.ensure_connection(max_retries=1)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Redis unavailable: {exc}")
        queue = connection.SimpleQueue(f"pzz-it-{uuid.uuid4().hex}")
        try:
            queue.put({"task": "ping"})
            message = queue.get(timeout=5)
            assert message.payload == {"task": "ping"}
            message.ack()
        finally:
            queue.clear()
            queue.close()


def test_task_files_round_trip_through_minio(tmp_path):
    from service.infrastructure.storage import MinioStorage, is_remote_path

    endpoint = _env("PZZ_TEST_MINIO_ENDPOINT")
    try:
        storage = MinioStorage(
            endpoint,
            _env("PZZ_TEST_MINIO_ACCESS_KEY"),
            _env("PZZ_TEST_MINIO_SECRET_KEY"),
            bucket="pzz-integration",
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"MinIO unavailable: {exc}")

    source = tmp_path / "zones.geojson"
    source.write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
    stored = storage.upload_file(str(source), f"it/{uuid.uuid4().hex}/zones.geojson")
    assert is_remote_path(stored)
    try:
        assert b"".join(storage.open_stream(stored)) == source.read_bytes()
        copy = storage.download_file(stored, str(tmp_path / "copy.geojson"))
        assert Path(copy).read_bytes() == source.read_bytes()
    finally:
        storage.delete(stored)
    with pytest.raises(Exception):
        storage.open_stream(stored)
