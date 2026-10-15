"""Aggregated logs: what the Redis sink writes and what GET /logs returns."""

from __future__ import annotations

import json

import pytest
import redis
from fastapi.testclient import TestClient
from loguru import logger

from service import app as app_module
from service import log_sink
from service.logging_config import bind_log_context, reset_log_context, setup_logging


class _FakeRedis:
    """Just enough of redis.Redis for the sink (pipeline) and /logs (lrange)."""

    def __init__(self, entries: list[dict] | None = None) -> None:
        self.items: list[bytes] = [json.dumps(e).encode() for e in entries or []]

    def pipeline(self, transaction: bool = False) -> "_FakeRedis":
        return self

    def lpush(self, key: str, raw: bytes) -> None:
        self.items.insert(0, raw)

    def ltrim(self, key: str, start: int, stop: int) -> None:
        self.items = self.items[start : stop + 1]

    def execute(self) -> None:
        pass

    def lrange(self, key: str, start: int, stop: int) -> list[bytes]:
        return self.items[start : stop + 1]


@pytest.fixture
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> _FakeRedis:
    fake = _FakeRedis()
    monkeypatch.setattr(redis.Redis, "from_url", lambda *a, **k: fake)
    return fake


def test_redis_sink_keeps_full_traceback_and_task_context(fake_redis) -> None:
    setup_logging()
    log_sink.setup_redis_sink(redis_url="redis://fake", service_name="worker")
    token = bind_log_context(task_id="t-1", celery_task_id="c-1")
    try:

        def _inner_step() -> None:
            raise ValueError("bad layer")

        try:
            _inner_step()
        except ValueError:
            logger.exception("pipeline step failed")
        logger.complete()
    finally:
        reset_log_context(token)
        setup_logging()  # drop the sink again

    entry = json.loads(fake_redis.items[0])
    assert entry["service"] == "worker"
    assert entry["level"] == "ERROR"
    assert entry["message"] == "pipeline step failed"
    assert entry["task_id"] == "t-1"
    assert entry["celery_task_id"] == "c-1"
    # enqueue=True used to cut this down to just "ValueError: bad layer".
    assert entry["exception"].startswith("Traceback (most recent call last):")
    assert "_inner_step" in entry["exception"]
    assert entry["exception"].rstrip().endswith("ValueError: bad layer")


def test_redis_sink_carries_subprocess_traceback_text(fake_redis) -> None:
    setup_logging()
    log_sink.setup_redis_sink(redis_url="redis://fake", service_name="worker")
    try:
        logger.bind(
            logger_name="pipeline.stderr", exception_text="Traceback ...\nKeyError: 'x'"
        ).error("pipeline subprocess exited with code 1")
        logger.complete()
    finally:
        setup_logging()

    entry = json.loads(fake_redis.items[0])
    assert entry["logger"] == "pipeline.stderr"
    assert entry["exception"] == "Traceback ...\nKeyError: 'x'"


_ENTRIES = [
    {
        "level": "INFO",
        "service": "api",
        "logger": "uvicorn.access",
        "message": "GET /tasks",
    },
    {
        "level": "ERROR",
        "service": "worker",
        "logger": "pipeline_modules.step",
        "message": "step failed",
        "exception": "Traceback ...\nKeyError: 'vri'",
        "task_id": "t-1",
    },
    {
        "level": "WARNING",
        "service": "worker",
        "logger": "service.tasks",
        "message": "retrying",
        "task_id": "t-2",
    },
    {
        "level": "INFO",
        "service": "worker",
        "logger": "celery.app.trace",
        "message": "Task succeeded",
        "task_id": "t-1",
    },
]


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, ["GET /tasks", "step failed", "retrying", "Task succeeded"]),
        ({"min_level": "warning"}, ["step failed", "retrying"]),
        ({"errors_only": "true"}, ["step failed"]),
        ({"task_id": "t-1"}, ["step failed", "Task succeeded"]),
        ({"logger": "pipeline_modules"}, ["step failed"]),
        ({"q": "keyerror"}, ["step failed"]),
        ({"service": "worker", "limit": 2}, ["step failed", "retrying"]),
        ({"level": "INFO", "service": "worker"}, ["Task succeeded"]),
    ],
)
def test_logs_endpoint_filters(
    monkeypatch: pytest.MonkeyPatch, params: dict, expected: list[str]
) -> None:
    fake = _FakeRedis(_ENTRIES)
    monkeypatch.setattr(redis.Redis, "from_url", lambda *a, **k: fake)
    client = TestClient(app_module.app)

    response = client.get("/logs", params=params)

    assert response.status_code == 200
    assert [e["message"] for e in response.json()] == expected
