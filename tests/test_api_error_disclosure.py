"""Infrastructure error text must not reach API clients.

`/readiness` is unauthenticated and its broker error carried the Redis URL,
credentials included; the recompute path additionally persisted the broker
error into `error_text` and the task event `details`, both of which the task
API returns. Exception text belongs in the log, not in the response.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import redis
from fastapi.testclient import TestClient

from service import app as app_module
from service.api import system as system_module

SERVICE_ROOT = Path(__file__).resolve().parents[1] / "service"
GUARDED_MODULES = [
    SERVICE_ROOT / "api/tasks.py",
    SERVICE_ROOT / "api/system.py",
]
BROKER_URL = "redis://:s3cr3t-pass@redis-internal.local:6379/0"


def _caught_exception_names(tree: ast.AST) -> set[str]:
    return {
        handler.name
        for handler in ast.walk(tree)
        if isinstance(handler, ast.ExceptHandler) and handler.name
    }


def _referenced_names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _client_facing_values(tree: ast.AST):
    """Yield (label, node) for every call argument the client can read back."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name == "HTTPException":
            for keyword in node.keywords:
                if keyword.arg == "detail":
                    yield "HTTPException detail", keyword.value
        elif name == "append_event":
            for keyword in node.keywords:
                if keyword.arg == "details":
                    yield "event details", keyword.value
        elif name == "set_error":
            for argument in node.args[1:]:
                yield "set_error", argument
            for keyword in node.keywords:
                yield "set_error", keyword.value


@pytest.mark.parametrize("module_path", GUARDED_MODULES, ids=lambda p: p.name)
def test_client_facing_text_never_interpolates_an_exception(module_path: Path) -> None:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    caught = _caught_exception_names(tree)
    assert caught, f"expected {module_path.name} to name caught exceptions"

    offenders = [
        f"{label} at line {value.lineno}"
        for label, value in _client_facing_values(tree)
        if _referenced_names(value) & caught
    ]

    assert not offenders, f"exception text reaches the client: {offenders}"


def _raise_connection_error(*args, **kwargs):
    raise redis.exceptions.ConnectionError(f"Error connecting to {BROKER_URL}.")


def test_readiness_hides_the_broker_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(redis.Redis, "from_url", _raise_connection_error)
    client = TestClient(app_module.app, raise_server_exceptions=False)

    response = client.get("/readiness")

    assert response.status_code == 503
    body = response.text
    assert "s3cr3t-pass" not in body
    assert "redis-internal.local" not in body
    assert response.json()["detail"] == system_module.BROKER_UNAVAILABLE_MESSAGE


class _FailingRedis:
    """`from_url` does not connect; the first command is what fails."""

    def lrange(self, *args, **kwargs):
        _raise_connection_error()


def test_logs_endpoint_hides_the_redis_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(redis.Redis, "from_url", lambda *a, **k: _FailingRedis())
    client = TestClient(app_module.app, raise_server_exceptions=False)

    response = client.get("/logs")

    assert response.status_code == 503
    assert "s3cr3t-pass" not in response.text
    assert response.json()["detail"] == system_module.REDIS_UNAVAILABLE_MESSAGE
