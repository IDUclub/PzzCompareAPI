"""4xx responses log their detail so the reason shows up in GET /logs."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from service.api import error_logging


@pytest.fixture
def warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    captured: list[str] = []
    monkeypatch.setattr(
        error_logging.logger,
        "warning",
        lambda msg, *args: captured.append(msg % args),
    )
    return captured


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    error_logging.register_client_error_logging(app)

    @app.post("/scenarios/{scenario_id}/chat/stream")
    def chat(scenario_id: int):
        raise HTTPException(status_code=422, detail="Нет функциональных зон")

    @app.get("/server-error")
    def server_error():
        raise HTTPException(status_code=503, detail="down")

    @app.get("/items")
    def items(limit: int):
        return {"limit": limit}

    return TestClient(app)


def test_http_422_detail_is_logged_and_response_unchanged(client, warnings) -> None:
    response = client.post("/scenarios/14656/chat/stream")

    assert response.status_code == 422
    assert response.json() == {"detail": "Нет функциональных зон"}
    assert warnings == [
        "POST /scenarios/14656/chat/stream -> 422: Нет функциональных зон"
    ]


def test_validation_error_is_logged_without_input(client, warnings) -> None:
    response = client.get("/items", params={"limit": "secret-value"})

    assert response.status_code == 422
    assert len(warnings) == 1
    assert warnings[0].startswith("GET /items -> 422: ")
    assert "limit" in warnings[0]
    assert "secret-value" not in warnings[0]


def test_404_and_5xx_are_not_logged(client, warnings) -> None:
    assert client.get("/missing").status_code == 404
    assert client.get("/server-error").status_code == 503
    assert warnings == []


def test_main_app_registers_handlers() -> None:
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from service.app import app

    assert StarletteHTTPException in app.exception_handlers
    assert RequestValidationError in app.exception_handlers
