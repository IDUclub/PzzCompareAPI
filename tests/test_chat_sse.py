"""Tests for the gMART-format SSE mapping and chat endpoint auth (phase 4)."""
import json

import pytest
from fastapi import HTTPException

from service.api.security import _get_token_from_header
from service.api.tasks import _chat_event_to_sse, _final_answer_chunk_sse


def _payload(sse):
    return sse.event, json.loads(sse.data)


def test_chat_created_maps_to_service_event() -> None:
    sse = _chat_event_to_sse(
        {"type": "chat_created", "chat_id": "c-1", "title": "Заголовок"}
    )
    event, data = _payload(sse)
    assert event == "service_event"
    assert data["type"] == "service_event"
    assert data["content"]["event_type"] == "storage_event"
    inner = data["content"]["event"]
    assert inner == {
        "storage_event_type": "chat_created",
        "chat_id": "c-1",
        "chat_title": "Заголовок",
    }


def test_token_maps_to_ollama_like_chunk() -> None:
    sse = _chat_event_to_sse({"type": "token", "content": "Привет"})
    event, data = _payload(sse)
    assert event == "chunk"
    assert data == {"type": "chunk", "content": {"text": "Привет", "done": False}}


def test_error_maps_to_error_envelope() -> None:
    sse = _chat_event_to_sse({"type": "error", "stage": "llm", "detail": "boom"})
    _, data = _payload(sse)
    assert data == {
        "type": "error",
        "content": {"message": "boom", "stage": "llm"},
    }


def test_done_event_is_not_emitted_as_sse() -> None:
    assert _chat_event_to_sse({"type": "done", "chat_id": "c-1"}) is None


def test_final_answer_chunk_marks_done() -> None:
    _, data = _payload(_final_answer_chunk_sse())
    assert data == {"type": "chunk", "content": {"text": "", "done": True}}


def test_missing_bearer_token_rejected() -> None:
    with pytest.raises(HTTPException) as exc:
        _get_token_from_header(None)
    assert exc.value.status_code == 401


def test_scenario_chat_stream_requires_auth() -> None:
    """The chat endpoint depends on verify_token — no Bearer header is rejected."""
    from fastapi.testclient import TestClient

    from service import app as app_module

    client = TestClient(app_module.app)
    resp = client.post(
        "/scenarios/1/chat/stream",
        data={"user_query": "q", "year": 2026, "source": "User"},
    )
    assert resp.status_code in (401, 403)
