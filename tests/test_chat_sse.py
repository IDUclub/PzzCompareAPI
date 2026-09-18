"""Tests for the gMART-format SSE mapping and chat endpoint auth (phase 4)."""

import asyncio
import json

import httpx
import pytest
from fastapi import HTTPException
from sse_starlette.sse import ServerSentEvent

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


def test_warning_maps_to_warning_envelope() -> None:
    sse = _chat_event_to_sse(
        {
            "type": "warning",
            "stage": "create_chat",
            "detail": "chat_storage returned 401: Token expired",
            "message": "Ответ сформирован, но не сохранён в историю чата.",
        }
    )
    event, data = _payload(sse)
    assert event == "warning"
    assert data["type"] == "warning"
    assert data["content"]["stage"] == "create_chat"
    assert (
        data["content"]["message"]
        == "Ответ сформирован, но не сохранён в историю чата."
    )
    assert "401" in data["content"]["detail"]


def test_done_event_is_not_emitted_as_sse() -> None:
    assert _chat_event_to_sse({"type": "done", "chat_id": "c-1"}) is None


def test_final_answer_chunk_marks_done() -> None:
    _, data = _payload(_final_answer_chunk_sse())
    assert data == {"type": "chunk", "content": {"text": "", "done": True}}


def test_stream_chat_answer_managed_surfaces_error_instead_of_raising(
    monkeypatch,
) -> None:
    """A failure mid-stream must yield an ``error`` event, not propagate.

    The caller appends a terminal ``done`` after draining this generator; an
    escaping exception would abort the SSE stream before ``done`` and hang the
    frontend.
    """
    from types import SimpleNamespace

    from service.api import tasks as tasks_module

    class _DummyClientCM:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(
        tasks_module, "build_chat_llm_client", lambda settings: _DummyClientCM()
    )
    monkeypatch.setattr(tasks_module, "load_system_prompt", lambda path: "sys")

    async def _boom(**kwargs):
        """Async generator matching stream_chat_answer that raises on iteration.

        The trailing ``yield`` is what makes this an async generator; the raise
        fires on the first iteration.
        """
        raise httpx.ConnectError("[Errno -2] Name or service not known")
        yield

    monkeypatch.setattr(tasks_module, "stream_chat_answer", _boom)

    app_settings = SimpleNamespace(chat_system_prompt_path="x")

    async def run():
        events = []
        async for ev in tasks_module._stream_chat_answer_managed(
            app_settings,
            user_id=None,
            user_query="q",
            classification_context="ctx",
            chat_id=None,
        ):
            events.append(ev)
        return events

    events = asyncio.run(run())
    assert events == [
        {
            "type": "error",
            "stage": "chat_answer",
            "detail": "Не удалось сформировать ответ по результатам классификации.",
        }
    ]


def test_task_chat_stream_always_emits_terminal_done(monkeypatch) -> None:
    """A crash inside the poll loop must still end with ``error`` + ``done``.

    An exception escaping the SSE generator closes the connection with no
    terminal event, and the frontend then reports the whole check as failed even
    when the result layer was already delivered.
    """
    from types import SimpleNamespace

    from service.api import tasks as tasks_module

    def _boom_session_scope():
        raise RuntimeError("database is down")

    monkeypatch.setattr(tasks_module, "session_scope", _boom_session_scope)

    request = SimpleNamespace(is_disconnected=lambda: asyncio.sleep(0, result=False))

    async def run():
        events = []
        async for sse in tasks_module.task_stream_with_chat_generator(
            "ext-1",
            group_by="zone",
            poll_interval=0.01,
            request=request,
            app_settings=SimpleNamespace(),
            initial={"external_id": "ext-1"},
            user_id=None,
            user_query="q",
        ):
            events.append(_payload(sse))
        return events

    events = asyncio.run(run())
    assert [event for event, _ in events] == ["task", "error", "done"]
    assert events[1][1]["content"]["stage"] == "stream"
    assert events[2][1] == {"status": "unknown", "chat_id": None}


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


def test_classify_only_chat_stream_requires_auth() -> None:
    """The classify-only chat endpoint also depends on verify_token."""
    from fastapi.testclient import TestClient

    from service import app as app_module

    client = TestClient(app_module.app)
    resp = client.post(
        "/tasks/classify-only/chat/stream",
        data={"user_query": "q", "cadastral_vri_col": "vri"},
    )
    assert resp.status_code in (401, 403)


def _classify_geojson():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": None,
                "properties": {
                    "ВРИ_ЕГРН": "Магазин",
                    "Топ1_возможный_ВРИ": "4.4 Магазины",
                    "Топ5_возможных_ВРИ": "4.4 Магазины, 4.6 Общепит",
                    "Причина": "string-match + embed",
                },
            },
            {
                "type": "Feature",
                "geometry": None,
                "properties": {
                    "ВРИ_ЕГРН": "Нечто непонятное",
                    "Топ1_возможный_ВРИ": None,
                    "Топ5_возможных_ВРИ": None,
                    "Причина": "Проверка по ПЗЗ отключена.",
                },
            },
        ],
    }


def test_build_classify_summary_response(monkeypatch) -> None:
    from types import SimpleNamespace

    from service.api import tasks as tasks_module

    monkeypatch.setattr(
        tasks_module, "_load_result_geojson", lambda *a, **k: _classify_geojson()
    )
    task = SimpleNamespace(status="finished", result_path="result.geojson")
    app_settings = SimpleNamespace(outputs_dir="/tmp")

    report = tasks_module.build_classify_summary_response(task, "ext-1", app_settings)

    assert report["task_external_id"] == "ext-1"
    assert report["summary"] == {
        "total": 2,
        "with_candidate": 1,
        "without_candidate": 1,
    }
    assert report["objects"][0]["matched_vri"] == "4.4 Магазины"
    assert report["objects"][0]["fit"] == "matched"
    assert report["objects"][1]["matched_vri"] is None
    assert report["objects"][1]["fit"] == "unclear"
    assert "Классифицировано объектов: 2." in report["chat_message"]
    assert "С подобранным ВРИ: 1." in report["chat_message"]


def _geo_layer(name: str, role: str) -> dict:
    return {
        "name": name,
        "title": name,
        "role": role,
        "url": f"https://pzz.example/files/{name}/ext-1",
        "download_url": None,
        "filename": f"{name}.geojson",
        "mime_type": "application/geo+json",
        "source_service": "pzz",
    }


def _run_finished_chat_stream(monkeypatch, result_layers, **generator_kwargs):
    """Drive ``task_stream_with_chat_generator`` over an already finished task.

    Returns the emitted ``file`` layer names and the file parts handed to chat
    history persistence.
    """
    from contextlib import contextmanager
    from types import SimpleNamespace

    from service.api import tasks as tasks_module
    from service.models import TaskStatus

    task = SimpleNamespace(id=1, status=TaskStatus.finished)

    class _Result:
        def scalar_one_or_none(self):
            return task

        def scalars(self):
            return SimpleNamespace(all=lambda: [])

    @contextmanager
    def _session_scope():
        yield SimpleNamespace(execute=lambda stmt: _Result())

    persisted: dict = {}

    async def _fake_chat_answer(app_settings, **kwargs):
        persisted["file_parts"] = kwargs["assistant_file_parts"]
        yield {"type": "done", "chat_id": "chat-1"}

    monkeypatch.setattr(tasks_module, "session_scope", _session_scope)
    monkeypatch.setattr(
        tasks_module,
        "TaskOut",
        SimpleNamespace(
            model_validate=lambda t: SimpleNamespace(model_dump=lambda mode: {})
        ),
    )
    monkeypatch.setattr(
        tasks_module, "build_result_geo_layers", lambda *a, **k: result_layers
    )
    monkeypatch.setattr(tasks_module, "_stream_chat_answer_managed", _fake_chat_answer)

    request = SimpleNamespace(is_disconnected=lambda: asyncio.sleep(0, result=False))

    async def run():
        events = []
        async for sse in tasks_module.task_stream_with_chat_generator(
            "ext-1",
            group_by="zone",
            poll_interval=0.01,
            request=request,
            app_settings=SimpleNamespace(),
            initial={"external_id": "ext-1"},
            user_id=None,
            user_query="q",
            include_report=False,
            **generator_kwargs,
        ):
            events.append(_payload(sse))
        return events

    events = asyncio.run(run())
    streamed = [data["content"]["name"] for event, data in events if event == "file"]
    return streamed, persisted["file_parts"]


def test_task_chat_stream_persists_input_and_result_layers(monkeypatch) -> None:
    """Chat history must keep the uploaded input layers, not only the result.

    The frontend rebuilds the map from the persisted ``file`` parts when a chat
    is reopened, so a layer missing here disappears from the reopened chat.
    """
    from service.api import tasks as tasks_module

    input_layers = [
        _geo_layer("input_cadastral", "input"),
        _geo_layer("input_zones", "input"),
    ]
    monkeypatch.setattr(
        tasks_module, "build_input_geo_layers", lambda *a, **k: input_layers
    )

    streamed, file_parts = _run_finished_chat_stream(
        monkeypatch,
        [_geo_layer("classified_result", "result")],
        emit_input_files=True,
    )

    assert streamed == ["input_cadastral", "input_zones", "classified_result"]
    assert [part["name"] for part in file_parts] == streamed
    assert all("download_url" not in part for part in file_parts)


def test_task_chat_stream_uses_custom_input_layers_builder(monkeypatch) -> None:
    streamed, file_parts = _run_finished_chat_stream(
        monkeypatch,
        [_geo_layer("classified_result", "result")],
        emit_input_files=True,
        input_layers_builder=lambda *a: [_geo_layer("functional_zones", "input")],
    )

    assert streamed == ["functional_zones", "classified_result"]
    assert [part["name"] for part in file_parts] == streamed


def test_build_scenario_zone_geo_layers_exposes_functional_zones_only() -> None:
    from types import SimpleNamespace

    from service.api.tasks import build_scenario_zone_geo_layers

    app_settings = SimpleNamespace(
        public_base_url="https://pzz.example", app_name="pzz"
    )
    task = SimpleNamespace(
        cadastral_data_path="/inputs/ext-1/cadastral_feature_collection.geojson",
        pzz_zones_data_path="/inputs/ext-1/pzz_zones_feature_collection.geojson",
    )

    layers = build_scenario_zone_geo_layers(task, "ext-1", app_settings)

    assert layers == [
        {
            "name": "functional_zones",
            "title": "Функциональные зоны",
            "role": "input",
            "url": "https://pzz.example/files/zones/ext-1",
            "download_url": None,
            "filename": "functional_zones.geojson",
            "mime_type": "application/geo+json",
            "source_service": "pzz",
        }
    ]
    task.pzz_zones_data_path = None
    assert build_scenario_zone_geo_layers(task, "ext-1", app_settings) == []


def test_scenario_chat_stream_emits_functional_zones_layer(monkeypatch) -> None:
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from service import app as app_module
    from service.api import scenarios as scenarios_module
    from service.api.security import AuthUser, get_current_user
    from service.api.tasks import build_scenario_zone_geo_layers
    from service.dependencies import get_db

    captured: dict = {}

    async def _fake_build_task(**kwargs):
        return SimpleNamespace(external_id="ext-1")

    async def _fake_project_id(*args):
        return None

    async def _fake_stream(external_id, **kwargs):
        captured.update(kwargs)
        yield ServerSentEvent(data="{}", event="done")

    monkeypatch.setattr(
        scenarios_module, "_build_scenario_classification_task", _fake_build_task
    )
    monkeypatch.setattr(scenarios_module, "_fetch_project_id", _fake_project_id)
    monkeypatch.setattr(
        scenarios_module,
        "TaskOut",
        SimpleNamespace(
            model_validate=lambda t: SimpleNamespace(model_dump=lambda mode: {})
        ),
    )
    monkeypatch.setattr(
        scenarios_module, "task_stream_with_chat_generator", _fake_stream
    )
    app = app_module.app
    app.dependency_overrides[get_current_user] = lambda: AuthUser(
        token="t", user_id="u"
    )
    app.dependency_overrides[get_db] = lambda: SimpleNamespace(commit=lambda: None)
    try:
        resp = TestClient(app).post(
            "/scenarios/1/chat/stream",
            data={"user_query": "q", "year": 2026, "source": "User"},
        )
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_db, None)

    assert resp.status_code == 200
    assert captured["emit_input_files"] is True
    assert captured["input_layers_builder"] is build_scenario_zone_geo_layers
