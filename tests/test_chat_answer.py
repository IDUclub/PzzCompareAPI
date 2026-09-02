"""Tests for the conversational answer use-case (phase 3)."""

import asyncio
import json

from service.infrastructure.chat_llm_client import ChatLlmError
from service.application.use_cases.chat_answer import (
    build_classification_context,
    build_messages,
    stream_chat_answer,
)


class FakeOllama:
    def __init__(self, deltas=("a", "b"), error=False):
        self._deltas = deltas
        self._error = error
        self.seen_messages = None

    async def stream_chat(self, messages, *, model=None, temperature=None):
        self.seen_messages = messages
        if self._error:
            raise ChatLlmError(500, "boom")
        for d in self._deltas:
            yield d


class FakeChatStorage:
    def __init__(self, history_messages=None):
        self.calls = []
        self.parts_calls = []
        self._history_messages = history_messages or []

    async def create_chat(
        self, user_id, *, title=None, scenario_id=None, project_id=None, metadata=None
    ):
        self.calls.append(("create", scenario_id, project_id))
        return {"chat_id": "new-chat", "title": title}

    async def add_message(
        self, user_id, chat_id, *, role, content=None, parts=None, metadata=None
    ):
        self.calls.append(("msg", chat_id, role, content))
        if parts is not None:
            self.parts_calls.append((chat_id, role, parts))
        return {"message_id": f"m-{role}", "chat_id": chat_id}

    async def get_chat(self, user_id, chat_id):
        self.calls.append(("get_chat", chat_id))
        return {"chat_id": chat_id, "messages": self._history_messages}


def _collect(gen):
    async def run():
        return [ev async for ev in gen]

    return asyncio.run(run())


def test_creates_chat_and_persists_turn_when_no_chat_id() -> None:
    cs = FakeChatStorage()
    events = _collect(
        stream_chat_answer(
            chat_client=FakeOllama(("Привет", " мир")),
            chat_storage_client=cs,
            user_id="tok",
            system_prompt="SYS",
            user_query="Что не так?",
            classification_context="CTX",
            chat_id=None,
            scenario_id=772,
            project_id=42,
        )
    )
    types = [e["type"] for e in events]
    assert types[0] == "chat_created"
    assert events[0]["chat_id"] == "new-chat"
    assert types[-1] == "done"
    assert "".join(e["content"] for e in events if e["type"] == "token") == "Привет мир"
    # user before assistant, both persisted to the created chat.
    assert ("create", 772, 42) in cs.calls
    assert ("msg", "new-chat", "user", "Что не так?") in cs.calls
    assert ("msg", "new-chat", "assistant", "Привет мир") in cs.calls


def test_uses_supplied_chat_id_without_creating() -> None:
    cs = FakeChatStorage()
    events = _collect(
        stream_chat_answer(
            chat_client=FakeOllama(),
            chat_storage_client=cs,
            user_id="tok",
            system_prompt="SYS",
            user_query="q",
            chat_id="existing",
        )
    )
    assert not any(e["type"] == "chat_created" for e in events)
    assert all(c[0] != "create" for c in cs.calls)
    assert events[-1]["chat_id"] == "existing"


def test_no_persist_without_storage_still_streams() -> None:
    events = _collect(
        stream_chat_answer(
            chat_client=FakeOllama(("x", "y")),
            chat_storage_client=None,
            user_id=None,
            system_prompt="SYS",
            user_query="q",
            chat_id=None,
        )
    )
    assert "".join(e["content"] for e in events if e["type"] == "token") == "xy"
    assert events[-1]["type"] == "done"
    assert not any(e["type"] == "chat_created" for e in events)


def test_llm_error_emits_error_then_done() -> None:
    cs = FakeChatStorage()
    events = _collect(
        stream_chat_answer(
            chat_client=FakeOllama(error=True),
            chat_storage_client=cs,
            user_id="tok",
            system_prompt="SYS",
            user_query="q",
            chat_id="c1",
        )
    )
    assert any(e["type"] == "error" and e["stage"] == "llm" for e in events)
    assert events[-1]["type"] == "done"
    # No assistant message persisted when the answer is empty.
    assert all(not (c[0] == "msg" and c[2] == "assistant") for c in cs.calls)


def test_persistence_failure_emits_warning_not_error() -> None:
    from service.infrastructure.chat_storage_client import ChatStorageError

    class FailingChatStorage(FakeChatStorage):
        async def create_chat(self, user_id, **kwargs):
            raise ChatStorageError(401, "Token expired")

    cs = FailingChatStorage()
    events = _collect(
        stream_chat_answer(
            chat_client=FakeOllama(("x", "y")),
            chat_storage_client=cs,
            user_id="tok",
            system_prompt="SYS",
            user_query="q",
            chat_id=None,
        )
    )
    # The answer still streams to completion.
    assert "".join(e["content"] for e in events if e["type"] == "token") == "xy"
    assert events[-1]["type"] == "done"
    # Persistence failure is a WARNING, never an error (so the frontend doesn't
    # treat "answer fine, just not saved" as a service failure).
    warnings = [e for e in events if e["type"] == "warning"]
    assert any(w["stage"] == "create_chat" and w.get("message") for w in warnings)
    assert not any(e["type"] == "error" for e in events)


def test_existing_chat_loads_history_into_messages() -> None:
    history_messages = [
        {
            "role": "user",
            "parts": [{"kind": "text", "payload": {"text": "Прошлый вопрос"}}],
        },
        {"role": "assistant", "content": "Прошлый ответ"},
        {"role": "system", "content": "служебное — должно быть отброшено"},
    ]
    fake_llm = FakeOllama(("ok",))
    cs = FakeChatStorage(history_messages=history_messages)
    _collect(
        stream_chat_answer(
            chat_client=fake_llm,
            chat_storage_client=cs,
            user_id="tok",
            system_prompt="SYS",
            user_query="Новый вопрос",
            chat_id="existing",
        )
    )
    assert ("get_chat", "existing") in cs.calls
    roles = [m["role"] for m in fake_llm.seen_messages]
    # system, then prior user+assistant history, then the new user query.
    assert roles == ["system", "user", "assistant", "user"]
    assert fake_llm.seen_messages[1]["content"] == "Прошлый вопрос"
    assert fake_llm.seen_messages[2]["content"] == "Прошлый ответ"
    assert fake_llm.seen_messages[-1]["content"] == "Новый вопрос"


def test_assistant_message_carries_geo_file_part() -> None:
    cs = FakeChatStorage()
    file_part = {
        "url": "https://api.example.org/files/result/abc",
        "filename": "abc.geojson",
        "mime_type": "application/geo+json",
        "source_service": "PZZ Pipeline Service",
    }
    _collect(
        stream_chat_answer(
            chat_client=FakeOllama(("ответ",)),
            chat_storage_client=cs,
            user_id="tok",
            system_prompt="SYS",
            user_query="q",
            chat_id="c1",
            assistant_file_parts=[file_part],
        )
    )
    assert cs.parts_calls, "assistant message must be persisted as parts"
    _chat_id, role, parts = cs.parts_calls[-1]
    assert role == "assistant"
    assert [p["kind"] for p in parts] == ["text", "file"]
    assert parts[0]["payload"] == {"text": "ответ"}
    assert parts[1]["payload"] == file_part


def test_large_report_falls_back_to_problem_objects() -> None:
    # A report whose full JSON exceeds the cap: the full dump is dropped, but the
    # wrong/unclear objects are still surfaced (with the exact summary counts).
    zones = [
        {
            "zone_name": f"Зона {i}",
            "objects": [
                {
                    "vri_text": f"объект {i}",
                    "zone_name": f"Зона {i}",
                    "verdict": "not_allowed",
                    "fit": "wrong",
                    "reason": "очень длинная причина " * 50,
                },
            ],
        }
        for i in range(200)
    ]
    report = {
        "summary": {"total": 200, "in_wrong_zone": 200, "unclear": 0},
        "zones": zones,
    }
    ctx = build_classification_context(
        object_zone_fit=report, max_report_chars=2000, max_problem_objects=60
    )
    assert "Сводка" in ctx  # exact counts always present
    assert "Структурированный отчёт" not in ctx  # full dump dropped (too big)
    assert "Проблемные земельные участки" in ctx  # problem parcels included instead
    assert "показаны первые 60" in ctx  # capped
    # reason trimmed, so the block stays bounded
    assert len(ctx) < 40000


def test_build_messages_and_context() -> None:
    ctx = build_classification_context(chat_message="РЕЗЮМЕ", object_zone_fit={"a": 1})
    assert "РЕЗЮМЕ" in ctx and "JSON" in ctx
    msgs = build_messages("SYS", ctx, "вопрос")
    assert (
        msgs[0]["role"] == "system"
        and "SYS" in msgs[0]["content"]
        and "РЕЗЮМЕ" in msgs[0]["content"]
    )
    assert msgs[1] == {"role": "user", "content": "вопрос"}


_LEAKED_SECRET = "Bearer eyJhbGciOiJSUzI1NiJ9.secret-service-token.sig"


def test_storage_failure_never_forwards_the_upstream_body() -> None:
    """ChatStorage echoes the request headers (service token included) in its 5xx
    body. That body must stay in the logs, never reach a client event."""
    from service.infrastructure.chat_storage_client import ChatStorageError

    class LeakyChatStorage(FakeChatStorage):
        async def create_chat(self, user_id, **kwargs):
            raise ChatStorageError(
                500, {"request": {"headers": {"authorization": _LEAKED_SECRET}}}
            )

    events = _collect(
        stream_chat_answer(
            chat_client=FakeOllama(("x",)),
            chat_storage_client=LeakyChatStorage(),
            user_id="tok",
            system_prompt="SYS",
            user_query="q",
            chat_id=None,
        )
    )
    serialized = json.dumps(events, ensure_ascii=False)
    assert "secret-service-token" not in serialized
    assert "authorization" not in serialized.lower()
    warning = next(e for e in events if e["type"] == "warning")
    assert warning["detail"] == "chat_storage returned 500"


def test_llm_error_detail_carries_status_not_the_upstream_body() -> None:
    class LeakyLlm(FakeOllama):
        async def stream_chat(self, messages, *, model=None, temperature=None):
            raise ChatLlmError(500, {"internal": _LEAKED_SECRET})
            yield  # pragma: no cover — generator marker

    events = _collect(
        stream_chat_answer(
            chat_client=LeakyLlm(),
            chat_storage_client=FakeChatStorage(),
            user_id="tok",
            system_prompt="SYS",
            user_query="q",
            chat_id="c1",
        )
    )
    error = next(e for e in events if e["type"] == "error")
    assert error["detail"] == "chat backend returned 500"
    assert "secret-service-token" not in json.dumps(events, ensure_ascii=False)


def test_unreachable_and_rejected_storage_read_differently() -> None:
    """A rejected request must not be reported as an unavailable service —
    that wording sends an incident investigation the wrong way."""
    from service.infrastructure.chat_storage_client import ChatStorageError

    def _warning_for(status: int) -> dict:
        class Failing(FakeChatStorage):
            async def create_chat(self, user_id, **kwargs):
                raise ChatStorageError(status, "boom")

        events = _collect(
            stream_chat_answer(
                chat_client=FakeOllama(("x",)),
                chat_storage_client=Failing(),
                user_id="tok",
                system_prompt="SYS",
                user_query="q",
                chat_id=None,
            )
        )
        return next(e for e in events if e["type"] == "warning")

    unreachable = _warning_for(0)
    rejected = _warning_for(500)
    assert "недоступен" in unreachable["message"]
    assert unreachable["detail"] == "chat_storage unreachable"
    assert "отклонил запрос" in rejected["message"]
    assert rejected["detail"] == "chat_storage returned 500"
