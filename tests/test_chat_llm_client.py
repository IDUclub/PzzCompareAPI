"""Tests for the streaming chat clients (Ollama native + OpenAI-compatible vLLM)."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from service.infrastructure.chat_llm_client import (
    ChatLlmError,
    OllamaChatClient,
    VllmChatClient,
    build_chat_llm_client,
)


def _client(cls, handler):
    client = cls(base_url="http://llm.local", default_model="m", timeout_seconds=5)
    client._client = httpx.AsyncClient(
        base_url="http://llm.local", transport=httpx.MockTransport(handler)
    )
    return client


def _sse(events: list[str]) -> bytes:
    return "".join(f"data: {e}\n\n" for e in events).encode("utf-8")


# ── Ollama backend ───────────────────────────────────────────────────────────


def test_stream_chat_yields_deltas_and_stops_on_done() -> None:
    lines = [
        {"message": {"content": "Привет"}, "done": False},
        {"message": {"content": " мир"}, "done": False},
        {"message": {"content": ""}, "done": True},
        {"message": {"content": "после done"}, "done": False},  # must be ignored
    ]
    ndjson = "\n".join(json.dumps(x, ensure_ascii=False) for x in lines)
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["path"] = request.url.path
        return httpx.Response(200, content=ndjson.encode("utf-8"))

    async def run() -> list[str]:
        out: list[str] = []
        async with _client(OllamaChatClient, handler) as oc:
            async for delta in oc.stream_chat(
                [{"role": "user", "content": "hi"}], temperature=0.1
            ):
                out.append(delta)
        return out

    deltas = asyncio.run(run())
    assert "".join(deltas) == "Привет мир"
    assert seen["path"] == "/api/chat"
    assert seen["body"]["stream"] is True
    assert seen["body"]["options"]["temperature"] == 0.1


def test_non_2xx_raises_chat_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    async def run() -> None:
        async with _client(OllamaChatClient, handler) as oc:
            async for _ in oc.stream_chat([{"role": "user", "content": "hi"}]):
                pass

    with pytest.raises(ChatLlmError) as exc:
        asyncio.run(run())
    assert exc.value.status == 500


def test_transport_error_wrapped_as_chat_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno -2] Name or service not known")

    async def run() -> None:
        async with _client(OllamaChatClient, handler) as oc:
            async for _ in oc.stream_chat([{"role": "user", "content": "hi"}]):
                pass

    with pytest.raises(ChatLlmError) as exc:
        asyncio.run(run())
    assert exc.value.status == 0
    assert "request failed" in str(exc.value)


# ── vLLM backend ─────────────────────────────────────────────────────────────


def test_vllm_stream_ignores_reasoning_and_stops_on_done() -> None:
    events = [
        json.dumps({"choices": [{"delta": {"role": "assistant", "content": ""}}]}),
        json.dumps({"choices": [{"delta": {"reasoning": "надо подумать"}}]}),
        json.dumps({"choices": [{"delta": {"content": "Привет"}}]}, ensure_ascii=False),
        json.dumps({"choices": [{"delta": {"content": " мир"}}]}, ensure_ascii=False),
        "[DONE]",
        json.dumps({"choices": [{"delta": {"content": "после DONE"}}]}),
    ]
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["path"] = request.url.path
        return httpx.Response(200, content=_sse(events))

    async def run() -> list[str]:
        out: list[str] = []
        async with _client(VllmChatClient, handler) as vc:
            async for delta in vc.stream_chat(
                [{"role": "user", "content": "hi"}], temperature=0.1
            ):
                out.append(delta)
        return out

    deltas = asyncio.run(run())
    assert "".join(deltas) == "Привет мир"
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"]["stream"] is True
    assert seen["body"]["temperature"] == 0.1


def test_vllm_stream_sends_bearer_token_when_configured() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, content=_sse(["[DONE]"]))

    async def run() -> None:
        client = VllmChatClient(
            base_url="http://llm.local", default_model="m", api_key="s3cret"
        )
        client._client = httpx.AsyncClient(
            base_url="http://llm.local", transport=httpx.MockTransport(handler)
        )
        async with client as vc:
            async for _ in vc.stream_chat([{"role": "user", "content": "hi"}]):
                pass

    asyncio.run(run())
    assert seen["auth"] == "Bearer s3cret"


def test_vllm_stream_non_2xx_raises_chat_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b'{"error":"model not found"}')

    async def run() -> None:
        async with _client(VllmChatClient, handler) as vc:
            async for _ in vc.stream_chat([{"role": "user", "content": "hi"}]):
                pass

    with pytest.raises(ChatLlmError) as exc:
        asyncio.run(run())
    assert exc.value.status == 404
    assert "model not found" in str(exc.value)


def test_vllm_complete_json_uses_response_format_and_parses_content() -> None:
    schema = {
        "type": "object",
        "properties": {"column": {"type": "string", "enum": ["kad_nom"]}},
        "required": ["column"],
    }
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "reasoning": "the cadastral number lives in kad_nom",
                            "content": '{"column": "kad_nom"}',
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async def run() -> dict:
        async with _client(VllmChatClient, handler) as vc:
            return await vc.complete_json(
                [{"role": "user", "content": "pick"}], schema=schema
            )

    assert asyncio.run(run()) == {"column": "kad_nom"}
    body = seen["body"]
    assert body["stream"] is False
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert body["response_format"]["json_schema"]["strict"] is True


def test_vllm_complete_json_limits_reasoning_for_gpt_oss() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]},
        )

    async def run() -> None:
        async with _client(VllmChatClient, handler) as vc:
            await vc.complete_json(
                [{"role": "user", "content": "pick"}],
                schema={"type": "object"},
                model="gpt-oss-20b",
            )

    asyncio.run(run())
    assert seen["body"]["reasoning_effort"] == "low"


def test_vllm_complete_json_reports_truncated_reasoning() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"reasoning": "…", "content": ""},
                        "finish_reason": "length",
                    }
                ]
            },
        )

    async def run() -> None:
        async with _client(VllmChatClient, handler) as vc:
            await vc.complete_json(
                [{"role": "user", "content": "pick"}], schema={"type": "object"}
            )

    with pytest.raises(ChatLlmError) as exc:
        asyncio.run(run())
    assert "cut off while reasoning" in str(exc.value)


# ── Backend selection ────────────────────────────────────────────────────────


def _settings(backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        llm_backend=backend,
        ollama_base_url="http://ollama.local:11434",
        vllm_base_url="http://vllm.local:8001",
        vllm_api_key="key",
        chat_model="gpt-oss-20b",
        generate_model="generate",
        chat_request_timeout_seconds=900.0,
        chat_temperature=0.3,
    )


@pytest.mark.parametrize(
    "backend, expected_type, expected_host",
    [
        ("vllm", VllmChatClient, "http://vllm.local:8001"),
        ("ollama", OllamaChatClient, "http://ollama.local:11434"),
    ],
)
def test_factory_selects_backend_and_host(
    backend: str, expected_type: type, expected_host: str
) -> None:
    client = build_chat_llm_client(_settings(backend))
    assert isinstance(client, expected_type)
    assert client._base_url == expected_host
    assert client._default_model == "gpt-oss-20b"


def test_factory_falls_back_to_generate_model() -> None:
    settings = _settings("vllm")
    settings.chat_model = ""
    assert build_chat_llm_client(settings)._default_model == "generate"
