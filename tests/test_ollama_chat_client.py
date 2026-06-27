"""Tests for the streaming Ollama /api/chat client (phase 2)."""
import asyncio
import json

import httpx
import pytest

from service.infrastructure.ollama_chat_client import (
    OllamaChatClient,
    OllamaChatError,
)


def _client(handler) -> OllamaChatClient:
    client = OllamaChatClient(
        base_url="http://llm.local", default_model="m", timeout_seconds=5
    )
    client._client = httpx.AsyncClient(
        base_url="http://llm.local", transport=httpx.MockTransport(handler)
    )
    return client


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
        return httpx.Response(200, content=ndjson.encode("utf-8"))

    async def run() -> list[str]:
        out: list[str] = []
        async with _client(handler) as oc:
            async for delta in oc.stream_chat(
                [{"role": "user", "content": "hi"}], temperature=0.1
            ):
                out.append(delta)
        return out

    deltas = asyncio.run(run())
    assert "".join(deltas) == "Привет мир"
    assert seen["body"]["stream"] is True
    assert seen["body"]["options"]["temperature"] == 0.1


def test_non_2xx_raises_ollama_chat_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    async def run() -> None:
        async with _client(handler) as oc:
            async for _ in oc.stream_chat([{"role": "user", "content": "hi"}]):
                pass

    with pytest.raises(OllamaChatError) as exc:
        asyncio.run(run())
    assert exc.value.status == 500
