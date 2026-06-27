"""Tests for the ChatStorage HTTP client (phase 2)."""
import asyncio
import json

import httpx
import pytest

from service.infrastructure.chat_storage_client import (
    ChatStorageClient,
    ChatStorageError,
)


def _client(handler) -> ChatStorageClient:
    client = ChatStorageClient(base_url="http://cs.local", timeout_seconds=5)
    # Swap the real transport for a deterministic mock.
    client._client = httpx.AsyncClient(
        base_url="http://cs.local", transport=httpx.MockTransport(handler)
    )
    return client


def test_create_chat_posts_contract_and_returns_summary() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            201,
            json={"chat_id": "chat-1", "title": "T", "scenario_id": 772, "project_id": 42},
        )

    async def run() -> dict:
        async with _client(handler) as cs:
            return await cs.create_chat(
                "tok", title="T", scenario_id=772, project_id=42, metadata={"k": "v"}
            )

    result = asyncio.run(run())
    assert result["chat_id"] == "chat-1"
    assert seen["url"].endswith("/api/v1/chat_history/create_chat")
    assert seen["auth"] == "Bearer tok"
    assert seen["body"]["scenario_id"] == 772
    assert seen["body"]["metadata"] == {"k": "v"}


def test_add_message_simple_text() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"message_id": "m-1", "chat_id": "chat-1"})

    async def run() -> dict:
        async with _client(handler) as cs:
            return await cs.add_message(
                "tok", "chat-1", role="user", content="привет"
            )

    result = asyncio.run(run())
    assert result["message_id"] == "m-1"
    assert seen["url"].endswith("/api/v1/chat_history/chat-1/message")
    assert seen["body"] == {"role": "user", "metadata": {}, "content": "привет"}


def test_non_2xx_raises_chat_storage_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Chat not found"})

    async def run() -> None:
        async with _client(handler) as cs:
            await cs.add_message("tok", "missing", role="user", content="x")

    with pytest.raises(ChatStorageError) as exc:
        asyncio.run(run())
    assert exc.value.status == 404
