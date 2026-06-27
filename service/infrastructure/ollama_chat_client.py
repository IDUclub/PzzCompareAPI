"""Async streaming client for Ollama's ``/api/chat`` endpoint.

Used by the conversational flow to generate a natural-language answer over
the PZZ classification results and stream the tokens to the frontend via
SSE (mirrors gMART's ``BaseLlmClient.execute_request``, but built on
``httpx`` instead of the ``ollama`` package — httpx is already a project
dependency, ``ollama`` is not).

The pipeline already talks to the same Ollama backend via raw ``/api/chat``
(see ``pipeline_modules/business/clients.py``); this is the async,
token-streaming counterpart.

Request shape (Ollama):

    POST /api/chat
    {"model": ..., "stream": true, "messages": [{"role", "content"}, ...],
     "options": {"temperature": ...}}

Each streamed line is a JSON object ``{"message": {"content": "..."},
"done": false}``; the final line sets ``"done": true``.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx


class OllamaChatError(RuntimeError):
    """Non-2xx response (or malformed stream) from Ollama ``/api/chat``."""

    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self.body = body
        super().__init__(f"ollama /api/chat returned {status}: {body!r}")


class OllamaChatClient:
    """Thin async wrapper that streams assistant tokens from ``/api/chat``."""

    def __init__(
        self,
        base_url: str,
        *,
        default_model: str,
        timeout_seconds: float = 900.0,
        temperature: float = 0.3,
    ) -> None:
        if not base_url:
            raise RuntimeError("ollama_base_url is not configured.")
        if not default_model:
            raise RuntimeError("a chat model must be configured (CHAT_MODEL / GENERATE_MODEL).")
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._temperature = temperature
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout_seconds)

    async def __aenter__(self) -> "OllamaChatClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._client.aclose()

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        think: Any = None,
    ) -> AsyncIterator[str]:
        """Stream assistant content deltas for ``messages``.

        Yields the incremental ``message.content`` chunks as they arrive.
        Raises ``OllamaChatError`` on a non-2xx status or unparseable stream.
        """
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "stream": True,
            "messages": messages,
            "options": {
                "temperature": self._temperature if temperature is None else temperature
            },
        }
        if think is not None:
            payload["think"] = think

        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                raise OllamaChatError(resp.status_code, body.decode("utf-8", "replace"))
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise OllamaChatError(resp.status_code, line) from exc
                delta = (chunk.get("message") or {}).get("content") or ""
                if delta:
                    yield delta
                if chunk.get("done"):
                    break
