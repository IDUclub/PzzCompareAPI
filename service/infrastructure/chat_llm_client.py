"""Async chat-LLM clients for the conversational flow.

Two backends behind one interface, picked by ``LLM_BACKEND`` (see
``build_chat_llm_client``) — the same switch the pipeline uses:

- ``vllm``   → OpenAI-compatible ``POST /v1/chat/completions`` (SSE stream,
  ``response_format`` for structured output);
- ``ollama`` → native ``POST /api/chat`` (JSONL stream, ``format`` for
  structured output).

Both expose ``stream_chat`` (yields assistant content deltas) and
``complete_json`` (one non-streaming call returning a schema-conforming dict).

Reasoning models (gpt-oss) emit a chain-of-thought *before* the answer. On the
vLLM side it arrives in a separate ``delta.reasoning`` field, which is dropped —
only ``delta.content`` is streamed to the user.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, AsyncIterator

import httpx

if TYPE_CHECKING:
    from ..settings import Settings

_STRUCTURED_OUTPUT_NAME = "structured_output"


class ChatLlmError(RuntimeError):
    """Non-2xx response, malformed stream, or transport failure from a chat backend."""

    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self.body = body
        super().__init__(f"chat backend returned {status}: {body!r}")


def _loads_json_object(content: str) -> Any:
    """Parse a JSON object from model ``content``, tolerating stray wrapping.

    Structured output is requested from both backends, but reasoning models can
    still leak a chain-of-thought or ```json code fences around the object. Try a
    strict parse first, then fall back to the outermost ``{...}`` span.
    """
    text = (content or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ChatLlmError(200, content) from exc
    raise ChatLlmError(200, content)


class ChatLlmClient:
    """Shared lifecycle for the chat backends. Caller owns it via ``async with``."""

    def __init__(
        self,
        base_url: str,
        *,
        default_model: str,
        timeout_seconds: float = 900.0,
        temperature: float = 0.3,
        api_key: str = "",
    ) -> None:
        if not base_url:
            raise RuntimeError("a chat base url is not configured.")
        if not default_model:
            raise RuntimeError(
                "a chat model must be configured (CHAT_MODEL / GENERATE_MODEL)."
            )
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._temperature = temperature
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=timeout_seconds
        )

    async def __aenter__(self) -> "ChatLlmClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._client.aclose()

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        raise NotImplementedError

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        raise NotImplementedError


class VllmChatClient(ChatLlmClient):
    """OpenAI-compatible client for vLLM's ``/v1/chat/completions``."""

    _PATH = "/v1/chat/completions"

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream assistant content deltas for ``messages``.

        Yields the incremental ``choices[0].delta.content`` chunks. Reasoning
        deltas are dropped. Raises ``ChatLlmError`` on a non-2xx status, an
        unparseable stream, or a transport failure.
        """
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "stream": True,
            "messages": messages,
            "temperature": (self._temperature if temperature is None else temperature),
        }

        try:
            async with self._client.stream(
                "POST", self._PATH, json=payload, headers=self._headers
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise ChatLlmError(
                        resp.status_code, body.decode("utf-8", "replace")
                    )
                async for line in resp.aiter_lines():
                    data = line.strip()
                    if not data.startswith("data:"):
                        continue
                    data = data[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise ChatLlmError(resp.status_code, data) from exc
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content") or ""
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise ChatLlmError(0, f"request failed: {exc!s}") from exc

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Non-streaming call with OpenAI ``response_format`` structured output.

        Raises ``ChatLlmError`` on a non-2xx status, when the answer is not valid
        JSON, or when the model spent its budget on reasoning and returned no
        content at all.
        """
        selected_model = model or self._default_model
        payload: dict[str, Any] = {
            "model": selected_model,
            "stream": False,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": _STRUCTURED_OUTPUT_NAME,
                    "schema": schema,
                    "strict": True,
                },
            },
            "temperature": temperature,
        }
        # gpt-oss reasons before answering; without this the chain-of-thought can
        # consume the whole budget and leave ``content`` empty.
        if selected_model.lower().startswith("gpt-oss"):
            payload["reasoning_effort"] = "low"

        resp = await self._client.post(self._PATH, json=payload, headers=self._headers)
        if resp.status_code >= 400:
            raise ChatLlmError(resp.status_code, resp.text)
        choices = resp.json().get("choices") or [{}]
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if not content and choices[0].get("finish_reason") == "length":
            raise ChatLlmError(
                resp.status_code, "no content: the answer was cut off while reasoning"
            )
        parsed = _loads_json_object(content)
        if not isinstance(parsed, dict):
            raise ChatLlmError(resp.status_code, content)
        return parsed


class OllamaChatClient(ChatLlmClient):
    """Native client for Ollama's ``/api/chat``."""

    _PATH = "/api/chat"

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream assistant content deltas for ``messages``.

        Yields the incremental ``message.content`` chunks as they arrive.
        Raises ``ChatLlmError`` on a non-2xx status, an unparseable stream, or
        a transport failure (connection refused, DNS/host resolution error, …).
        """
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "stream": True,
            "messages": messages,
            "options": {
                "temperature": (
                    self._temperature if temperature is None else temperature
                )
            },
        }

        try:
            async with self._client.stream(
                "POST", self._PATH, json=payload, headers=self._headers
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise ChatLlmError(
                        resp.status_code, body.decode("utf-8", "replace")
                    )
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ChatLlmError(resp.status_code, line) from exc
                    delta = (chunk.get("message") or {}).get("content") or ""
                    if delta:
                        yield delta
                    if chunk.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise ChatLlmError(0, f"request failed: {exc!s}") from exc

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Non-streaming ``/api/chat`` with structured output (Ollama ``format``).

        Sends ``stream: false`` and ``format=schema`` so the model must return
        JSON conforming to ``schema`` (e.g. an ``enum`` of real column names, so
        it cannot hallucinate a field). Parses ``message.content`` and returns it
        as a dict. Raises ``ChatLlmError`` on a non-2xx status or when the
        content is not valid JSON.
        """
        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "stream": False,
            "messages": messages,
            "format": schema,
            # Reasoning models (e.g. gpt-oss) otherwise prepend a chain-of-thought
            # to ``content`` and wrap the JSON in code fences, breaking the parse.
            "think": False,
            "options": {"temperature": temperature},
        }
        resp = await self._client.post(self._PATH, json=payload, headers=self._headers)
        if resp.status_code >= 400:
            raise ChatLlmError(resp.status_code, resp.text)
        content = (resp.json().get("message") or {}).get("content") or ""
        parsed = _loads_json_object(content)
        if not isinstance(parsed, dict):
            raise ChatLlmError(resp.status_code, content)
        return parsed


def build_chat_llm_client(settings: "Settings" | None = None) -> ChatLlmClient:
    """Build a fresh streaming chat client. Caller owns it via ``async with``.

    ``llm_backend`` picks the backend and therefore the host: ``vllm`` talks to
    ``vllm_base_url`` over the OpenAI-compatible API, anything else talks to
    ``ollama_base_url`` over Ollama's native API. Both are runtime-overridable,
    so the fleet can be repointed without a redeploy. The model is chosen per
    request, with ``chat_model`` (or ``generate_model``) as the default.
    """
    from ..settings import get_settings

    settings = settings or get_settings()
    default_model = settings.chat_model or settings.generate_model
    if settings.llm_backend == "vllm":
        return VllmChatClient(
            base_url=settings.vllm_base_url,
            default_model=default_model,
            timeout_seconds=settings.chat_request_timeout_seconds,
            temperature=settings.chat_temperature,
            api_key=settings.vllm_api_key,
        )
    return OllamaChatClient(
        base_url=settings.ollama_base_url,
        default_model=default_model,
        timeout_seconds=settings.chat_request_timeout_seconds,
        temperature=settings.chat_temperature,
    )
