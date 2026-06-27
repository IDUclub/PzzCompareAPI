"""Conversational answer over PZZ classification results (streamed).

This is the gMART-style layer: take the user's free-text ``user_query``,
ground a dedicated chat LLM (Ollama ``/api/chat``) with the classification
results + a configured system prompt, stream the assistant tokens to the
frontend, and persist the user+assistant turn to ChatStorage.

The chat LLM is a SEPARATE backend from the pipeline's classification LLM
(see ``build_ollama_chat_client``). Persistence is best-effort: a
ChatStorage failure is surfaced as an ``error`` event but never aborts the
token stream.

Designed as an async generator of plain ``dict`` events so the SSE endpoint
(phase 4) can map them to ``ServerSentEvent``s:

- ``{"type": "chat_created", "chat_id", "title"}`` — a new chat was created
  (only when the frontend did not supply ``chat_id``); the frontend should
  store it.
- ``{"type": "token", "content"}`` — an assistant content delta.
- ``{"type": "error", "stage", "detail"}`` — a non-fatal persistence/LLM error.
- ``{"type": "done", "chat_id", "assistant_message_id"}`` — terminal marker.

The clients are injected (already opened) so the endpoint owns their
lifetime via ``async with`` and tests can pass fakes.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator

from ...infrastructure.chat_storage_client import ChatStorageClient, ChatStorageError
from ...infrastructure.ollama_chat_client import OllamaChatClient, OllamaChatError

logger = logging.getLogger("service.chat")

_DEFAULT_SYSTEM_PROMPT = (
    "Ты — ассистент по правилам землепользования и застройки (ПЗЗ). "
    "Отвечай на запрос пользователя на русском языке, опираясь только на "
    "переданные результаты проверки. Не выдумывай данные."
)


@lru_cache(maxsize=8)
def load_system_prompt(path: str) -> str:
    """Read the system prompt from ``path`` (cached), or fall back to a default."""
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
        return text or _DEFAULT_SYSTEM_PROMPT
    except OSError:
        logger.warning("chat system prompt not found at %s, using default", path)
        return _DEFAULT_SYSTEM_PROMPT


def _extract_problem_objects(
    object_zone_fit: dict[str, Any],
    cap: int,
    reason_chars: int = 240,
) -> list[dict[str, Any]]:
    """Return the wrong/unclear objects (key fields only), capped in count.

    Works for both ``group_by`` shapes: a flat ``objects`` list or per-zone
    ``zones[].objects``. Keeps the answer-relevant fields so the model can be
    specific without ingesting the whole (potentially huge) report.
    """
    objects = object_zone_fit.get("objects")
    if not objects:
        objects = []
        for zone in object_zone_fit.get("zones") or []:
            objects.extend(zone.get("objects") or [])
    problems = [o for o in objects if o.get("fit") in ("wrong", "unclear")]
    trimmed: list[dict[str, Any]] = []
    for obj in problems[:cap]:
        reason = obj.get("reason")
        trimmed.append(
            {
                "vri_text": obj.get("vri_text"),
                "zone_name": obj.get("zone_name"),
                "verdict": obj.get("verdict"),
                "fit": obj.get("fit"),
                "reason": (reason[:reason_chars] if isinstance(reason, str) else reason),
            }
        )
    return trimmed


def build_classification_context(
    *,
    chat_message: str | None = None,
    object_zone_fit: dict[str, Any] | None = None,
    zones_info: dict[str, Any] | None = None,
    max_report_chars: int = 12000,
    max_problem_objects: int = 60,
) -> str:
    """Assemble a grounding context block from available classification outputs.

    The full ``object_zone_fit`` report grows with the object count and can
    overflow the chat model's context window on large scenarios (thousands of
    objects), yielding an empty answer. So:

    - the compact ``summary`` (exact counts) is always included;
    - the full report JSON is included only when under ``max_report_chars``;
    - otherwise a capped list of just the wrong/unclear objects is included, so
      the model still has the answer-relevant detail without overflowing.
    """
    parts: list[str] = []
    if chat_message:
        parts.append("Готовое резюме проверки ПЗЗ:\n" + chat_message)
    if object_zone_fit:
        summary = object_zone_fit.get("summary")
        if summary:
            parts.append("Сводка:\n" + json.dumps(summary, ensure_ascii=False, default=str))
        report_json = json.dumps(object_zone_fit, ensure_ascii=False, default=str)
        if len(report_json) <= max_report_chars:
            parts.append("Структурированный отчёт (JSON):\n" + report_json)
        else:
            problems = _extract_problem_objects(object_zone_fit, max_problem_objects)
            if problems:
                parts.append(
                    f"Неуместные/спорные объекты (показаны первые {len(problems)}; "
                    "точные итоги — в «Сводка»):\n"
                    + json.dumps(problems, ensure_ascii=False, default=str)
                )
    if zones_info:
        zones_json = json.dumps(zones_info, ensure_ascii=False, default=str)
        if len(zones_json) <= max_report_chars:
            parts.append("Справка по зонам (JSON):\n" + zones_json)
    return "\n\n".join(parts)


def build_llm_history(
    messages: list[dict[str, Any]],
    max_messages: int = 10,
) -> list[dict[str, str]]:
    """Convert ChatStorage messages to a compact Ollama-compatible history.

    Mirrors gMART's ``build_llm_history``: keeps only user/assistant turns and
    extracts plain text (ChatStorage returns text as ``parts[*].payload.text``;
    a top-level string ``content`` is also accepted). Status/tool-call parts are
    skipped so internal pipeline details don't pollute the LLM context. Returns
    at most the ``max_messages`` most recent turns.
    """
    result: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            result.append({"role": role, "content": content.strip()})
            continue

        texts = [
            part["payload"]["text"]
            for part in (message.get("parts") or [])
            if part.get("kind") == "text" and (part.get("payload") or {}).get("text")
        ]
        combined = "\n".join(texts).strip()
        if combined:
            result.append({"role": role, "content": combined})

    return result[-max_messages:]


def build_messages(
    system_prompt: str,
    classification_context: str,
    user_query: str,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build the ``messages`` array: system (prompt + context), prior history, user query."""
    system_content = system_prompt
    if classification_context:
        system_content = f"{system_prompt}\n\n{classification_context}"
    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_query})
    return messages


async def stream_chat_answer(
    *,
    ollama_client: OllamaChatClient,
    chat_storage_client: ChatStorageClient | None,
    token: str | None,
    system_prompt: str,
    user_query: str,
    classification_context: str = "",
    chat_id: str | None = None,
    scenario_id: int | str | None = None,
    project_id: int | str | None = None,
    chat_title: str | None = None,
    message_metadata: dict[str, Any] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    assistant_file_parts: list[dict[str, Any]] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream a grounded assistant answer and persist the turn to ChatStorage.

    Persistence requires both a token and a ChatStorage client; when either is
    missing the answer is still streamed, just not stored.

    ``assistant_file_parts`` are ChatStorage ``file``-part payloads (geo-layer
    links) attached to the assistant message alongside the answer text.
    """
    persist = chat_storage_client is not None and bool(token)

    # 0. For an existing chat (frontend supplied chat_id), load prior turns so
    # the model has conversational memory. Done before appending the new user
    # turn so history reflects only previous messages. Non-fatal on failure.
    history: list[dict[str, str]] = []
    if persist and chat_id:
        try:
            existing = await chat_storage_client.get_chat(token, chat_id)
            history = build_llm_history(existing.get("messages") or [])
        except ChatStorageError as exc:
            logger.warning("chat_storage get_chat (history) failed: %s", exc)
            yield {"type": "error", "stage": "load_history", "detail": str(exc)}

    # 1. Ensure a chat exists. Create one if the frontend didn't supply chat_id.
    if persist and not chat_id:
        try:
            created = await chat_storage_client.create_chat(
                token,
                title=chat_title,
                scenario_id=scenario_id,
                project_id=project_id,
                metadata=message_metadata,
            )
            chat_id = created.get("chat_id")
            yield {"type": "chat_created", "chat_id": chat_id, "title": created.get("title")}
        except ChatStorageError as exc:
            logger.warning("chat_storage create_chat failed: %s", exc)
            yield {"type": "error", "stage": "create_chat", "detail": str(exc)}
            persist = False

    # 2. Persist the user turn before generating the answer.
    if persist and chat_id:
        try:
            await chat_storage_client.add_message(
                token,
                chat_id,
                role="user",
                content=user_query,
                metadata=message_metadata,
            )
        except ChatStorageError as exc:
            logger.warning("chat_storage add user message failed: %s", exc)
            yield {"type": "error", "stage": "add_user_message", "detail": str(exc)}

    # 3. Stream the assistant answer from the dedicated chat LLM.
    messages = build_messages(system_prompt, classification_context, user_query, history)
    collected: list[str] = []
    try:
        async for delta in ollama_client.stream_chat(
            messages, model=model, temperature=temperature
        ):
            collected.append(delta)
            yield {"type": "token", "content": delta}
    except OllamaChatError as exc:
        logger.warning("chat LLM stream failed: %s", exc)
        yield {"type": "error", "stage": "llm", "detail": str(exc)}

    answer = "".join(collected).strip()

    # 4. Persist the assistant turn (answer text + any geo-layer file links).
    assistant_message_id: str | None = None
    if persist and chat_id and (answer or assistant_file_parts):
        try:
            if assistant_file_parts:
                parts: list[dict[str, Any]] = []
                if answer:
                    parts.append({"kind": "text", "payload": {"text": answer}})
                parts.extend(
                    {"kind": "file", "payload": payload}
                    for payload in assistant_file_parts
                )
                stored = await chat_storage_client.add_message(
                    token, chat_id, role="assistant", parts=parts, metadata=message_metadata
                )
            else:
                stored = await chat_storage_client.add_message(
                    token, chat_id, role="assistant", content=answer, metadata=message_metadata
                )
            assistant_message_id = stored.get("message_id")
        except ChatStorageError as exc:
            logger.warning("chat_storage add assistant message failed: %s", exc)
            yield {"type": "error", "stage": "add_assistant_message", "detail": str(exc)}

    yield {
        "type": "done",
        "chat_id": chat_id,
        "assistant_message_id": assistant_message_id,
    }
