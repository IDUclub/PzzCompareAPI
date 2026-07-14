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
- ``{"type": "warning", "stage", "detail", "message"}`` — a NON-fatal problem:
  chat history couldn't be persisted/loaded (e.g. expired token). The answer is
  still generated and streamed; it just won't be saved to history. Distinct from
  ``error`` so the frontend can tell "your answer is fine, just not saved" apart
  from a real service failure.
- ``{"type": "error", "stage": "llm", "detail"}`` — a FATAL error: the answer
  itself could not be generated.
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
    "Ты — ассистент по правилам землепользования и застройки (ПЗЗ). Тебе "
    "передаются результаты автоматической проверки загруженных земельных "
    "участков: для каждого участка проверяется, допустим ли его вид "
    "разрешённого использования (ВРИ) в той территориальной зоне ПЗЗ, куда "
    "участок попадает. Отвечай на русском языке, опираясь только на переданные "
    "результаты. Пиши о «земельных участках» и «территориальных зонах», не "
    "называй участки «объектами» и не используй внутренние метки вроде "
    "«неясно»/«unclear» — про участки без однозначного результата пиши "
    "«требуют ручной проверки». Не выдумывай данные."
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


@lru_cache(maxsize=4)
def _load_vri_names_cached(path: str) -> tuple[tuple[str, str], ...]:
    """Load ``{vri_code: name}`` from a Rosreestr classifier's ``by_code`` block.

    Cached; returns a tuple of pairs (hashable). Used to enrich the grounding
    with human-readable ВРИ names, since the deterministic runner leaves
    ``matched_vri_name`` empty for services / most physical-object types.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    by_code = data.get("by_code") if isinstance(data, dict) else None
    if not isinstance(by_code, dict):
        return ()
    pairs: list[tuple[str, str]] = []
    for code, entry in by_code.items():
        if isinstance(entry, dict):
            name = entry.get("name_plain") or entry.get("name")
        else:
            name = entry
        if name:
            pairs.append((str(code), str(name)))
    return tuple(pairs)


def load_vri_names(path: str | None) -> dict[str, str]:
    """Public helper: ``{code: name}`` from a classifier file, {} on any failure."""
    return dict(_load_vri_names_cached(path)) if path else {}


def _extract_problem_objects(
    object_zone_fit: dict[str, Any],
    cap: int,
    reason_chars: int = 240,
    vri_names: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return the wrong/unclear objects (key fields only), capped in count.

    Works for both ``group_by`` shapes: a flat ``objects`` list or per-zone
    ``zones[].objects``. Keeps the answer-relevant fields — including the ВРИ
    the runner assigned (code + name) and the verdict reason — so the model can
    explain each problem parcel specifically without ingesting the whole report.
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
        code = obj.get("matched_vri_code") or ""
        name = obj.get("matched_vri_name") or ""
        if not name and code and vri_names:
            name = vri_names.get(code, "")
        trimmed.append(
            {
                "vri_text": obj.get("vri_text"),
                "zone_name": obj.get("zone_name"),
                "verdict": obj.get("verdict"),
                "assigned_vri_code": code or None,
                "assigned_vri_name": name or None,
                "основание_подбора_ВРИ": obj.get("resolution_basis") or None,
                "reason": (reason[:reason_chars] if isinstance(reason, str) else reason),
            }
        )
    return trimmed


def build_classification_context(
    *,
    chat_message: str | None = None,
    object_zone_fit: dict[str, Any] | None = None,
    zones_info: dict[str, Any] | None = None,
    vri_names: dict[str, str] | None = None,
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
        # Compact per-zone breakdown, always included (it's tiny — one row per
        # zone). Lets the answer give a detailed per-zone report even when the
        # full report JSON below is dropped for exceeding max_report_chars.
        zones = object_zone_fit.get("zones")
        if zones:
            per_zone = [
                {
                    "зона_ПЗЗ": z.get("zone_name") or z.get("zone_type_id"),
                    "всего": (z.get("summary") or {}).get("total"),
                    "допустимо": (z.get("summary") or {}).get("in_correct_zone"),
                    "не_соответствует": (z.get("summary") or {}).get("in_wrong_zone"),
                    "ручная_проверка": (z.get("summary") or {}).get("unclear"),
                }
                for z in zones
                if z.get("zone_type_id")  # skip the synthetic "no zone" bucket
            ]
            if per_zone:
                parts.append(
                    "Разбивка по территориальным зонам ПЗЗ:\n"
                    + json.dumps(per_zone, ensure_ascii=False, default=str)
                )
        # Aggregated (verdict, reason) -> count for the non-clean objects, so the
        # answer can say WHY parcels need manual review / were rejected, not just
        # how many. Always small (a handful of distinct reasons).
        all_objects = object_zone_fit.get("objects")
        if not all_objects:
            all_objects = []
            for z in object_zone_fit.get("zones") or []:
                all_objects.extend(z.get("objects") or [])
        reason_counts: dict[tuple[str, str], int] = {}
        for o in all_objects:
            if o.get("fit") in ("wrong", "unclear"):
                key = (o.get("verdict") or "—", (o.get("reason") or "").strip())
                reason_counts[key] = reason_counts.get(key, 0) + 1
        if reason_counts:
            reasons = [
                {"вердикт": v, "причина": r, "участков": n}
                for (v, r), n in sorted(reason_counts.items(), key=lambda kv: -kv[1])
            ]
            parts.append(
                "Причины по проблемным участкам (агрегировано):\n"
                + json.dumps(reasons, ensure_ascii=False, default=str)
            )
        report_json = json.dumps(object_zone_fit, ensure_ascii=False, default=str)
        if len(report_json) <= max_report_chars:
            parts.append("Структурированный отчёт (JSON):\n" + report_json)
        else:
            problems = _extract_problem_objects(
                object_zone_fit, max_problem_objects, vri_names=vri_names
            )
            if problems:
                parts.append(
                    "Проблемные земельные участки (потенциальные нарушения и "
                    f"требующие ручной проверки; показаны первые {len(problems)}, "
                    "точные итоги — в «Сводка»; для каждого: присвоенный ВРИ и "
                    "причина вердикта):\n"
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
    user_id: str | None,
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

    Persistence requires both a ``user_id`` and a ChatStorage client; when
    either is missing the answer is still streamed, just not stored. ChatStorage
    is authenticated with our service token (inside the client) and the chat is
    owned by ``user_id`` (sent as ``X-User-Id``).

    ``assistant_file_parts`` are ChatStorage ``file``-part payloads (geo-layer
    links) attached to the assistant message alongside the answer text.
    """
    persist = chat_storage_client is not None and bool(user_id)

    # 0. For an existing chat (frontend supplied chat_id), load prior turns so
    # the model has conversational memory. Done before appending the new user
    # turn so history reflects only previous messages. Non-fatal on failure.
    history: list[dict[str, str]] = []
    if persist and chat_id:
        try:
            existing = await chat_storage_client.get_chat(user_id, chat_id)
            history = build_llm_history(existing.get("messages") or [])
        except ChatStorageError as exc:
            logger.warning("chat_storage get_chat (history) failed: %s", exc)
            yield {
                "type": "warning",
                "stage": "load_history",
                "detail": str(exc),
                "message": "Не удалось загрузить историю чата — отвечаю без учёта "
                "предыдущих сообщений.",
            }

    # 1. Ensure a chat exists. Create one if the frontend didn't supply chat_id.
    if persist and not chat_id:
        try:
            created = await chat_storage_client.create_chat(
                user_id,
                title=chat_title,
                scenario_id=scenario_id,
                project_id=project_id,
                metadata=message_metadata,
            )
            chat_id = created.get("chat_id")
            yield {"type": "chat_created", "chat_id": chat_id, "title": created.get("title")}
        except ChatStorageError as exc:
            logger.warning("chat_storage create_chat failed: %s", exc)
            yield {
                "type": "warning",
                "stage": "create_chat",
                "detail": str(exc),
                "message": "Ответ сформирован, но не сохранён в историю чата "
                "(сервис истории недоступен).",
            }
            persist = False

    # 2. Persist the user turn before generating the answer.
    if persist and chat_id:
        try:
            await chat_storage_client.add_message(
                user_id,
                chat_id,
                role="user",
                content=user_query,
                metadata=message_metadata,
            )
        except ChatStorageError as exc:
            logger.warning("chat_storage add user message failed: %s", exc)
            yield {
                "type": "warning",
                "stage": "add_user_message",
                "detail": str(exc),
                "message": "Ответ сформирован, но не сохранён в историю чата "
                "(сервис истории недоступен).",
            }

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
                    user_id, chat_id, role="assistant", parts=parts, metadata=message_metadata
                )
            else:
                stored = await chat_storage_client.add_message(
                    user_id, chat_id, role="assistant", content=answer, metadata=message_metadata
                )
            assistant_message_id = stored.get("message_id")
        except ChatStorageError as exc:
            logger.warning("chat_storage add assistant message failed: %s", exc)
            yield {
                "type": "warning",
                "stage": "add_assistant_message",
                "detail": str(exc),
                "message": "Ответ сформирован, но не сохранён в историю чата "
                "(сервис истории недоступен).",
            }

    yield {
        "type": "done",
        "chat_id": chat_id,
        "assistant_message_id": assistant_message_id,
    }
