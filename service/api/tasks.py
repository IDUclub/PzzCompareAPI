"""Task management endpoints: get / list / cancel / recompute / events / result / stream."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from starlette.responses import FileResponse, RedirectResponse, Response

from ..application.use_cases.chat_answer import (
    build_classification_context,
    load_system_prompt,
    load_vri_names,
    stream_chat_answer,
)
from ..db import session_scope
from ..dependencies import (
    build_chat_storage_client,
    build_ollama_chat_client,
    get_app_settings,
    get_db,
    get_event_repo,
    get_task_repo,
)
from ..domain.ports.event_repository import EventRepository
from ..domain.ports.task_repository import TaskRepository
from ..domain.task_state import ensure_transition
from ..infrastructure.pzz_mapping import lookup_zone_summary
from ..infrastructure.storage import get_object_storage, is_remote_path
from ..models import PipelineTask, TaskEvent, TaskStatus

_TERMINAL_STATUSES = {TaskStatus.finished, TaskStatus.failed}
from ..schemas import TaskEventOut, TaskListOut, TaskOut
from ..settings import Settings
from ..tasks import celery_app, enqueue_pipeline_task, execute_pipeline_task
from ..time_utils import utc_now
from .utils import api_log

router = APIRouter(tags=["tasks"])
_SCENARIO_IDEMPOTENCY_PREFIX = "sc:"
_BUILDING_IDEMPOTENCY_PREFIX = "bld:"


def get_task_or_404(external_id: str, task_repo: TaskRepository) -> PipelineTask:
    task = task_repo.get_by_external_id(external_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


def _is_scenario_task(external_id: str, task_repo: TaskRepository) -> bool:
    key = task_repo.get_idempotency_key_by_external_id(external_id)
    return bool(key and key.startswith(_SCENARIO_IDEMPOTENCY_PREFIX))


def _is_building_upload_task(external_id: str, task_repo: TaskRepository) -> bool:
    key = task_repo.get_idempotency_key_by_external_id(external_id)
    return bool(key and key.startswith(_BUILDING_IDEMPOTENCY_PREFIX))


def get_public_task_or_404(external_id: str, task_repo: TaskRepository) -> PipelineTask:
    task = get_task_or_404(external_id, task_repo)
    if _is_scenario_task(external_id, task_repo):
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.get("/tasks/{external_id}", response_model=TaskOut)
def get_task_endpoint(
    external_id: str,
    task_repo: TaskRepository = Depends(get_task_repo),
) -> TaskOut:
    task = get_public_task_or_404(external_id, task_repo)
    api_log("get_task", "found", task_id=task.id, external_id=task.external_id)
    return TaskOut.model_validate(task)


def build_cancel_task_response(
    task: PipelineTask,
    task_repo: TaskRepository,
    event_repo: EventRepository,
    session: Session,
) -> TaskOut:
    """Cancel an already-authorized task; shared logic between routers."""
    if task.status in {TaskStatus.finished, TaskStatus.failed}:
        raise HTTPException(status_code=409, detail=f"Task already in terminal state: {task.status.value}")

    if task.celery_task_id:
        celery_app.control.revoke(task.celery_task_id, terminate=True, signal="SIGTERM")

    if task.status in {TaskStatus.queued, TaskStatus.waiting_capacity}:
        ensure_transition(task.status.value, TaskStatus.failed.value)
        task_repo.update_status(task.id, TaskStatus.failed, finished_at=utc_now())
        task_repo.set_error(task.id, "Cancelled by client")
        event_repo.append_event(task_id=task.id, stage="api", status="cancelled")

    session.flush()
    session.refresh(task)
    api_log("cancel_task", "ok", task_id=task.id, external_id=task.external_id)
    return TaskOut.model_validate(task)


def build_recompute_task_response(
    task: PipelineTask,
    task_repo: TaskRepository,
    event_repo: EventRepository,
    session: Session,
) -> TaskOut:
    """Re-enqueue an already-authorized task; shared logic.

    Allowed for all non-running states.  For tasks stuck in ``queued`` or
    ``waiting_capacity`` (e.g. Celery message was lost) the old Celery task
    is revoked before a fresh one is enqueued.
    """
    _rerunnable = {TaskStatus.finished, TaskStatus.failed, TaskStatus.queued, TaskStatus.waiting_capacity}
    if task.status not in _rerunnable:
        raise HTTPException(
            status_code=409,
            detail=f"Task is currently running; cancel it first",
        )

    # Revoke the stale Celery message so it doesn't race with the new one.
    if task.celery_task_id and task.status in {TaskStatus.queued, TaskStatus.waiting_capacity}:
        try:
            celery_app.control.revoke(task.celery_task_id)
        except Exception:  # noqa: BLE001
            pass

    task_repo.set_error(task.id, None)
    task_repo.set_result(task.id, None)
    task.started_at = None
    task.finished_at = None
    task_repo.update_status(task.id, TaskStatus.queued)
    session.commit()

    try:
        celery_result = enqueue_pipeline_task(
            task.id,
            is_scenario=_is_scenario_task(task.external_id, task_repo),
            is_building_upload=_is_building_upload_task(task.external_id, task_repo),
        )
    except Exception as exc:  # noqa: BLE001
        task_repo.update_status(task.id, TaskStatus.failed, finished_at=utc_now())
        task_repo.set_error(task.id, f"Failed to enqueue Celery task: {exc}")
        event_repo.append_event(
            task_id=task.id,
            stage="queue",
            status="recompute_enqueue_error",
            details=str(exc),
        )
        raise HTTPException(status_code=503, detail=f"Failed to enqueue: {exc}") from exc

    celery_task_id = getattr(celery_result, "id", None)
    task_repo.update_status(task.id, TaskStatus.queued, celery_task_id=celery_task_id)
    event_repo.append_event(
        task_id=task.id,
        stage="queue",
        status="recomputed",
        details=f"celery_id={celery_task_id}",
    )

    session.flush()
    session.refresh(task)
    api_log("recompute_task", "ok", task_id=task.id, external_id=task.external_id)
    return TaskOut.model_validate(task)


def build_task_events_response(task: PipelineTask, session: Session) -> list[TaskEventOut]:
    """Return events for an already-authorized task; shared logic."""
    events = session.execute(
        select(TaskEvent)
        .where(TaskEvent.task_id == task.id)
        .order_by(TaskEvent.created_at.asc(), TaskEvent.id.asc())
    ).scalars().all()
    return [TaskEventOut.model_validate(event) for event in events]


@router.delete("/tasks/{external_id}", response_model=TaskOut)
def cancel_task_endpoint(
    external_id: str,
    task_repo: TaskRepository = Depends(get_task_repo),
    event_repo: EventRepository = Depends(get_event_repo),
    session: Session = Depends(get_db),
) -> TaskOut:
    """Cancel a queued or running task.

    For ``queued`` / ``waiting_capacity`` the message is revoked from the
    broker. For ``running`` the worker process is terminated (SIGTERM via
    Celery's revoke ``terminate=True``); the worker's signal handler will
    write the ``failed`` status and release capacity in Phase 3.
    """
    task = get_public_task_or_404(external_id, task_repo)
    return build_cancel_task_response(task, task_repo, event_repo, session)


@router.post("/tasks/{external_id}/recompute", response_model=TaskOut)
def recompute_task_endpoint(
    external_id: str,
    task_repo: TaskRepository = Depends(get_task_repo),
    event_repo: EventRepository = Depends(get_event_repo),
    session: Session = Depends(get_db),
) -> TaskOut:
    """Force-recompute a task that has already reached a terminal state.

    Re-enqueues the existing task (same ``external_id``) using its stored
    input paths — no re-upload required. Allowed only for ``finished`` and
    ``failed`` tasks; active tasks (``queued`` / ``running`` /
    ``waiting_capacity``) return 409 Conflict.

    Clears ``error_text``, ``result_path`` and ``finished_at`` so the new
    run starts from a clean slate.
    """
    task = get_public_task_or_404(external_id, task_repo)
    return build_recompute_task_response(task, task_repo, event_repo, session)


@router.get("/tasks/{external_id}/events", response_model=list[TaskEventOut])
def get_task_events_endpoint(
    external_id: str,
    task_repo: TaskRepository = Depends(get_task_repo),
    session: Session = Depends(get_db),
) -> list[TaskEventOut]:
    task = get_public_task_or_404(external_id, task_repo)
    return build_task_events_response(task, session)


@router.get("/tasks_list", response_model=TaskListOut)
def list_tasks_endpoint(
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
    task_repo: TaskRepository = Depends(get_task_repo),
) -> TaskListOut:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    parsed_status = None
    if status is not None:
        try:
            parsed_status = TaskStatus(status)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid status filter") from exc

    items, total = task_repo.list_tasks(status=parsed_status, limit=limit, offset=offset)
    return TaskListOut(
        items=[TaskOut.model_validate(task) for task in items],
        total=total,
        limit=limit,
        offset=offset,
    )


async def _task_sse_generator(
    external_id: str,
    poll_interval: float,
    request: Request,
) -> AsyncIterator[ServerSentEvent]:
    last_event_id = 0
    last_status: TaskStatus | None = None

    while True:
        if await request.is_disconnected():
            break

        with session_scope() as session:
            task = session.execute(
                select(PipelineTask).where(PipelineTask.external_id == external_id)
            ).scalar_one_or_none()

            if task is None:
                yield ServerSentEvent(data=json.dumps({"error": "Task not found"}), event="error")
                break

            new_events = session.execute(
                select(TaskEvent)
                .where(TaskEvent.task_id == task.id, TaskEvent.id > last_event_id)
                .order_by(TaskEvent.id.asc())
            ).scalars().all()

            for ev in new_events:
                last_event_id = ev.id
                yield ServerSentEvent(
                    data=json.dumps(TaskEventOut.model_validate(ev).model_dump(mode="json")),
                    event="task_event",
                )

            current_status = task.status
            if current_status != last_status:
                last_status = current_status
                yield ServerSentEvent(
                    data=json.dumps(TaskOut.model_validate(task).model_dump(mode="json")),
                    event="status",
                )

            if current_status in _TERMINAL_STATUSES:
                yield ServerSentEvent(
                    data=json.dumps({"status": current_status.value}),
                    event="done",
                )
                break

        await asyncio.sleep(poll_interval)


async def task_stream_with_report_generator(
    external_id: str,
    *,
    group_by: str,
    poll_interval: float,
    request: Request,
    app_settings: Settings,
    initial: dict[str, Any],
    include_report: bool = True,
    emit_input_files: bool = False,
) -> AsyncIterator[ServerSentEvent]:
    """Stream a task's lifecycle and, on success, the object-zone-fit report.

    Used by the combined "create + stream" scenario endpoint. Emits:
      - ``task``        once, upfront, with the created task descriptor (so a
                        client that drops can reconnect to /stream by external_id);
      - ``file``        (upload flow) links to uploaded input layers, once,
                        early; and the result layer when finished;
      - ``task_event``  per new pipeline event;
      - ``status``      on each status change;
      - ``geojson``     the classified result FeatureCollection (geometry +
                        verdict properties) when the task finishes;
      - ``report``      the object-zone-fit summary when finished (skipped when
                        ``include_report`` is False, e.g. classify-only runs
                        that have no zones);
      - ``done``        terminal marker, then the stream closes.
    """
    yield ServerSentEvent(data=json.dumps(initial), event="task")

    last_event_id = 0
    last_status: TaskStatus | None = None
    inputs_emitted = False
    while True:
        if await request.is_disconnected():
            break

        with session_scope() as session:
            task = session.execute(
                select(PipelineTask).where(PipelineTask.external_id == external_id)
            ).scalar_one_or_none()
            if task is None:
                yield ServerSentEvent(data=json.dumps({"error": "Task not found"}), event="error")
                break

            if emit_input_files and not inputs_emitted:
                inputs_emitted = True
                for layer in build_input_geo_layers(task, external_id, app_settings, request):
                    yield ServerSentEvent(
                        data=json.dumps({"type": "file", "content": layer}),
                        event="file",
                    )

            new_events = session.execute(
                select(TaskEvent)
                .where(TaskEvent.task_id == task.id, TaskEvent.id > last_event_id)
                .order_by(TaskEvent.id.asc())
            ).scalars().all()
            for ev in new_events:
                last_event_id = ev.id
                yield ServerSentEvent(
                    data=json.dumps(TaskEventOut.model_validate(ev).model_dump(mode="json")),
                    event="task_event",
                )

            current_status = task.status
            if current_status != last_status:
                last_status = current_status
                yield ServerSentEvent(
                    data=json.dumps(TaskOut.model_validate(task).model_dump(mode="json")),
                    event="status",
                )

            if current_status in _TERMINAL_STATUSES:
                if current_status == TaskStatus.finished:
                    try:
                        if task.result_path:
                            geojson = _load_result_geojson(
                                task.result_path, app_settings.outputs_dir
                            )
                            yield ServerSentEvent(
                                data=json.dumps(geojson), event="geojson"
                            )
                        if include_report:
                            report = build_object_zone_fit_response(
                                task, external_id, group_by, app_settings
                            )
                            yield ServerSentEvent(data=json.dumps(report), event="report")
                    except HTTPException as exc:
                        yield ServerSentEvent(
                            data=json.dumps({"error": exc.detail}), event="error"
                        )
                    # Durable link(s) to the result layer(s) (alongside inline geojson,
                    # so the frontend can switch to download-by-link for big files).
                    # building_pzz_check yields two — здания + сервисы.
                    for layer in build_result_geo_layers(task, external_id, app_settings, request):
                        yield ServerSentEvent(
                            data=json.dumps({"type": "file", "content": layer}),
                            event="file",
                        )
                yield ServerSentEvent(
                    data=json.dumps({"status": current_status.value}),
                    event="done",
                )
                break

        await asyncio.sleep(poll_interval)


async def _stream_chat_answer_managed(
    app_settings: Settings,
    *,
    user_id: str | None,
    user_query: str,
    classification_context: str,
    chat_id: str | None,
    scenario_id: int | str | None = None,
    project_id: int | str | None = None,
    chat_title: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    assistant_file_parts: list[dict[str, Any]] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Open the chat clients and delegate to ``stream_chat_answer``.

    Persistence is enabled only when ChatStorage + the Keycloak service token
    client are configured (``build_chat_storage_client``) and a ``user_id`` is
    present; otherwise the answer streams without being stored. The endpoint
    layer owns client lifetimes via ``async with`` here so the use-case stays
    pure and testable.
    """
    system_prompt = load_system_prompt(app_settings.chat_system_prompt_path)

    async with build_ollama_chat_client(app_settings) as ollama_client:
        chat_storage_client = (
            build_chat_storage_client(app_settings) if user_id else None
        )
        try:
            async for event in stream_chat_answer(
                ollama_client=ollama_client,
                chat_storage_client=chat_storage_client,
                user_id=user_id,
                system_prompt=system_prompt,
                user_query=user_query,
                classification_context=classification_context,
                chat_id=chat_id,
                scenario_id=scenario_id,
                project_id=project_id,
                chat_title=chat_title,
                model=model,
                temperature=temperature,
                assistant_file_parts=assistant_file_parts,
            ):
                yield event
        finally:
            if chat_storage_client is not None:
                await chat_storage_client.__aexit__(None, None, None)


def _chat_event_to_sse(event: dict[str, Any]) -> ServerSentEvent | None:
    """Map an abstract chat-answer event to a gMART-format SSE event.

    Uses gMART's ``{"type", "content"}`` envelope. Returns None for events that
    don't map to an SSE message (``done`` is folded into the generator's own
    terminal ``done`` event).
    """
    kind = event["type"]
    if kind == "chat_created":
        return ServerSentEvent(
            data=json.dumps(
                {
                    "type": "service_event",
                    "content": {
                        "event_type": "storage_event",
                        "event": {
                            "storage_event_type": "chat_created",
                            "chat_id": event.get("chat_id"),
                            "chat_title": event.get("title"),
                        },
                    },
                }
            ),
            event="service_event",
        )
    if kind == "token":
        return ServerSentEvent(
            data=json.dumps(
                {"type": "chunk", "content": {"text": event["content"], "done": False}}
            ),
            event="chunk",
        )
    if kind == "warning":
        # Non-fatal: the answer streamed fine, but chat history couldn't be
        # persisted/loaded. Distinct event so the frontend doesn't treat it as a
        # service failure (it can show a soft notice instead).
        return ServerSentEvent(
            data=json.dumps(
                {
                    "type": "warning",
                    "content": {
                        "message": event.get("message") or event.get("detail"),
                        "stage": event.get("stage"),
                        "detail": event.get("detail"),
                    },
                }
            ),
            event="warning",
        )
    if kind == "error":
        return ServerSentEvent(
            data=json.dumps(
                {
                    "type": "error",
                    "content": {"message": event.get("detail"), "stage": event.get("stage")},
                }
            ),
            event="error",
        )
    return None


def _final_answer_chunk_sse() -> ServerSentEvent:
    """gMART-style end-of-answer marker: an empty chunk with ``done: true``."""
    return ServerSentEvent(
        data=json.dumps({"type": "chunk", "content": {"text": "", "done": True}}),
        event="chunk",
    )


def narrative_chunk_sse(text: str) -> ServerSentEvent:
    """A gMART ``chunk`` SSE carrying agent narrative text (not end-of-answer)."""
    return ServerSentEvent(
        data=json.dumps({"type": "chunk", "content": {"text": text, "done": False}}),
        event="chunk",
    )


async def prepend_narrative_generator(
    narrative: str,
    inner: AsyncIterator[ServerSentEvent],
) -> AsyncIterator[ServerSentEvent]:
    """Emit a leading narrative chunk, then delegate to ``inner``.

    Used by the auto-detect chat endpoint so the agent announces the detected
    columns (\"поле X определено как …\") before the task lifecycle starts.

    A trailing blank line is appended so the streamed answer that follows starts
    as a separate paragraph instead of being glued to the last narrative line.
    """
    if narrative:
        yield narrative_chunk_sse(narrative + "\n\n")
    async for event in inner:
        yield event


async def detection_failed_generator(
    narrative: str,
    detail: str,
) -> AsyncIterator[ServerSentEvent]:
    """Announce detection results + an error, without running a task.

    Emitted when a required column could not be determined: the frontend shows
    the narrative (which columns are missing) and the user can retry with the
    columns specified.
    """
    if narrative:
        yield narrative_chunk_sse(narrative)
    yield ServerSentEvent(
        data=json.dumps(
            {"type": "error", "content": {"message": detail, "stage": "detect_columns"}}
        ),
        event="error",
    )
    yield _final_answer_chunk_sse()
    yield ServerSentEvent(
        data=json.dumps({"status": "detection_failed", "chat_id": None}),
        event="done",
    )


async def task_stream_with_chat_generator(
    external_id: str,
    *,
    group_by: str,
    poll_interval: float,
    request: Request,
    app_settings: Settings,
    initial: dict[str, Any],
    user_id: str | None,
    user_query: str,
    chat_id: str | None = None,
    scenario_id: int | str | None = None,
    project_id: int | str | None = None,
    chat_title: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    include_report: bool = True,
    report_kind: str = "object_zone_fit",
    emit_input_files: bool = False,
) -> AsyncIterator[ServerSentEvent]:
    """Stream a task to completion, then a grounded LLM answer over its report.

    Like ``task_stream_with_report_generator``, but on success it also builds
    a report into a grounding context and streams a conversational answer
    (Ollama ``/api/chat``), persisting the user+assistant turn to ChatStorage.

    ``report_kind`` selects which report grounds the answer and which SSE event
    carries it:
      - ``"object_zone_fit"`` (default, PZZ-check): the per-zone fit report,
        emitted as ``object_zone_fit``;
      - ``"classify"`` (classify-only): the classifier-candidate summary,
        emitted as ``classify_summary`` (no zones / spatial fit).

    The conversational part mirrors gMART's frontend contract: events carry a
    ``{"type", "content"}`` envelope in the SSE ``data`` field. Emitted, in order:

      - ``task`` / ``task_event`` / ``status`` — task lifecycle (existing
        named SSE events, unchanged);
      - ``object_zone_fit`` — the report when the task finishes;
      - ``service_event`` — ``{event_type: "storage_event", event:
        {storage_event_type: "chat_created", chat_id, chat_title}}`` when a new
        chat was created;
      - ``chunk`` — ``{text, done}`` assistant deltas; a final ``{text: "",
        done: true}`` marks the end of the answer;
      - ``warning`` — ``{message, stage, detail}`` NON-fatal: the answer streamed
        fine but wasn't saved to chat history (e.g. expired token). The frontend
        should show a soft notice, not treat it as a failure;
      - ``error`` — ``{message, stage}`` FATAL: the answer couldn't be generated;
      - ``done`` — ``{status, chat_id}`` terminal marker; the stream closes.
    """
    yield ServerSentEvent(data=json.dumps(initial), event="task")

    last_event_id = 0
    last_status: TaskStatus | None = None
    classification_context = ""
    geo_layers: list[dict[str, Any]] = []
    inputs_emitted = False
    while True:
        if await request.is_disconnected():
            break

        with session_scope() as session:
            task = session.execute(
                select(PipelineTask).where(PipelineTask.external_id == external_id)
            ).scalar_one_or_none()
            if task is None:
                yield ServerSentEvent(data=json.dumps({"error": "Task not found"}), event="error")
                break

            if emit_input_files and not inputs_emitted:
                inputs_emitted = True
                for layer in build_input_geo_layers(task, external_id, app_settings, request):
                    yield ServerSentEvent(
                        data=json.dumps({"type": "file", "content": layer}),
                        event="file",
                    )

            new_events = session.execute(
                select(TaskEvent)
                .where(TaskEvent.task_id == task.id, TaskEvent.id > last_event_id)
                .order_by(TaskEvent.id.asc())
            ).scalars().all()
            for ev in new_events:
                last_event_id = ev.id
                yield ServerSentEvent(
                    data=json.dumps(TaskEventOut.model_validate(ev).model_dump(mode="json")),
                    event="task_event",
                )

            current_status = task.status
            if current_status != last_status:
                last_status = current_status
                yield ServerSentEvent(
                    data=json.dumps(TaskOut.model_validate(task).model_dump(mode="json")),
                    event="status",
                )

            if current_status not in _TERMINAL_STATUSES:
                await asyncio.sleep(poll_interval)
                continue

            # Terminal: build the grounding context + the result layer link
            # inside the session scope (need the task row), then stream the
            # answer outside any I/O on it.
            if current_status == TaskStatus.finished:
                geo_layers = build_result_geo_layers(task, external_id, app_settings, request)
                if include_report:
                    try:
                        if report_kind == "classify":
                            report = build_classify_summary_response(
                                task, external_id, app_settings
                            )
                            report_event = "classify_summary"
                        else:
                            report = build_object_zone_fit_response(
                                task, external_id, group_by, app_settings
                            )
                            report_event = "object_zone_fit"
                        yield ServerSentEvent(
                            data=json.dumps(report), event=report_event
                        )
                        classification_context = build_classification_context(
                            chat_message=report.get("chat_message"),
                            object_zone_fit=report,
                            vri_names=load_vri_names(
                                app_settings.default_vri_classifier_path
                            ),
                        )
                    except HTTPException as exc:
                        yield ServerSentEvent(
                            data=json.dumps({"error": exc.detail}), event="error"
                        )
                for geo_layer in geo_layers:
                    yield ServerSentEvent(
                        data=json.dumps({"type": "file", "content": geo_layer}),
                        event="file",
                    )
            break

    # Outside the poll loop / session scope: generate + stream the answer.
    # Conversational events use gMART's {"type", "content"} envelope.
    chat_id_final = chat_id
    streamed_answer = False
    assistant_file_parts = [geo_layer_to_file_part(layer) for layer in geo_layers] or None
    if last_status == TaskStatus.finished:
        async for event in _stream_chat_answer_managed(
            app_settings,
            user_id=user_id,
            assistant_file_parts=assistant_file_parts,
            user_query=user_query,
            classification_context=classification_context,
            chat_id=chat_id,
            scenario_id=scenario_id,
            project_id=project_id,
            chat_title=chat_title,
            model=model,
            temperature=temperature,
        ):
            kind = event["type"]
            if kind == "token":
                streamed_answer = True
            elif kind in ("chat_created", "done"):
                chat_id_final = event.get("chat_id") or chat_id_final
            sse = _chat_event_to_sse(event)
            if sse is not None:
                yield sse

        if streamed_answer:
            yield _final_answer_chunk_sse()

    terminal = last_status.value if last_status is not None else "unknown"
    yield ServerSentEvent(
        data=json.dumps({"status": terminal, "chat_id": chat_id_final}),
        event="done",
    )


@router.get("/tasks/{external_id}/stream")
async def stream_task_status_endpoint(
    request: Request,
    external_id: str,
    poll_interval: float = Query(2.0, ge=0.5, le=10.0),
    task_repo: TaskRepository = Depends(get_task_repo),
) -> EventSourceResponse:
    """Stream task status and events via Server-Sent Events.

    Pushes ``task_event`` for each new pipeline event and ``status`` on
    status changes. Sends ``done`` and closes the stream when the task
    reaches a terminal state (``finished`` or ``failed``).
    """
    get_public_task_or_404(external_id, task_repo)
    return EventSourceResponse(_task_sse_generator(external_id, poll_interval, request))


@router.get("/tasks/{external_id}/result")
def get_task_result_endpoint(
    external_id: str,
    task_repo: TaskRepository = Depends(get_task_repo),
    app_settings: Settings = Depends(get_app_settings),
) -> FileResponse:
    task = get_public_task_or_404(external_id, task_repo)
    return build_task_result_response(task, external_id, app_settings)


def build_task_result_response(
    task: PipelineTask,
    external_id: str,
    app_settings: Settings,
) -> FileResponse:
    """Build the streaming result response for an already-authorized task."""

    if task.status in {"queued", "running", "waiting_capacity"}:
        raise HTTPException(status_code=409, detail=f"Task is not ready yet (status: {task.status})")

    if task.status == "failed":
        raise HTTPException(status_code=422, detail=task.error_text or "Task execution failed")

    if task.status != "finished" or not task.result_path:
        raise HTTPException(status_code=404, detail="Task result not found")

    if is_remote_path(task.result_path):
        outputs_dir = Path(app_settings.outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        cache_filename = task.result_path.split("/")[-1]
        cache_path = outputs_dir / cache_filename
        if not cache_path.is_file():
            try:
                get_object_storage().download_file(task.result_path, str(cache_path))
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to fetch result from object storage: {exc}",
                ) from exc
        return FileResponse(
            path=str(cache_path),
            media_type="application/geo+json",
            filename=_result_label(task.include_pzz_check)[1],
        )

    selected_path = Path(task.result_path)

    outputs_dir = Path(app_settings.outputs_dir).resolve()
    resolved_path = selected_path.resolve()
    try:
        resolved_path.relative_to(outputs_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Result path is outside outputs directory") from exc

    if not resolved_path.is_file():
        raise HTTPException(status_code=404, detail="Task result file not found")

    return FileResponse(
        path=str(resolved_path),
        media_type="application/geo+json",
        filename=_result_label(task.include_pzz_check)[1],
    )


# Download "slots" exposed via /files/{slot}/{external_id}. Each maps to a task
# column holding the stored path. ``result`` is gated on a finished task; the
# input layers are available as soon as the task exists.
_FILE_SLOTS: dict[str, str] = {
    "result": "result_path",
    "cadastral": "cadastral_data_path",
    "zones": "pzz_zones_data_path",
}

# Human-readable label (``title``, RU — shown in chat/layer panel) + ASCII
# download ``filename`` per slot. The layer ``name`` stays a stable machine id
# (frontend layer key); ``title`` / ``filename`` are what the user sees and
# saves — previously both were the opaque task hash (``<external_id>.geojson``).
# Input slots map 1:1; the ``result`` slot depends on the run mode (PZZ check
# vs classify-only), so it's resolved via ``_result_label`` instead.
_SLOT_LABELS: dict[str, tuple[str, str]] = {
    # slot -> (title, filename)
    "cadastral": ("Исходные участки", "input_parcels.geojson"),
    "zones": ("Зоны ПЗЗ", "pzz_zones.geojson"),
}
_RESULT_LABEL_PZZ = ("Результат проверки ПЗЗ", "pzz_check_result.geojson")
_RESULT_LABEL_CLASSIFY = ("Результат классификации ВРИ", "classification_result.geojson")

# building_pzz_check emits the result as TWO layers — здания and сервисы — split
# from the single combined result by the «Категория_объекта» feature property.
# These slots have no task column: /files/{slot} filters the combined result on
# the fly (works for both local and MinIO storage). slot -> (category, name,
# title, filename).
_COL_CATEGORY = "Категория_объекта"
_RESULT_SPLIT_SLOTS: dict[str, tuple[str, str, str, str]] = {
    "result_buildings": ("Здание", "buildings_result", "Результат — здания", "buildings_result.geojson"),
    "result_services": ("Сервис", "services_result", "Результат — сервисы", "services_result.geojson"),
}


def _is_building_task(task: PipelineTask) -> bool:
    """True for building_pzz_check tasks (they carry detected building columns)."""
    return bool(
        getattr(task, "building_type_col", None)
        or getattr(task, "building_service_col", None)
    )


def _result_label(include_pzz_check: bool | None) -> tuple[str, str]:
    """(title, download filename) for a result layer, by run mode.

    ``None`` (an un-flushed in-memory task) maps to PZZ check to match the DB
    column default (``include_pzz_check`` defaults to True).
    """
    is_pzz = True if include_pzz_check is None else bool(include_pzz_check)
    return _RESULT_LABEL_PZZ if is_pzz else _RESULT_LABEL_CLASSIFY


def _file_durable_url(
    slot: str,
    external_id: str,
    app_settings: Settings,
    request: Request | None = None,
) -> str:
    """Stable, never-expiring URL for a task's file slot.

    Absolute when ``PUBLIC_BASE_URL`` is set (best for storing in chat history),
    otherwise derived from the request, else a relative path.
    """
    path = f"/files/{slot}/{external_id}"
    if app_settings.public_base_url:
        return f"{app_settings.public_base_url}{path}"
    if request is not None:
        return str(request.base_url).rstrip("/") + path
    return path


def _build_geo_layer(
    *,
    slot: str,
    name: str,
    title: str,
    filename: str,
    role: str,
    stored_path: str | None,
    external_id: str,
    app_settings: Settings,
    request: Request | None,
) -> dict[str, Any] | None:
    """Build one geo-layer link descriptor, or None when there's no file."""
    if not stored_path:
        return None
    download_url: str | None = None
    if is_remote_path(stored_path):
        download_url = get_object_storage().presigned_url(
            stored_path, app_settings.geo_layer_url_ttl_seconds
        )
    return {
        "name": name,
        "title": title,
        "role": role,
        "url": _file_durable_url(slot, external_id, app_settings, request),
        "download_url": download_url,
        "filename": filename,
        "mime_type": "application/geo+json",
        "source_service": app_settings.app_name,
    }


def build_result_geo_layer(
    task: PipelineTask,
    external_id: str,
    app_settings: Settings,
    request: Request | None = None,
) -> dict[str, Any] | None:
    """Geo-layer link descriptor for a finished task's result GeoJSON, or None.

    Single decision point for the result artefact — extend ``build_input_geo_layers``
    / ``_FILE_SLOTS`` to add more.
    """
    if task.status != "finished":
        return None
    title, filename = _result_label(task.include_pzz_check)
    return _build_geo_layer(
        slot="result",
        name="classified_result",
        title=title,
        filename=filename,
        role="result",
        stored_path=task.result_path,
        external_id=external_id,
        app_settings=app_settings,
        request=request,
    )


def build_result_geo_layers(
    task: PipelineTask,
    external_id: str,
    app_settings: Settings,
    request: Request | None = None,
) -> list[dict[str, Any]]:
    """Result-layer link descriptors for a finished task (one or two layers).

    building_pzz_check returns TWO layers — «здания» and «сервисы» — served by
    filtering the combined result on the fly (durable ``url`` only, no presigned
    ``download_url``). Every other mode returns the single combined result layer.
    """
    if task.status != "finished" or not task.result_path:
        return []
    if not _is_building_task(task):
        single = build_result_geo_layer(task, external_id, app_settings, request)
        return [single] if single is not None else []
    layers: list[dict[str, Any]] = []
    for slot, (_category, name, title, filename) in _RESULT_SPLIT_SLOTS.items():
        layers.append(
            {
                "name": name,
                "title": title,
                "role": "result",
                "url": _file_durable_url(slot, external_id, app_settings, request),
                "download_url": None,
                "filename": filename,
                "mime_type": "application/geo+json",
                "source_service": app_settings.app_name,
            }
        )
    return layers


def build_input_geo_layers(
    task: PipelineTask,
    external_id: str,
    app_settings: Settings,
    request: Request | None = None,
) -> list[dict[str, Any]]:
    """Geo-layer link descriptors for a task's uploaded input layers.

    Covers the cadastral parcels and PZZ zones the user uploaded (both stored
    per-task under ``inputs/``). Optional config files (labels/classifier) are
    intentionally excluded — they're often static defaults, not user uploads.
    """
    specs = (
        ("cadastral", "cadastral_data_path", "input_cadastral"),
        ("zones", "pzz_zones_data_path", "input_zones"),
    )
    layers: list[dict[str, Any]] = []
    for slot, column, name in specs:
        title, filename = _SLOT_LABELS[slot]
        layer = _build_geo_layer(
            slot=slot,
            name=name,
            title=title,
            filename=filename,
            role="input",
            stored_path=getattr(task, column, None),
            external_id=external_id,
            app_settings=app_settings,
            request=request,
        )
        if layer is not None:
            layers.append(layer)
    return layers


def geo_layer_to_file_part(layer: dict[str, Any]) -> dict[str, Any]:
    """ChatStorage ``file`` part payload from a layer descriptor.

    Stores only the durable ``url`` — the presigned ``download_url`` is
    ephemeral and must not be persisted into permanent chat history.
    """
    return {
        "url": layer["url"],
        "name": layer.get("name"),
        "title": layer.get("title"),
        "filename": layer.get("filename"),
        "mime_type": layer.get("mime_type"),
        "source_service": layer.get("source_service"),
    }


def _serve_result_split(
    task: PipelineTask, slot: str, app_settings: Settings
) -> Response:
    """Serve one half (здания / сервисы) of a building_pzz_check result.

    Loads the combined result GeoJSON (local or MinIO), filters features by their
    «Категория_объекта», and returns the filtered FeatureCollection as a download.
    Generated on the fly so no second artefact is stored.
    """
    category, _name, _title, filename = _RESULT_SPLIT_SLOTS[slot]
    if task.status != "finished" or not task.result_path:
        raise HTTPException(status_code=404, detail="Task result not available yet")
    geojson = _load_result_geojson(task.result_path, app_settings.outputs_dir)
    features = [
        f
        for f in (geojson.get("features") or [])
        if (f.get("properties") or {}).get(_COL_CATEGORY) == category
    ]
    fc = {"type": "FeatureCollection", "features": features}
    return Response(
        content=json.dumps(fc, ensure_ascii=False, default=str),
        media_type="application/geo+json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/files/{slot}/{external_id}")
def get_task_file_redirect(
    slot: str,
    external_id: str,
    task_repo: TaskRepository = Depends(get_task_repo),
    app_settings: Settings = Depends(get_app_settings),
):
    """Durable, open download link for a task's file (result or uploaded input).

    Redirects (307) to a fresh presigned MinIO URL so the link never expires
    (big files download straight from object storage); falls back to streaming
    the file when storage is local. Intentionally unauthenticated so links saved
    in chat history keep working — the ``external_id`` (a uuid) is the capability.
    """
    if slot in _RESULT_SPLIT_SLOTS:
        task = get_task_or_404(external_id, task_repo)
        return _serve_result_split(task, slot, app_settings)

    column = _FILE_SLOTS.get(slot)
    if column is None:
        raise HTTPException(status_code=404, detail=f"Unknown file slot '{slot}'")
    task = get_task_or_404(external_id, task_repo)
    if slot == "result" and task.status != "finished":
        raise HTTPException(status_code=404, detail="Task result not available yet")
    stored_path = getattr(task, column, None)
    if not stored_path:
        raise HTTPException(status_code=404, detail="File not found")

    if is_remote_path(stored_path):
        url = get_object_storage().presigned_url(
            stored_path, app_settings.geo_layer_url_ttl_seconds
        )
        if url:
            return RedirectResponse(url, status_code=307)

    if slot == "result":
        return build_task_result_response(task, external_id, app_settings)
    # Local storage fallback for input layers: serve the file under task_inputs.
    local_path = Path(stored_path).resolve()
    inputs_root = Path(app_settings.task_inputs_dir).resolve()
    try:
        local_path.relative_to(inputs_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="File path is outside inputs directory") from exc
    if not local_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=str(local_path),
        media_type="application/geo+json",
        filename=_SLOT_LABELS[slot][1],
    )


_COL_VRI_TEXT = "ВРИ_ЕГРН"
_COL_ZONE_CODE = "Код фактической зоны нахождения кадастра"
_COL_ZONE_NAME = "Название фактической зоны нахождения кадастра"
_COL_VERDICT = "Вердикт_ПЗЗ"
_COL_REASON = "Причина"
_COL_MATCHED_VRI_NAME = "Подобранный_ВРИ"
_COL_MATCHED_VRI_CODE = "Код_подобранного_ВРИ"
_COL_RESOLUTION_BASIS = "Основание_подбора_ВРИ"
_COL_TOP1_CANDIDATE = "Топ1_возможный_ВРИ"
_COL_TOP5_CANDIDATES = "Топ5_возможных_ВРИ"

# ``Вердикт_ПЗЗ`` now carries the human-readable Russian label (not the machine
# ``allowed_main``). Bucket those labels into correct/wrong/unclear.
_STATUS_CORRECT = {"Разрешен", "Условно разрешен", "Разрешен как вспомогательный"}
_STATUS_WRONG = {"Не разрешен"}


def _load_result_geojson(result_path: str, outputs_dir: str) -> dict[str, Any]:
    """Read a task result (local or MinIO) and return parsed GeoJSON dict."""
    if is_remote_path(result_path):
        cache_root = Path(outputs_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / result_path.split("/")[-1]
        if not cache_path.is_file():
            get_object_storage().download_file(result_path, str(cache_path))
        local_path = cache_path
    else:
        local_path = Path(result_path).resolve()
        if not local_path.is_file():
            raise HTTPException(status_code=404, detail="Task result file not found")
    with local_path.open("rb") as fh:
        return json.load(fh)


def _classify_verdict(status: str | None) -> str:
    """Map the Russian ``Статус`` label to one of: correct / wrong / unclear."""
    v = (status or "").strip()
    if v in _STATUS_CORRECT:
        return "correct"
    if v in _STATUS_WRONG:
        return "wrong"
    return "unclear"


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _verdict_breakdown_lines(summary: dict[str, Any]) -> list[str]:
    """Split the 'требуют ручной проверки' bucket by actual Вердикт_ПЗЗ reason.

    Reads ``summary['by_verdict']`` (exact Russian-label counts) and keeps only
    the labels outside correct/wrong — the reasons a parcel ends up needing
    manual review (e.g. «Нет пересечения с ПЗЗ», «Нет описания зоны в шаблоне»).
    Returns one '- «label»: N;' line per reason (most common first) so the answer
    can tell the user exactly which «Вердикт_ПЗЗ» value to filter by, instead of
    lumping distinct causes into one generic "требуют ручной проверки".
    """
    by_verdict = summary.get("by_verdict") or {}
    reasons = {
        label: count
        for label, count in by_verdict.items()
        if label and label not in _STATUS_CORRECT and label not in _STATUS_WRONG
    }
    if not reasons:
        return []
    lines = [
        "Причины ручной проверки (эти земельные участки можно проверить "
        "по атрибуту «Вердикт_ПЗЗ»):"
    ]
    for label, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"- «{label}»: {count};")
    return lines


def _reconciled_intro(summary: dict[str, Any]) -> list[str]:
    """Intro lines with totals that add up, using the exact counts in ``summary``.

    Fixes the "277 находятся в границах 3 зон" contradiction: the no-intersection
    parcels are reported separately (``not_in_zone``) and the real-zone count uses
    ``zones_count`` (which excludes the empty bucket), not ``len(zones)``.
    """
    total = summary["total"]
    correct = summary["in_correct_zone"]
    wrong = summary["in_wrong_zone"]
    unclear = summary["unclear"]
    zones_count = summary.get("zones_count", 0)
    not_in_zone = summary.get("not_in_zone", 0)

    intro = f"Проверено земельных участков: {total}."
    if not_in_zone:
        intro += (
            f" Из них {total - not_in_zone} находятся в границах {zones_count} "
            f"территориальных зон ПЗЗ, {not_in_zone} не пересеклись ни с одной "
            "зоной ПЗЗ."
        )
    elif zones_count:
        intro += f" Все они находятся в границах {zones_count} территориальных зон ПЗЗ."
    return [
        "ВРИ — вид разрешённого использования земельного участка; ПЗЗ — правила "
        "землепользования и застройки.",
        "",
        intro,
        "Результат проверки соответствия ВРИ каждого земельного участка правилам "
        f"его территориальной зоны ПЗЗ ({correct} + {wrong} + {unclear} = {total}):",
        f"- ВРИ допустим, нарушений ПЗЗ нет: {correct};",
        f"- ВРИ не соответствует зоне ПЗЗ (потенциальное нарушение): {wrong};",
        f"- требуют ручной проверки: {unclear}.",
    ]


def _build_chat_message_objects(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """Chatbot-friendly plain-text summary for group_by=object."""
    lines = _reconciled_intro(summary)
    if summary["unclear"]:
        breakdown = _verdict_breakdown_lines(summary)
        if breakdown:
            lines += ["", *breakdown]

    wrong = [r for r in rows if r["fit"] == "wrong"]
    if wrong:
        lines += ["", "Земельные участки с недопустимым в их зоне ВРИ:"]
        for row in wrong[:10]:
            obj_label = row.get("vri_text") or "—"
            zone_label = row.get("zone_name") or row.get("zone_type_id") or "—"
            reason = _truncate(row.get("reason") or "причина не указана", 200)
            lines.append(
                f"- #{row['feature_index']}: «{obj_label}» в зоне «{zone_label}» — {reason}"
            )
        if len(wrong) > 10:
            lines.append(
                f"...и ещё {len(wrong) - 10} земельных участков с недопустимым "
                "в их зоне ВРИ."
            )
    elif not summary["unclear"]:
        lines += ["", "У всех земельных участков ВРИ допустим в их территориальной зоне."]

    return "\n".join(lines)


def _build_chat_message_zones(zones: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """Chatbot-friendly plain-text summary for group_by=zone.

    Compact headline only — totals, per-verdict counts and manual-review reasons.
    The per-zone breakdown deliberately lives in the conversational LLM answer
    (grounded on the object-zone-fit report) so the two messages complement each
    other instead of duplicating. ``zones`` is kept in the signature for a
    uniform call site with the object variant.
    """
    lines = _reconciled_intro(summary)
    if summary["unclear"]:
        breakdown = _verdict_breakdown_lines(summary)
        if breakdown:
            lines += ["", *breakdown]
    return "\n".join(lines)


@router.get("/tasks/{external_id}/object-zone-fit")
def get_object_zone_fit_endpoint(
    external_id: str,
    group_by: str = Query("zone", pattern="^(zone|object)$"),
    task_repo: TaskRepository = Depends(get_task_repo),
    app_settings: Settings = Depends(get_app_settings),
) -> dict[str, Any]:
    """Aggregate a finished task's per-object verdicts into a structured view.

    Reads the result GeoJSON (downloaded from MinIO if needed), extracts
    per-feature verdict / zone / reason, and:

    - **group_by=zone** (default): groups objects by their actual zone,
      attaches the PZZ справка via the functional_zone → PZZ mapping.
    - **group_by=object**: returns a flat list of objects.

    Returns 409 if the task isn't finished; 404 if not found.
    Objects are identified by ``feature_index`` (their position in the
    result GeoJSON) — the pipeline drops upstream IDs.
    """
    task = get_public_task_or_404(external_id, task_repo)
    return build_object_zone_fit_response(task, external_id, group_by, app_settings)


def build_object_zone_fit_response(
    task: PipelineTask,
    external_id: str,
    group_by: str,
    app_settings: Settings,
) -> dict[str, Any]:
    """Build the object-zone fit payload for an already-authorized task."""
    if task.status != "finished":
        raise HTTPException(
            status_code=409,
            detail=f"Task is not finished (status: {task.status})",
        )
    if not task.result_path:
        raise HTTPException(status_code=404, detail="Task has no result")

    try:
        geojson = _load_result_geojson(task.result_path, app_settings.outputs_dir)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load result GeoJSON: {exc}",
        ) from exc

    rows: list[dict[str, Any]] = []
    for idx, feature in enumerate(geojson.get("features") or []):
        props = feature.get("properties") or {}
        verdict = props.get(_COL_VERDICT)
        fit = _classify_verdict(verdict)
        rows.append({
            "feature_index": idx,
            "vri_text": props.get(_COL_VRI_TEXT),
            "zone_type_id": props.get(_COL_ZONE_CODE),
            "zone_name": props.get(_COL_ZONE_NAME),
            "verdict": verdict,
            "is_in_correct_zone": fit == "correct",
            "fit": fit,
            "reason": props.get(_COL_REASON),
            "matched_vri_name": props.get(_COL_MATCHED_VRI_NAME),
            "matched_vri_code": props.get(_COL_MATCHED_VRI_CODE),
            "resolution_basis": props.get(_COL_RESOLUTION_BASIS),
        })

    by_verdict: dict[str, int] = {}
    for r in rows:
        label = r["verdict"] or "—"
        by_verdict[label] = by_verdict.get(label, 0) + 1
    summary = {
        "total": len(rows),
        "in_correct_zone": sum(1 for r in rows if r["fit"] == "correct"),
        "in_wrong_zone": sum(1 for r in rows if r["fit"] == "wrong"),
        "unclear": sum(1 for r in rows if r["fit"] == "unclear"),
        # Number of REAL territorial zones the parcels fall into (excludes the
        # "no zone" bucket) and how many parcels fell outside every zone — so the
        # answer can reconcile totals instead of counting the empty-zone bucket.
        "zones_count": len({r["zone_type_id"] for r in rows if r["zone_type_id"]}),
        "not_in_zone": sum(1 for r in rows if not r["zone_type_id"]),
        # Exact per-verdict breakdown (Russian labels) so the answer can split
        # "требуют ручной проверки" by reason without recomputing.
        "by_verdict": by_verdict,
    }

    if group_by == "object":
        return {
            "task_external_id": external_id,
            "group_by": "object",
            "summary": summary,
            "chat_message": _build_chat_message_objects(rows, summary),
            "objects": rows,
        }

    zones_by_id: dict[Any, dict[str, Any]] = {}
    for row in rows:
        z_id = row["zone_type_id"] or "__no_zone__"
        bucket = zones_by_id.get(z_id)
        if bucket is None:
            bucket = {
                "zone_type_id": row["zone_type_id"],
                "zone_name": row["zone_name"],
                "pzz_summary": lookup_zone_summary(row["zone_type_id"]),
                "objects": [],
                "summary": {"total": 0, "in_correct_zone": 0, "in_wrong_zone": 0, "unclear": 0},
            }
            zones_by_id[z_id] = bucket
        bucket["objects"].append(row)
        bucket["summary"]["total"] += 1
        if row["fit"] == "correct":
            bucket["summary"]["in_correct_zone"] += 1
        elif row["fit"] == "wrong":
            bucket["summary"]["in_wrong_zone"] += 1
        else:
            bucket["summary"]["unclear"] += 1

    zones_list = sorted(
        zones_by_id.values(),
        key=lambda z: (-z["summary"]["in_wrong_zone"], -z["summary"]["total"]),
    )
    return {
        "task_external_id": external_id,
        "group_by": "zone",
        "summary": summary,
        "chat_message": _build_chat_message_zones(zones_list, summary),
        "zones": zones_list,
    }


def _build_chat_message_classify(rows: list[dict[str, Any]], summary: dict[str, int]) -> str:
    """Chatbot-friendly plain-text summary for a classify-only run.

    Classify-only has no zones / spatial verdict — each object just gets a
    ranked list of classifier VRI candidates. We surface the top candidate per
    object and flag the ones where the classifier found nothing.
    """
    lines = [
        f"Классифицировано объектов: {summary['total']}.",
        "",
        f"С подобранным ВРИ: {summary['with_candidate']}.",
        f"Без однозначного кандидата: {summary['without_candidate']}.",
    ]

    sample = rows[:10]
    if sample:
        lines += ["", "Подобранные ВРИ:"]
        for row in sample:
            obj_label = row.get("vri_text") or "—"
            matched = row.get("matched_vri") or "кандидат не найден"
            lines.append(f"- #{row['feature_index']}: «{obj_label}» → {_truncate(matched, 200)}")
        if len(rows) > 10:
            lines.append(f"...и ещё {len(rows) - 10} объектов.")

    return "\n".join(lines)


@router.get("/tasks/{external_id}/classify-summary")
def get_classify_summary_endpoint(
    external_id: str,
    task_repo: TaskRepository = Depends(get_task_repo),
    app_settings: Settings = Depends(get_app_settings),
) -> dict[str, Any]:
    """Aggregate a finished classify-only task's per-object candidates.

    The classify-only counterpart of ``/object-zone-fit``: reads the result
    GeoJSON and returns, per feature, the original VRI text plus the top-1 and
    top-5 classifier candidates. Returns 409 if the task isn't finished; 404 if
    not found.
    """
    task = get_public_task_or_404(external_id, task_repo)
    return build_classify_summary_response(task, external_id, app_settings)


def build_classify_summary_response(
    task: PipelineTask,
    external_id: str,
    app_settings: Settings,
) -> dict[str, Any]:
    """Build the classify-only candidate summary for an already-authorized task."""
    if task.status != "finished":
        raise HTTPException(
            status_code=409,
            detail=f"Task is not finished (status: {task.status})",
        )
    if not task.result_path:
        raise HTTPException(status_code=404, detail="Task has no result")

    try:
        geojson = _load_result_geojson(task.result_path, app_settings.outputs_dir)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load result GeoJSON: {exc}",
        ) from exc

    rows: list[dict[str, Any]] = []
    for idx, feature in enumerate(geojson.get("features") or []):
        props = feature.get("properties") or {}
        top1 = props.get(_COL_TOP1_CANDIDATE)
        top5 = props.get(_COL_TOP5_CANDIDATES)
        matched_vri = top1.strip() if isinstance(top1, str) and top1.strip() else None
        rows.append({
            "feature_index": idx,
            "vri_text": props.get(_COL_VRI_TEXT),
            "matched_vri": matched_vri,
            "candidates": top5 if isinstance(top5, str) and top5.strip() else None,
            "reason": props.get(_COL_REASON),
            # Reuse the object-zone-fit "fit" vocabulary so the shared grounding
            # context builder can surface the no-candidate objects on big runs.
            "fit": "matched" if matched_vri else "unclear",
        })

    summary = {
        "total": len(rows),
        "with_candidate": sum(1 for r in rows if r["fit"] == "matched"),
        "without_candidate": sum(1 for r in rows if r["fit"] == "unclear"),
    }
    return {
        "task_external_id": external_id,
        "group_by": "object",
        "summary": summary,
        "chat_message": _build_chat_message_classify(rows, summary),
        "objects": rows,
    }
