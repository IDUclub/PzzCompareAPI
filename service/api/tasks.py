"""Task management endpoints: get / list / cancel / recompute / events / result."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.responses import FileResponse

from ..dependencies import get_app_settings, get_db, get_event_repo, get_task_repo
from ..domain.ports.event_repository import EventRepository
from ..domain.ports.task_repository import TaskRepository
from ..domain.task_state import ensure_transition
from ..infrastructure.pzz_mapping import lookup_zone_summary
from ..infrastructure.storage import get_object_storage, is_remote_path
from ..models import PipelineTask, TaskEvent, TaskStatus
from ..schemas import TaskEventOut, TaskListOut, TaskOut
from ..settings import Settings
from ..tasks import celery_app, execute_pipeline_task
from ..time_utils import utc_now
from .utils import api_log

router = APIRouter(tags=["tasks"])
_SCENARIO_IDEMPOTENCY_PREFIX = "sc:"


def get_task_or_404(external_id: str, task_repo: TaskRepository) -> PipelineTask:
    task = task_repo.get_by_external_id(external_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


def _is_scenario_task(external_id: str, task_repo: TaskRepository) -> bool:
    key = task_repo.get_idempotency_key_by_external_id(external_id)
    return bool(key and key.startswith(_SCENARIO_IDEMPOTENCY_PREFIX))


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
    """Re-enqueue an already-authorized terminal task; shared logic."""
    if task.status not in {TaskStatus.finished, TaskStatus.failed}:
        raise HTTPException(
            status_code=409,
            detail=f"Task is not in a terminal state (current: {task.status.value}); "
                   "cancel it first or wait for it to finish",
        )

    task_repo.set_error(task.id, None)
    task_repo.set_result(task.id, None)
    task.started_at = None
    task.finished_at = None
    task_repo.update_status(task.id, TaskStatus.queued)
    session.commit()

    try:
        celery_result = execute_pipeline_task.delay(task.id)
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
            filename=f"{external_id}.geojson",
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
        filename=f"{external_id}.geojson",
    )


_COL_VRI_TEXT = "ВРИ_ЕГРН"
_COL_ZONE_CODE = "Код фактической зоны нахождения кадастра"
_COL_ZONE_NAME = "Название фактической зоны нахождения кадастра"
_COL_VERDICT = "Вердикт_ПЗЗ"
_COL_REASON = "Причина"
_COL_MATCHED_VRI_NAME = "Подобранный_ВРИ"
_COL_MATCHED_VRI_CODE = "Код_подобранного_ВРИ"

_ALLOWED_VERDICTS = {"allowed_main", "allowed_conditional", "allowed_auxiliary"}
_UNCLEAR_VERDICTS = {
    "unclear", "unknown", "classifier_only", "no_actual_zone", "no_zone_metadata", ""
}


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


def _classify_verdict(verdict: str | None) -> str:
    """Map raw verdict to one of: correct / wrong / unclear."""
    v = (verdict or "").strip()
    if v in _ALLOWED_VERDICTS:
        return "correct"
    if v == "not_allowed":
        return "wrong"
    return "unclear"


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _build_chat_message_objects(rows: list[dict[str, Any]], summary: dict[str, int]) -> str:
    """Chatbot-friendly plain-text summary for group_by=object."""
    lines = [
        f"Проверено объектов: {summary['total']}.",
        "",
        f"В подходящих зонах: {summary['in_correct_zone']}.",
        f"Не в своих зонах: {summary['in_wrong_zone']}.",
    ]
    if summary["unclear"]:
        lines.append(f"Без чёткой оценки: {summary['unclear']}.")

    wrong = [r for r in rows if r["fit"] == "wrong"]
    if wrong:
        lines += ["", "Объекты не в своих зонах:"]
        for row in wrong[:10]:
            obj_label = row.get("vri_text") or "—"
            zone_label = row.get("zone_name") or row.get("zone_type_id") or "—"
            reason = _truncate(row.get("reason") or "причина не указана", 200)
            lines.append(
                f"- #{row['feature_index']}: «{obj_label}» в зоне «{zone_label}» — {reason}"
            )
        if len(wrong) > 10:
            lines.append(f"...и ещё {len(wrong) - 10} объектов не в своих зонах.")
    else:
        lines += ["", "Все объекты находятся в подходящих зонах."]

    return "\n".join(lines)


def _build_chat_message_zones(zones: list[dict[str, Any]], summary: dict[str, int]) -> str:
    """Chatbot-friendly plain-text summary for group_by=zone."""
    lines = [
        f"Проверено объектов: {summary['total']} в {len(zones)} зонах.",
        f"В подходящих зонах: {summary['in_correct_zone']}. "
        f"Не в своих зонах: {summary['in_wrong_zone']}. "
        f"Без чёткой оценки: {summary['unclear']}.",
        "",
    ]
    for zone in zones:
        z_sum = zone["summary"]
        zone_label = zone.get("zone_name") or zone.get("zone_type_id") or "—"
        total = z_sum["total"]
        wrong = z_sum["in_wrong_zone"]
        unclear_n = z_sum["unclear"]

        if wrong == 0 and unclear_n == 0:
            note = f"все {total} в порядке"
        elif wrong > 0:
            note = f"{wrong} из {total} не в своей зоне"
        else:
            note = f"{unclear_n} из {total} требуют ручной проверки"
        lines.append(f"Зона «{zone_label}»: {note}.")

        if wrong > 0:
            wrong_objs = [o for o in zone.get("objects") or [] if o["fit"] == "wrong"]
            if wrong_objs:
                first_reason = (wrong_objs[0].get("reason") or "").strip()
                if first_reason:
                    lines.append("    " + _truncate(first_reason.split(".")[0], 200))

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
        })

    summary = {
        "total": len(rows),
        "in_correct_zone": sum(1 for r in rows if r["fit"] == "correct"),
        "in_wrong_zone": sum(1 for r in rows if r["fit"] == "wrong"),
        "unclear": sum(1 for r in rows if r["fit"] == "unclear"),
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
