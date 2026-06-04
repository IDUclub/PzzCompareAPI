from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy.exc import IntegrityError

from service.domain.ports.event_repository import EventRepository
from service.domain.ports.task_repository import TaskRepository
from service.domain.task_state import TaskStatus
from service.models import PipelineTask
from service.schemas import TaskCreate
from service.settings import Settings


def _write_json_file(path: Path, payload: dict | list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path.resolve())


def prepare_input_paths(payload: TaskCreate, external_id: str, settings: Settings) -> dict[str, str]:
    task_dir = Path(settings.task_inputs_dir) / external_id
    task_dir.mkdir(parents=True, exist_ok=True)

    cadastral_data_path = _write_json_file(
        task_dir / "cadastral_feature_collection.geojson",
        payload.cadastral_feature_collection,
    )
    if payload.pzz_zones_feature_collection is not None:
        pzz_zones_data_path = _write_json_file(
            task_dir / "pzz_zones_feature_collection.geojson",
            payload.pzz_zones_feature_collection,
        )
    else:
        pzz_zones_data_path = ""

    if payload.pzz_zone_vri_labels is not None:
        pzz_zone_vri_labels_path = _write_json_file(
            task_dir / "pzz_zone_vri_labels.json",
            payload.pzz_zone_vri_labels,
        )
    elif payload.include_pzz_check:
        pzz_zone_vri_labels_path = str(Path(settings.default_pzz_zone_labels_path).resolve())
    else:
        pzz_zone_vri_labels_path = ""

    if payload.vri_classifier is not None:
        vri_classifier_path = _write_json_file(
            task_dir / "vri_classifier.json",
            payload.vri_classifier,
        )
    else:
        vri_classifier_path = str(Path(settings.default_vri_classifier_path).resolve())

    return {
        "cadastral_data_path": cadastral_data_path,
        "pzz_zones_data_path": pzz_zones_data_path,
        "pzz_zone_vri_labels_path": pzz_zone_vri_labels_path,
        "vri_classifier_path": vri_classifier_path,
    }


def create_task(
    *,
    payload: TaskCreate,
    settings: Settings,
    task_repo: TaskRepository,
    event_repo: EventRepository,
    enqueue_task: Callable[[int], object],
    idempotency_key: str | None = None,
    retry_failed: bool = False,
    force_recompute: bool = False,
    external_id: str | None = None,
    input_paths: dict[str, str] | None = None,
    session: Any = None,
    revoke_task: Callable[[str], object] | None = None,
) -> PipelineTask:
    """Create or re-enqueue a task.

    When the API endpoint has already streamed uploads to disk, it can pass
    the pre-generated ``external_id`` and ``input_paths`` to skip the
    ``prepare_input_paths`` re-serialization step. Tests and callers that
    don't write files themselves omit both arguments and the use case
    falls back to writing them from the parsed payload.
    """
    if idempotency_key:
        existing = task_repo.get_by_idempotency_key(idempotency_key)
        if existing is not None:
            _non_terminal = {TaskStatus.queued, TaskStatus.waiting_capacity}
            should_rerun = (
                force_recompute and existing.status in {
                    TaskStatus.failed, TaskStatus.finished,
                    TaskStatus.queued, TaskStatus.waiting_capacity,
                }
            ) or (
                retry_failed and existing.status == TaskStatus.failed
            )
            if should_rerun:
                # Revoke the old Celery message when the task is stuck in a
                # non-terminal state so the stale message doesn't race with
                # the newly enqueued one.
                if revoke_task is not None and existing.celery_task_id and existing.status in _non_terminal:
                    try:
                        revoke_task(existing.celery_task_id)
                    except Exception:  # noqa: BLE001
                        pass
                task_repo.set_error(existing.id, None)
                task_repo.set_result(existing.id, None)
                existing.started_at = None
                existing.finished_at = None
                if input_paths is not None:
                    existing.cadastral_data_path = input_paths["cadastral_data_path"]
                    existing.pzz_zones_data_path = input_paths["pzz_zones_data_path"]
                    existing.pzz_zone_vri_labels_path = input_paths["pzz_zone_vri_labels_path"]
                    existing.vri_classifier_path = input_paths["vri_classifier_path"]
                task_repo.update_status(existing.id, TaskStatus.queued)
                if session is not None:
                    session.commit()
                try:
                    celery_result = enqueue_task(existing.id)
                except Exception as exc:  # noqa: BLE001
                    task_repo.update_status(existing.id, TaskStatus.failed)
                    task_repo.set_error(existing.id, f"Failed to enqueue Celery task: {exc}")
                    event_repo.append_event(
                        task_id=existing.id,
                        stage="queue",
                        status="re_enqueue_error",
                        details=str(exc),
                    )
                    raise

                celery_task_id = getattr(celery_result, "id", None)
                task_repo.update_status(existing.id, TaskStatus.queued, celery_task_id=celery_task_id)
                trigger = "force_recompute" if force_recompute else "retry_failed"
                event_repo.append_event(
                    task_id=existing.id,
                    stage="queue",
                    status="re_enqueued",
                    details=f"trigger={trigger}; celery_id={celery_task_id}",
                )
            return existing

    if external_id is None:
        external_id = uuid4().hex
    if input_paths is None:
        input_paths = prepare_input_paths(payload, external_id, settings)

    task = task_repo.create(
        external_id=external_id,
        cadastral_data_path=input_paths["cadastral_data_path"],
        pzz_zones_data_path=input_paths["pzz_zones_data_path"],
        pzz_zone_vri_labels_path=input_paths["pzz_zone_vri_labels_path"],
        vri_classifier_path=input_paths["vri_classifier_path"],
        include_pzz_check=payload.include_pzz_check,
        cadastral_vri_col=payload.cadastral_vri_col,
        pzz_zone_code_col=payload.pzz_zone_code_col,
        pzz_zone_name_col=payload.pzz_zone_name_col,
        priority=payload.priority,
        status=TaskStatus.queued,
    )

    if idempotency_key:
        try:
            task_repo.bind_idempotency_key(key=idempotency_key, external_id=external_id)
            if session is not None:
                session.flush()
        except IntegrityError:
            if session is not None:
                session.rollback()
            winner = task_repo.get_by_idempotency_key(idempotency_key)
            if winner is not None:
                return winner
            raise

    if session is not None:
        session.commit()

    try:
        celery_result = enqueue_task(task.id)
    except Exception as exc:  # noqa: BLE001
        task_repo.update_status(task.id, TaskStatus.failed)
        task_repo.set_error(task.id, f"Failed to enqueue Celery task: {exc}")
        event_repo.append_event(
            task_id=task.id,
            stage="queue",
            status="enqueue_error",
            details=str(exc),
        )
        raise

    celery_task_id = getattr(celery_result, "id", None)
    task_repo.update_status(
        task.id,
        TaskStatus.queued,
        celery_task_id=celery_task_id,
    )
    event_repo.append_event(
        task_id=task.id,
        stage="queue",
        status="enqueued",
        details=f"celery_id={celery_task_id}",
    )

    return task
