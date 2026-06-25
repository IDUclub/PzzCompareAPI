from __future__ import annotations

import json
import logging
import time
from datetime import timezone
from pathlib import Path
from time import perf_counter

from celery import Celery
from celery.exceptions import MaxRetriesExceededError
from celery.schedules import schedule
from celery.signals import after_setup_logger, worker_ready
from sqlalchemy import func, select

from .infrastructure.runners.pipeline_runner import PipelineRunnerFactory
from .application.use_cases.finish_task import finish_task
from .application.use_cases.start_task import StartTaskResult, start_task
from .db import session_scope
from .domain.ports.task_repository import TaskNotFoundError
from .infrastructure.repositories.sqlalchemy_config_repository import SqlAlchemyConfigRepository
from .infrastructure.repositories.sqlalchemy_event_repository import SqlAlchemyEventRepository
from .infrastructure.repositories.sqlalchemy_task_repository import SqlAlchemyTaskRepository
from .log_sink import setup_redis_sink
from .logging_config import setup_logging
from .metrics import queue_wait_seconds, task_fail_total, task_retry_total, task_run_seconds
from .models import PipelineTask, TaskStatus
from .settings import get_settings
from .time_utils import utc_now

settings = get_settings()
logger = logging.getLogger("service.tasks")
celery_app = Celery("pzz_pipeline", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.beat_schedule = {
    "reconcile-priority-current-sum": {
        "task": "service.tasks.reconcile_priority_current_sum",
        "schedule": schedule(run_every=settings.reconcile_interval_seconds),
    },
    "cleanup-stale-output-files": {
        "task": "service.tasks.cleanup_stale_output_files_task",
        "schedule": schedule(run_every=settings.outputs_cleanup_interval_seconds),
    },
}


@worker_ready.connect
def _setup_worker_log_sink(**kwargs) -> None:  # noqa: ARG001
    """Wire up the Redis log sink once the worker is fully started."""
    setup_redis_sink(redis_url=settings.redis_url)


@worker_ready.connect
def _start_worker_metrics_server(**kwargs) -> None:  # noqa: ARG001
    """Expose task metrics over HTTP from the worker process.

    Task metrics are recorded here, in the worker — the API's /metrics
    cannot see them (separate process, separate registry). We start a
    dedicated Prometheus exposition server so the worker can be scraped as
    its own target. Fires only inside a worker (not on plain module import
    by the API), so the API never tries to bind this port.
    """
    from prometheus_client import start_http_server

    port = settings.worker_metrics_port
    try:
        start_http_server(port)
        logger.info("worker metrics server started on :%s", port)
    except OSError as exc:  # noqa: BLE001
        logger.warning("could not start worker metrics server on :%s: %s", port, exc)


@after_setup_logger.connect
def _configure_worker_logging(logger, **kwargs) -> None:  # noqa: ARG001
    """Apply our log format after Celery has set up its own handlers.

    Calling setup_logging() at module level causes duplicate log lines: basicConfig()
    adds a StreamHandler to the root logger, then Celery's worker startup adds another.
    after_setup_logger fires once Celery is done configuring, so basicConfig(force=True)
    replaces the handler rather than stacking another one.
    """
    setup_logging(force=True)


def _log_structured(
    *,
    task_id: int,
    external_id: str | None,
    celery_task_id: str | None,
    stage: str,
    status: str,
    duration_ms: int | None = None,
    level: int = logging.INFO,
    **extra: object,
) -> None:
    payload = {
        "task_id": task_id,
        "external_id": external_id,
        "celery_task_id": celery_task_id,
        "stage": stage,
        "status": status,
        "duration_ms": duration_ms,
        **extra,
    }
    logger.log(level, json.dumps(payload, ensure_ascii=False))


@celery_app.task(
    bind=True,
    max_retries=50,
    retry_backoff=True,
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=settings.task_soft_time_limit_seconds,
    time_limit=settings.task_time_limit_seconds,
)
def execute_pipeline_task(self, task_id: int) -> None:
    """Execute pipeline task in background worker.

    Structured as three short DB sessions separated by the long pipeline run:

        Phase 1 (session)  — reserve capacity, transition to running.
        Phase 2 (no session) — run pipeline. DB connection is released here.
        Phase 3 (session)  — release capacity, write finished / failed.
    """
    celery_task_id = self.request.id
    started = perf_counter()
    external_id: str | None = None

    outcome: StartTaskResult | None = None

    with session_scope() as session:
        task = session.get(PipelineTask, task_id)
        if task is None:
            _log_structured(
                task_id=task_id, external_id=None, celery_task_id=celery_task_id,
                stage="worker", status="task_not_visible_yet", level=logging.WARNING,
            )
            raise self.retry(countdown=1)

        external_id = task.external_id
        if task.created_at is not None:
            created_at = task.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            queue_wait = (utc_now() - created_at).total_seconds()
            queue_wait_seconds.observe(max(queue_wait, 0))

        _log_structured(
            task_id=task_id, external_id=external_id, celery_task_id=celery_task_id,
            stage="worker", status="start",
        )

        task_repo = SqlAlchemyTaskRepository(session)
        config_repo = SqlAlchemyConfigRepository(session)
        event_repo = SqlAlchemyEventRepository(session)

        outcome = start_task(
            task_id=task_id,
            task_repo=task_repo,
            config_repo=config_repo,
            event_repo=event_repo,
            settings=settings,
        )

    if outcome is None:
        _log_structured(
            task_id=task_id, external_id=external_id, celery_task_id=celery_task_id,
            stage="worker", status="task_not_found", level=logging.WARNING,
        )
        return

    if outcome.retry_in_seconds > 0:
        task_retry_total.inc()
        _log_structured(
            task_id=task_id, external_id=external_id, celery_task_id=celery_task_id,
            stage="capacity", status="retry",
        )
        try:
            raise self.retry(countdown=outcome.retry_in_seconds)
        except MaxRetriesExceededError:
            try:
                with session_scope() as session:
                    task_repo = SqlAlchemyTaskRepository(session)
                    event_repo = SqlAlchemyEventRepository(session)
                    task_repo.update_status(task_id, TaskStatus.failed, finished_at=utc_now())
                    task_repo.set_error(task_id, "Max retries exceeded waiting for capacity")
                    event_repo.append_event(task_id=task_id, stage="capacity", status="max_retries_exceeded")
            except TaskNotFoundError:
                _log_structured(
                    task_id=task_id, external_id=external_id, celery_task_id=celery_task_id,
                    stage="capacity", status="task_gone", level=logging.WARNING,
                )
                return
            task_fail_total.inc()
            _log_structured(
                task_id=task_id, external_id=external_id, celery_task_id=celery_task_id,
                stage="capacity", status="max_retries_exceeded", level=logging.ERROR,
            )
            return

    assert outcome.request is not None
    output_path: str | None = None
    error_text: str | None = None

    stage_started = perf_counter()
    _log_structured(
        task_id=task_id, external_id=external_id, celery_task_id=celery_task_id,
        stage="external.pipeline", status="start",
    )

    try:
        output_path = PipelineRunnerFactory.create(settings, outcome.request).run(outcome.request)
    except Exception as exc:  # noqa: BLE001
        error_text = str(exc)
        _log_structured(
            task_id=task_id, external_id=external_id, celery_task_id=celery_task_id,
            stage="external.pipeline", status="error",
            duration_ms=int((perf_counter() - stage_started) * 1000),
            error=str(exc), level=logging.ERROR,
        )
    else:
        _log_structured(
            task_id=task_id, external_id=external_id, celery_task_id=celery_task_id,
            stage="external.pipeline", status="finished",
            duration_ms=int((perf_counter() - stage_started) * 1000),
        )

    try:
        with session_scope() as session:
            task_repo = SqlAlchemyTaskRepository(session)
            config_repo = SqlAlchemyConfigRepository(session)
            event_repo = SqlAlchemyEventRepository(session)

            finish_task(
                task_id=task_id,
                task_priority=outcome.task_priority,
                output_path=output_path,
                error_text=error_text,
                task_repo=task_repo,
                config_repo=config_repo,
                event_repo=event_repo,
            )
    except TaskNotFoundError:
        _log_structured(
            task_id=task_id, external_id=external_id, celery_task_id=celery_task_id,
            stage="worker", status="task_gone_in_finalize", level=logging.WARNING,
        )
        return

    duration_ms = int((perf_counter() - started) * 1000)
    task_run_seconds.observe(duration_ms / 1000)

    if error_text is not None:
        task_fail_total.inc()
        _log_structured(
            task_id=task_id, external_id=external_id, celery_task_id=celery_task_id,
            stage="worker", status="failed", duration_ms=duration_ms, level=logging.ERROR,
        )
    else:
        _log_structured(
            task_id=task_id, external_id=external_id, celery_task_id=celery_task_id,
            stage="worker", status="finished", duration_ms=duration_ms,
        )


@celery_app.task(name="service.tasks.reconcile_priority_current_sum")
def reconcile_priority_current_sum() -> dict[str, int]:
    """Recompute ``priority_current_sum`` from currently running tasks.

    Guards against capacity leaks when a worker dies between Phase 2 and
    Phase 3 of ``execute_pipeline_task`` — the priority reservation would
    otherwise stay on the books forever.
    """
    with session_scope() as session:
        actual_sum = session.execute(
            select(func.coalesce(func.sum(PipelineTask.priority), 0)).where(
                PipelineTask.status == TaskStatus.running
            )
        ).scalar_one()
        actual_sum = int(actual_sum)

        config_repo = SqlAlchemyConfigRepository(session)
        recorded_sum = config_repo.get_int("priority_current_sum", 0, for_update=True)
        if recorded_sum != actual_sum:
            config_repo.set_int("priority_current_sum", actual_sum)
            logger.warning(
                "priority_current_sum reconciled: recorded=%s actual=%s",
                recorded_sum,
                actual_sum,
            )

    return {"recorded": recorded_sum, "actual": actual_sum}


@celery_app.task(name="service.tasks.cleanup_stale_output_files_task")
def cleanup_stale_output_files_task() -> dict[str, int]:
    """Delete stale files in outputs_dir that aren't referenced by any task.

    Runs on celery beat schedule (was previously only at API startup, which
    meant long-running API processes never cleaned anything up).
    """
    outputs_dir = Path(settings.outputs_dir).resolve()
    if not outputs_dir.exists():
        return {"removed": 0}

    referenced_paths: set[Path] = set()
    with session_scope() as session:
        result_paths = session.execute(
            select(PipelineTask.result_path).where(PipelineTask.result_path.is_not(None))
        ).scalars().all()
        for result_path in result_paths:
            if result_path is None:
                continue
            referenced_paths.add(Path(result_path).resolve())

    expiration_ts = time.time() - (settings.outputs_cleanup_max_age_hours * 60 * 60)
    removed = 0
    for file_path in outputs_dir.rglob("*"):
        if not file_path.is_file():
            continue
        resolved = file_path.resolve()
        if resolved in referenced_paths:
            continue
        if file_path.stat().st_mtime > expiration_ts:
            continue
        file_path.unlink(missing_ok=True)
        removed += 1

    if removed:
        logger.info("Removed %s stale output files from %s", removed, outputs_dir)
    return {"removed": removed}
