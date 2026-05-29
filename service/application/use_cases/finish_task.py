from __future__ import annotations

from service.domain.capacity_policy import release
from service.domain.ports.config_repository import ConfigRepository
from service.domain.ports.event_repository import EventRepository
from service.domain.ports.task_repository import TaskRepository
from service.domain.task_state import TaskStatus, ensure_transition
from service.time_utils import utc_now

_EVENT_DETAILS_MAX_LEN = 10_000


def finish_task(
    *,
    task_id: int,
    task_priority: int,
    output_path: str | None,
    error_text: str | None,
    task_repo: TaskRepository,
    config_repo: ConfigRepository,
    event_repo: EventRepository,
) -> None:
    """Release capacity and finalize task status.

    Called after the pipeline has run (success or failure). The task must be
    in ``running`` status at this point — ``start_task`` ensures that.
    """
    current_sum = config_repo.get_int("priority_current_sum", 0)
    config_repo.set_int("priority_current_sum", release(current_sum, task_priority))

    if error_text is not None:
        ensure_transition(TaskStatus.running.value, TaskStatus.failed.value)
        task_repo.update_status(task_id, TaskStatus.failed, finished_at=utc_now())
        task_repo.set_error(task_id, error_text)
        event_repo.append_event(
            task_id=task_id,
            stage="pipeline",
            status="failed",
            details=error_text[:_EVENT_DETAILS_MAX_LEN],
        )
    else:
        ensure_transition(TaskStatus.running.value, TaskStatus.finished.value)
        task_repo.update_status(task_id, TaskStatus.finished, finished_at=utc_now())
        task_repo.set_result(task_id, output_path)
        event_repo.append_event(
            task_id=task_id,
            stage="pipeline",
            status="finished",
            details=output_path,
        )
