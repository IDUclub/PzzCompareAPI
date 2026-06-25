from __future__ import annotations

from dataclasses import dataclass

from service.domain.capacity_policy import release, try_reserve
from service.domain.contracts import PipelineRequest
from service.domain.ports.config_repository import ConfigRepository
from service.domain.ports.event_repository import EventRepository
from service.domain.ports.task_repository import TaskRepository
from service.domain.task_state import TaskStatus, ensure_transition
from service.settings import Settings
from service.time_utils import utc_now


@dataclass(frozen=True)
class StartTaskResult:
    task_priority: int
    request: PipelineRequest | None
    retry_in_seconds: int


def start_task(
    *,
    task_id: int,
    task_repo: TaskRepository,
    config_repo: ConfigRepository,
    event_repo: EventRepository,
    settings: Settings,
) -> StartTaskResult | None:
    """Reserve capacity and transition task to running.

    Returns:
        None                               — task not found in DB.
        StartTaskResult(retry_in > 0)      — waiting for capacity, caller should retry.
        StartTaskResult(request set)       — task started, pipeline should run.
    """
    task = task_repo.get_by_id(task_id)
    if task is None:
        return None

    task_priority = task.priority

    if task.status == TaskStatus.running:
        stale_sum = config_repo.get_int("priority_current_sum", 0)
        config_repo.set_int("priority_current_sum", release(stale_sum, task_priority))
        task_repo.update_status(task.id, TaskStatus.queued)

    if task.status == TaskStatus.failed:
        task_repo.update_status(task.id, TaskStatus.queued)
        task_repo.set_error(task.id, None)

    max_sum = config_repo.get_int("priority_max_sum", settings.priority_max_sum_default)
    current_sum = config_repo.get_int("priority_current_sum", 0, for_update=True)

    reserved, updated_sum = try_reserve(current_sum, task_priority, max_sum)
    if not reserved:
        ensure_transition(task.status.value, TaskStatus.waiting_capacity.value)
        task_repo.update_status(task.id, TaskStatus.waiting_capacity)
        event_repo.append_event(
            task_id=task.id,
            stage="capacity",
            status="wait",
            details=f"{current_sum}+{task_priority}>{max_sum}",
        )
        return StartTaskResult(task_priority=task_priority, request=None, retry_in_seconds=5)

    config_repo.set_int("priority_current_sum", updated_sum)
    ensure_transition(task.status.value, TaskStatus.running.value)
    task_repo.update_status(task.id, TaskStatus.running, started_at=utc_now())
    event_repo.append_event(task_id=task.id, stage="pipeline", status="start")

    idempotency_key = task_repo.get_idempotency_key_by_external_id(task.external_id)
    is_scenario = bool(idempotency_key and idempotency_key.startswith("sc:"))

    request = PipelineRequest(
        task_external_id=task.external_id,
        cadastral_data_path=task.cadastral_data_path,
        pzz_zones_data_path=task.pzz_zones_data_path,
        pzz_zone_vri_labels_path=task.pzz_zone_vri_labels_path,
        vri_classifier_path=task.vri_classifier_path,
        include_pzz_check=task.include_pzz_check,
        cadastral_vri_col=task.cadastral_vri_col,
        pzz_zone_code_col=task.pzz_zone_code_col,
        pzz_zone_name_col=task.pzz_zone_name_col,
        outputs_dir=settings.outputs_dir,
        is_scenario=is_scenario,
    )

    return StartTaskResult(task_priority=task_priority, request=request, retry_in_seconds=0)
