from __future__ import annotations

from service.domain.ports.task_repository import TaskRepository
from service.models import PipelineTask


def get_task_by_external_id(task_repo: TaskRepository, external_id: str) -> PipelineTask | None:
    return task_repo.get_by_external_id(external_id)
