from __future__ import annotations

from datetime import datetime
from typing import Protocol

from service.domain.task_state import TaskStatus
from service.models import PipelineTask


class TaskNotFoundError(LookupError):
    """Raised when a mutator targets a task that doesn't exist anymore."""

    def __init__(self, task_id: int) -> None:
        super().__init__(f"Task not found: id={task_id}")
        self.task_id = task_id


class TaskRepository(Protocol):
    def create(self, **task_data: object) -> PipelineTask:
        ...

    def get_by_id(self, task_id: int) -> PipelineTask | None:
        ...

    def get_by_external_id(self, external_id: str) -> PipelineTask | None:
        ...

    def get_by_idempotency_key(self, key: str) -> PipelineTask | None:
        ...

    def get_idempotency_key_by_external_id(self, external_id: str) -> str | None:
        ...

    def bind_idempotency_key(self, *, key: str, external_id: str) -> None:
        ...

    def update_status(
        self,
        task_id: int,
        status: TaskStatus,
        *,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        celery_task_id: str | None = None,
    ) -> None:
        ...

    def list_tasks(
        self,
        *,
        status: TaskStatus | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[PipelineTask], int]:
        ...

    def set_result(self, task_id: int, result_path: str | None) -> None:
        ...

    def set_error(self, task_id: int, error_text: str | None) -> None:
        ...
