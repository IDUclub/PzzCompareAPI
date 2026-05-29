from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from service.domain.ports.task_repository import TaskNotFoundError
from service.models import PipelineTask, TaskIdempotencyKey, TaskStatus


class SqlAlchemyTaskRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, **task_data: object) -> PipelineTask:
        task = PipelineTask(**task_data)
        self.session.add(task)
        self.session.flush()
        return task

    def get_by_id(self, task_id: int) -> PipelineTask | None:
        return self.session.get(PipelineTask, task_id)

    def get_by_external_id(self, external_id: str) -> PipelineTask | None:
        return self.session.execute(
            select(PipelineTask).where(PipelineTask.external_id == external_id)
        ).scalar_one_or_none()

    def list_tasks(
        self,
        *,
        status: TaskStatus | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[PipelineTask], int]:
        filters = []
        if status is not None:
            filters.append(PipelineTask.status == status)
        public_task_filter = or_(
            TaskIdempotencyKey.key.is_(None),
            ~TaskIdempotencyKey.key.startswith("sc:"),
        )

        query = select(PipelineTask).outerjoin(
            TaskIdempotencyKey,
            TaskIdempotencyKey.task_external_id == PipelineTask.external_id,
        ).where(public_task_filter)
        count_query = select(func.count()).select_from(PipelineTask).outerjoin(
            TaskIdempotencyKey,
            TaskIdempotencyKey.task_external_id == PipelineTask.external_id,
        ).where(public_task_filter)
        for condition in filters:
            query = query.where(condition)
            count_query = count_query.where(condition)

        items = self.session.execute(
            query.order_by(PipelineTask.created_at.desc()).limit(limit).offset(offset)
        ).scalars().all()
        total = self.session.execute(count_query).scalar_one()
        return items, total

    def get_by_idempotency_key(self, key: str) -> PipelineTask | None:
        mapping = self.session.execute(
            select(TaskIdempotencyKey).where(TaskIdempotencyKey.key == key)
        ).scalar_one_or_none()
        if mapping is None:
            return None
        return self.get_by_external_id(mapping.task_external_id)

    def get_idempotency_key_by_external_id(self, external_id: str) -> str | None:
        return self.session.execute(
            select(TaskIdempotencyKey.key).where(TaskIdempotencyKey.task_external_id == external_id)
        ).scalar_one_or_none()

    def bind_idempotency_key(self, *, key: str, external_id: str) -> None:
        self.session.add(TaskIdempotencyKey(key=key, task_external_id=external_id))

    def update_status(
        self,
        task_id: int,
        status: TaskStatus,
        *,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        celery_task_id: str | None = None,
    ) -> None:
        task = self.session.get(PipelineTask, task_id)
        if task is None:
            raise TaskNotFoundError(task_id)

        task.status = status
        if started_at is not None:
            task.started_at = started_at
        if finished_at is not None:
            task.finished_at = finished_at
        if celery_task_id is not None:
            task.celery_task_id = celery_task_id

    def set_result(self, task_id: int, result_path: str | None) -> None:
        task = self.session.get(PipelineTask, task_id)
        if task is None:
            raise TaskNotFoundError(task_id)
        task.result_path = result_path

    def set_error(self, task_id: int, error_text: str | None) -> None:
        task = self.session.get(PipelineTask, task_id)
        if task is None:
            raise TaskNotFoundError(task_id)
        task.error_text = error_text
