from __future__ import annotations

from sqlalchemy.orm import Session

from service.models import TaskEvent


class SqlAlchemyEventRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def append_event(
        self,
        *,
        task_id: int,
        stage: str,
        status: str,
        details: str | None = None,
    ) -> None:
        self.session.add(
            TaskEvent(
                task_id=task_id,
                stage=stage,
                status=status,
                details=details,
            )
        )
