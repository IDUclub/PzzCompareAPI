from __future__ import annotations

from typing import Protocol


class EventRepository(Protocol):
    def append_event(
        self,
        *,
        task_id: int,
        stage: str,
        status: str,
        details: str | None = None,
    ) -> None:
        ...
