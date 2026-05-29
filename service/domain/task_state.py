from __future__ import annotations

from enum import Enum


class TaskStatus(str, Enum):
    queued = "queued"
    waiting_capacity = "waiting_capacity"
    running = "running"
    finished = "finished"
    failed = "failed"


_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "queued":            {"waiting_capacity", "running", "failed"},
    "waiting_capacity":  {"running", "failed"},
    "running":           {"finished", "failed"},
    "failed":            {"queued"},
}


def can_transition(current: str, target: str) -> bool:
    """Return True when transition is allowed by the task state machine."""
    return target in _ALLOWED_TRANSITIONS.get(current, set())


def ensure_transition(current: str, target: str) -> None:
    """Raise ValueError for invalid task status transitions."""
    if not can_transition(current, target):
        raise ValueError(f"Invalid task status transition: {current} -> {target}")
