from __future__ import annotations


def try_reserve(current_sum: int, task_priority: int, max_sum: int) -> tuple[bool, int]:
    """Try reserving capacity for a task priority.

    Returns (reserved, new_sum).
    """
    if current_sum + task_priority > max_sum:
        return False, current_sum
    return True, current_sum + task_priority


def release(current_sum: int, task_priority: int) -> int:
    """Release previously reserved priority capacity.

    Never returns a negative value.
    """
    return max(0, current_sum - task_priority)
