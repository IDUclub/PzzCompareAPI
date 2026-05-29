from service.domain.capacity_policy import release, try_reserve
from service.domain.task_state import ensure_transition


def test_try_reserve_rejects_on_limit_exceeded():
    reserved, updated_sum = try_reserve(current_sum=8, task_priority=5, max_sum=10)

    assert reserved is False
    assert updated_sum == 8


def test_release_after_success_or_fail_is_correct():
    assert release(current_sum=7, task_priority=3) == 4


def test_release_never_goes_negative():
    assert release(current_sum=1, task_priority=3) == 0


def test_invalid_transition_is_blocked():
    try:
        ensure_transition("queued", "finished")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid transition")
