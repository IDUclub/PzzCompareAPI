from dataclasses import dataclass
from datetime import datetime

from service.application.use_cases.create_task import create_task
from service.models import TaskStatus
from service.schemas import TaskCreate
from service.settings import Settings


@dataclass
class FakeTask:
    id: int
    external_id: str
    cadastral_data_path: str
    pzz_zones_data_path: str
    pzz_zone_vri_labels_path: str
    vri_classifier_path: str
    cadastral_vri_col: str
    pzz_zone_code_col: str
    pzz_zone_name_col: str
    priority: int
    status: TaskStatus
    include_pzz_check: bool = True
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result_path: str | None = None
    error_text: str | None = None
    celery_task_id: str | None = None


class Repo:
    def __init__(self):
        self.tasks = {}
        self.next_id = 1
        self.keys = {}

    def create(self, **task_data):
        task = FakeTask(id=self.next_id, **task_data)
        self.tasks[task.id] = task
        self.next_id += 1
        return task

    def get_by_idempotency_key(self, key: str):
        external_id = self.keys.get(key)
        if not external_id:
            return None
        return next((t for t in self.tasks.values() if t.external_id == external_id), None)

    def bind_idempotency_key(self, *, key: str, external_id: str):
        self.keys[key] = external_id

    def update_status(self, task_id: int, status: TaskStatus, **kwargs):
        task = self.tasks[task_id]
        task.status = status
        for k, v in kwargs.items():
            if v is not None:
                setattr(task, k, v)

    def set_error(self, task_id: int, error_text: str | None):
        self.tasks[task_id].error_text = error_text

    def set_result(self, task_id: int, result_path: str | None):
        self.tasks[task_id].result_path = result_path


class EventRepo:
    def __init__(self):
        self.events = []

    def append_event(self, **kwargs):
        self.events.append(kwargs)


class Result:
    id = "celery-id"


def _payload() -> TaskCreate:
    return TaskCreate(
        cadastral_feature_collection={"type": "FeatureCollection", "features": []},
        pzz_zones_feature_collection={"type": "FeatureCollection", "features": []},
        cadastral_vri_col="vri",
    )


def _settings(tmp_path) -> Settings:
    return Settings(
        task_inputs_dir=str(tmp_path / "inputs"),
        default_pzz_zone_labels_path=str(tmp_path / "labels.json"),
        default_vri_classifier_path=str(tmp_path / "classifier.json"),
    )


def test_repeated_request_with_same_idempotency_key_returns_existing_task(tmp_path):
    repo = Repo()
    events = EventRepo()
    called = []

    def enqueue(task_id: int):
        called.append(task_id)
        return Result()

    first = create_task(payload=_payload(), settings=_settings(tmp_path), task_repo=repo, event_repo=events, enqueue_task=enqueue, idempotency_key="k-1")
    second = create_task(payload=_payload(), settings=_settings(tmp_path), task_repo=repo, event_repo=events, enqueue_task=enqueue, idempotency_key="k-1")

    assert first.id == second.id
    assert called == [first.id]


def test_client_network_retry_does_not_enqueue_duplicate_celery_task(tmp_path):
    repo = Repo()
    events = EventRepo()
    enqueue_count = 0

    def enqueue(_task_id: int):
        nonlocal enqueue_count
        enqueue_count += 1
        return Result()

    create_task(payload=_payload(), settings=_settings(tmp_path), task_repo=repo, event_repo=events, enqueue_task=enqueue, idempotency_key="retry-202")
    create_task(payload=_payload(), settings=_settings(tmp_path), task_repo=repo, event_repo=events, enqueue_task=enqueue, idempotency_key="retry-202")

    assert enqueue_count == 1


def test_failed_task_with_same_idempotency_key_is_reenqueued(tmp_path):
    repo = Repo()
    events = EventRepo()
    enqueue_count = 0

    def enqueue(task_id: int):
        nonlocal enqueue_count
        enqueue_count += 1
        return Result()

    task = create_task(payload=_payload(), settings=_settings(tmp_path), task_repo=repo, event_repo=events, enqueue_task=enqueue, idempotency_key="failed-1")
    repo.update_status(task.id, TaskStatus.failed)
    repo.set_error(task.id, "boom")

    not_retried = create_task(payload=_payload(), settings=_settings(tmp_path), task_repo=repo, event_repo=events, enqueue_task=enqueue, idempotency_key="failed-1")
    assert not_retried.id == task.id
    assert enqueue_count == 1
    assert repo.tasks[task.id].status == TaskStatus.failed

    retried = create_task(
        payload=_payload(),
        settings=_settings(tmp_path),
        task_repo=repo,
        event_repo=events,
        enqueue_task=enqueue,
        idempotency_key="failed-1",
        retry_failed=True,
    )

    assert retried.id == task.id
    assert enqueue_count == 2
    assert repo.tasks[task.id].status == TaskStatus.queued
    assert repo.tasks[task.id].error_text is None


def test_queued_task_is_reenqueued_on_force_recompute(tmp_path):
    """A task stuck in queued (e.g. Celery message lost) must be re-enqueued
    when force_recompute=True is passed.  This was the reported bug where the
    task appeared as 'enqueued' but was never actually processed."""
    repo = Repo()
    events = EventRepo()
    enqueue_count = 0

    def enqueue(task_id: int):
        nonlocal enqueue_count
        enqueue_count += 1
        return Result()

    task = create_task(
        payload=_payload(), settings=_settings(tmp_path),
        task_repo=repo, event_repo=events, enqueue_task=enqueue,
        idempotency_key="stuck-queued",
    )
    # Simulate task stuck in queued (Celery message lost — count stays 1).
    assert repo.tasks[task.id].status == TaskStatus.queued

    # Without force_recompute, still returns existing task and does NOT re-enqueue.
    not_rerun = create_task(
        payload=_payload(), settings=_settings(tmp_path),
        task_repo=repo, event_repo=events, enqueue_task=enqueue,
        idempotency_key="stuck-queued",
    )
    assert not_rerun.id == task.id
    assert enqueue_count == 1

    # With force_recompute, it MUST re-enqueue.
    rerun = create_task(
        payload=_payload(), settings=_settings(tmp_path),
        task_repo=repo, event_repo=events, enqueue_task=enqueue,
        idempotency_key="stuck-queued",
        force_recompute=True,
    )
    assert rerun.id == task.id
    assert enqueue_count == 2
    assert repo.tasks[task.id].status == TaskStatus.queued


def test_waiting_capacity_task_is_reenqueued_on_force_recompute(tmp_path):
    """Same fix applies to tasks stuck in waiting_capacity."""
    repo = Repo()
    events = EventRepo()
    enqueue_count = 0

    def enqueue(task_id: int):
        nonlocal enqueue_count
        enqueue_count += 1
        return Result()

    task = create_task(
        payload=_payload(), settings=_settings(tmp_path),
        task_repo=repo, event_repo=events, enqueue_task=enqueue,
        idempotency_key="stuck-capacity",
    )
    repo.update_status(task.id, TaskStatus.waiting_capacity)

    rerun = create_task(
        payload=_payload(), settings=_settings(tmp_path),
        task_repo=repo, event_repo=events, enqueue_task=enqueue,
        idempotency_key="stuck-capacity",
        force_recompute=True,
    )
    assert rerun.id == task.id
    assert enqueue_count == 2
    assert repo.tasks[task.id].status == TaskStatus.queued


def test_stuck_queued_task_old_celery_id_is_revoked_on_force_recompute(tmp_path):
    """The stale Celery message must be revoked to prevent a double-execution race."""
    repo = Repo()
    events = EventRepo()
    revoked = []

    def enqueue(task_id: int):
        return Result()

    def revoke(celery_id: str):
        revoked.append(celery_id)

    task = create_task(
        payload=_payload(), settings=_settings(tmp_path),
        task_repo=repo, event_repo=events, enqueue_task=enqueue,
        idempotency_key="revoke-test",
    )
    # Give it a fake stale Celery task ID.
    repo.tasks[task.id].celery_task_id = "old-celery-abc"

    create_task(
        payload=_payload(), settings=_settings(tmp_path),
        task_repo=repo, event_repo=events, enqueue_task=enqueue,
        idempotency_key="revoke-test",
        force_recompute=True,
        revoke_task=revoke,
    )

    assert revoked == ["old-celery-abc"]
