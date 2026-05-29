from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from service import app as app_module
from service.dependencies import get_task_repo
from service.models import PipelineTask, TaskStatus


class StubTaskRepo:
    """In-memory TaskRepository stub backing the /tasks_list endpoint."""

    def __init__(self, tasks: list[PipelineTask]):
        self._tasks = tasks

    def list_tasks(self, *, status=None, limit=20, offset=0):
        tasks = list(self._tasks)
        if status is not None:
            tasks = [task for task in tasks if task.status == status]
        tasks.sort(key=lambda task: task.created_at, reverse=True)
        return tasks[offset : offset + limit], len(tasks)


_id_counter = 0


def _make_task(external_id: str, status: TaskStatus, created_at: datetime) -> PipelineTask:
    global _id_counter
    _id_counter += 1
    return PipelineTask(
        id=_id_counter,
        external_id=external_id,
        include_pzz_check=True,
        cadastral_vri_col="vri",
        pzz_zone_code_col="code",
        pzz_zone_name_col="name",
        cadastral_data_path="in/a.json",
        pzz_zones_data_path="in/b.json",
        pzz_zone_vri_labels_path="in/c.json",
        vri_classifier_path="in/d.json",
        priority=1,
        status=status,
        created_at=created_at,
    )


def _make_client(tasks: list[PipelineTask]) -> TestClient:
    app = app_module.app
    app.dependency_overrides[get_task_repo] = lambda: StubTaskRepo(tasks)
    return TestClient(app)


def teardown_function() -> None:
    app_module.app.dependency_overrides.clear()


def test_list_tasks_with_status_filter():
    now = datetime.utcnow()
    tasks = [
        _make_task("t-1", TaskStatus.queued, now - timedelta(minutes=2)),
        _make_task("t-2", TaskStatus.running, now - timedelta(minutes=1)),
        _make_task("t-3", TaskStatus.queued, now),
    ]
    client = _make_client(tasks)

    response = client.get("/tasks_list", params={"status": "queued", "limit": 10, "offset": 0})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert [item["external_id"] for item in payload["items"]] == ["t-3", "t-1"]


def test_list_tasks_invalid_status():
    client = _make_client([])

    response = client.get("/tasks_list", params={"status": "unknown"})

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid status filter"
