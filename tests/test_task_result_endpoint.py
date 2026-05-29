from pathlib import Path

from fastapi.testclient import TestClient

from service import app as app_module
from service.dependencies import get_app_settings, get_task_repo
from service.models import PipelineTask, TaskStatus


class StubTaskRepo:
    """In-memory TaskRepository stub keyed by external_id."""

    def __init__(self, tasks: dict[str, PipelineTask]):
        self._tasks = tasks

    def get_by_external_id(self, external_id: str):
        return self._tasks.get(external_id)

    def get_idempotency_key_by_external_id(self, external_id: str):
        return None


def _make_task(
    external_id: str,
    status: TaskStatus,
    result_path: str | None = None,
    error_text: str | None = None,
) -> PipelineTask:
    return PipelineTask(
        external_id=external_id,
        cadastral_data_path="in/a.json",
        pzz_zones_data_path="in/b.json",
        pzz_zone_vri_labels_path="in/c.json",
        vri_classifier_path="in/d.json",
        status=status,
        result_path=result_path,
        error_text=error_text,
    )


def _make_client(tmp_path: Path, tasks: dict[str, PipelineTask]) -> TestClient:
    """Wire the real app but override DB-backed dependencies with stubs.

    Overriding ``get_task_repo`` short-circuits the whole get_db/session_scope
    chain, so no real database is touched.
    """
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    fake_settings = type("Settings", (), {"outputs_dir": str(outputs_dir)})()

    app = app_module.app
    app.dependency_overrides[get_task_repo] = lambda: StubTaskRepo(tasks)
    app.dependency_overrides[get_app_settings] = lambda: fake_settings
    return TestClient(app)


def teardown_function() -> None:
    app_module.app.dependency_overrides.clear()


def test_task_result_not_found(tmp_path: Path):
    client = _make_client(tmp_path, tasks={})

    response = client.get("/tasks/missing/result")

    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found"


def test_task_result_not_ready(tmp_path: Path):
    task = _make_task("t-1", TaskStatus.running)
    client = _make_client(tmp_path, tasks={"t-1": task})

    response = client.get("/tasks/t-1/result")

    assert response.status_code == 409
    assert "not ready" in response.json()["detail"]


def test_task_result_failed(tmp_path: Path):
    task = _make_task("t-2", TaskStatus.failed, error_text="Pipeline exploded")
    client = _make_client(tmp_path, tasks={"t-2": task})

    response = client.get("/tasks/t-2/result")

    assert response.status_code == 422
    assert response.json()["detail"] == "Pipeline exploded"


def test_task_result_finished_ok(tmp_path: Path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    result_file = outputs_dir / "result.geojson"
    result_file.write_text('{"type":"FeatureCollection","features":[]}')
    task = _make_task("t-3", TaskStatus.finished, result_path=str(result_file))
    client = _make_client(tmp_path, tasks={"t-3": task})

    response = client.get("/tasks/t-3/result")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")
    assert "filename=\"t-3.geojson\"" in response.headers.get("content-disposition", "")


def test_task_result_missing_file(tmp_path: Path):
    missing_file = tmp_path / "outputs" / "missing.geojson"
    task = _make_task("t-4", TaskStatus.finished, result_path=str(missing_file))
    client = _make_client(tmp_path, tasks={"t-4": task})

    response = client.get("/tasks/t-4/result")

    assert response.status_code == 404
    assert response.json()["detail"] == "Task result file not found"


def test_cleanup_stale_output_files_removes_unreferenced(monkeypatch, tmp_path: Path):
    from service import tasks as tasks_module

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    stale_file = outputs_dir / "stale.geojson"
    stale_file.write_text("{}")

    tasks_module.settings.outputs_dir = str(outputs_dir)
    tasks_module.settings.outputs_cleanup_max_age_hours = 0

    class ScalarResult:
        def scalars(self):
            return self

        def all(self):
            return []

    class SessionForCleanup:
        def execute(self, *args, **kwargs):
            return ScalarResult()

    class StubSessionScope:
        def __enter__(self):
            return SessionForCleanup()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(tasks_module, "session_scope", lambda: StubSessionScope())

    tasks_module.cleanup_stale_output_files_task()

    assert not stale_file.exists()
