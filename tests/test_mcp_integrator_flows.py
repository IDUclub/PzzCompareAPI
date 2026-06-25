import asyncio

from service.mcp_server.tools.scenarios import get_scenario_classification_report
from service.mcp_server.tools.tasks import get_task_result


def _run(coro):
    return asyncio.run(coro)


class FakeTaskApi:
    def __init__(self, task, result=None):
        self.task = task
        self.result = result or {"type": "FeatureCollection", "features": []}
        self.result_calls = 0

    async def get_task(self, external_id):
        assert external_id == self.task["external_id"]
        return self.task

    async def get_task_result(self, external_id):
        assert external_id == self.task["external_id"]
        self.result_calls += 1
        return self.result


class FakeScenarioApi:
    def __init__(self, task=None, report=None):
        self.task = task
        self.report = report or {
            "task_external_id": "sc-task",
            "summary": {"total": 0, "in_correct_zone": 0, "in_wrong_zone": 0, "unclear": 0},
            "chat_message": "ok",
            "zones": [],
        }
        self.report_calls = 0

    async def get_scenario_task(self, *, scenario_id, external_id, token=None):
        assert scenario_id == 42
        assert external_id == self.task["external_id"]
        return self.task

    async def get_scenario_object_zone_fit(
        self, *, scenario_id, external_id, group_by="zone", token=None
    ):
        assert scenario_id == 42
        assert external_id == self.task["external_id"]
        self.report_calls += 1
        return self.report


def test_mcp_get_task_result_returns_actionable_status_when_running():
    api = FakeTaskApi({"external_id": "task-1", "status": "running", "error_text": None})

    result = _run(get_task_result("task-1", api=api))

    assert result["ready"] is False
    assert result["status"] == "running"
    assert "get_task_status" in result["next_step"]
    assert api.result_calls == 0


def test_mcp_get_task_result_returns_actionable_status_when_failed():
    api = FakeTaskApi(
        {"external_id": "task-2", "status": "failed", "error_text": "Pipeline exploded"}
    )

    result = _run(get_task_result("task-2", api=api))

    assert result["ready"] is False
    assert result["status"] == "failed"
    assert result["error_text"] == "Pipeline exploded"
    assert "recompute_task" in result["next_step"]
    assert api.result_calls == 0


def test_mcp_get_task_result_downloads_geojson_when_finished():
    geojson = {"type": "FeatureCollection", "features": [{"type": "Feature"}]}
    api = FakeTaskApi({"external_id": "task-3", "status": "finished"}, result=geojson)

    result = _run(get_task_result("task-3", api=api))

    assert result == geojson
    assert api.result_calls == 1


def test_mcp_scenario_report_returns_actionable_status_when_not_finished():
    api = FakeScenarioApi({"external_id": "sc-task", "status": "queued", "error_text": None})

    result = _run(get_scenario_classification_report(42, "sc-task", api=api))

    assert result["ready"] is False
    assert result["status"] == "queued"
    assert "get_scenario_classification_status" in result["next_step"]
    assert api.report_calls == 0


def test_mcp_scenario_report_returns_actionable_status_when_failed():
    api = FakeScenarioApi(
        {"external_id": "sc-task", "status": "failed", "error_text": "No zones"}
    )

    result = _run(get_scenario_classification_report(42, "sc-task", api=api))

    assert result["ready"] is False
    assert result["status"] == "failed"
    assert result["error_text"] == "No zones"
    assert "recompute_scenario_classification" in result["next_step"]
    assert api.report_calls == 0


def test_mcp_scenario_report_returns_report_when_finished():
    api = FakeScenarioApi({"external_id": "sc-task", "status": "finished"})

    result = _run(get_scenario_classification_report(42, "sc-task", api=api))

    assert result["task_external_id"] == "sc-task"
    assert result["chat_message"] == "ok"
    assert api.report_calls == 1
