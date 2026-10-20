from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from service.api import tasks
from service.application.use_cases.service_reports import descriptor, report_text


@pytest.mark.parametrize("kind", ["pzz", "classify", "placement", "scenario"])
def test_report_endpoint_opens_and_downloads_full_results(monkeypatch, kind):
    task = SimpleNamespace(
        status="finished",
        placement_vri_code="3.1" if kind == "placement" else None,
        include_pzz_check=kind != "classify",
    )
    monkeypatch.setattr(tasks, "get_task_or_404", lambda *args: task)
    monkeypatch.setattr(tasks, "_is_scenario_task", lambda *args: kind == "scenario")
    calls = []
    requirement = "Полный текст требования " * 100

    def build(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "chat_message": "Краткий ответ.",
            "summary": {"total": 150},
            "objects": [
                {"feature_index": index, "reason": f"Объект {index}"}
                for index in range(150)
            ],
            "norms": [
                {
                    "requirement": requirement,
                    "note": "Нет данных для автоматической проверки",
                }
            ],
        }

    selected = {
        "classify": "build_classify_summary_response",
        "placement": "build_placement_summary_response",
    }.get(kind, "build_object_zone_fit_response")
    monkeypatch.setattr(tasks, selected, build)
    app = FastAPI()
    app.include_router(tasks.router)
    app.dependency_overrides[tasks.get_task_repo] = lambda: object()
    app.dependency_overrides[tasks.get_app_settings] = lambda: SimpleNamespace(
        app_name="pzz"
    )
    report = descriptor("pzz", "", "task-id", "Отчёт")
    client = TestClient(app)
    opened = client.get(report["url"])
    downloaded = client.get(report["download_url"])
    assert opened.status_code == downloaded.status_code == 200
    assert opened.content == downloaded.content
    assert "Краткий ответ." in opened.text
    assert "Объект 149" in opened.text
    assert requirement in opened.text
    assert "Нет данных для автоматической проверки" in opened.text
    assert opened.headers["content-type"] == "text/plain; charset=utf-8"
    assert opened.headers["content-disposition"].startswith("inline;")
    assert downloaded.headers["content-disposition"].startswith("attachment;")
    if kind in {"pzz", "scenario"}:
        assert calls[0][1]["scenario"] == (kind == "scenario")


def test_report_does_not_claim_an_unfinished_task_passed(monkeypatch):
    task = SimpleNamespace(
        status="running", placement_vri_code=None, include_pzz_check=True
    )
    monkeypatch.setattr(tasks, "get_task_or_404", lambda *args: task)
    monkeypatch.setattr(tasks, "_is_scenario_task", lambda *args: False)
    app = FastAPI()
    app.include_router(tasks.router)
    app.dependency_overrides[tasks.get_task_repo] = lambda: object()
    app.dependency_overrides[tasks.get_app_settings] = lambda: SimpleNamespace()
    assert TestClient(app).get("/reports/task-id").status_code == 409


def test_report_without_details_contains_the_entire_answer():
    assert report_text("Отчёт", "Полный ответ") == "Отчёт\n\nПолный ответ\n"
