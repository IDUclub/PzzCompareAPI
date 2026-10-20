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
    assert "filename*=UTF-8''%D0%9E" in downloaded.headers["content-disposition"]
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
    assert (
        report_text("Отчёт", "Полный ответ")
        == "# Отчёт\n\n## Результат работы\n\nПолный ответ\n"
    )


def test_report_has_russian_fields_zones_statuses_and_sections():
    import re

    details = {
        "task_external_id": "hidden-task-id",
        "mode": "pzz_check",
        "group_by": "object",
        "summary": {
            "in_correct_zone": 1,
            "in_wrong_zone": 1,
            "unclear": 1,
            "buildings_by_zone": {"residential": 2, "business": 1},
            "targets_source": "service_defaults",
            "regulations": {"source": "normgraph"},
            "classes": {"A*": {"zones": 1, "ha": 12.5}},
            "norms": [
                {
                    "document": "СП 1",
                    "clause": "2.1",
                    "strictness": "strict",
                    "status": "not_applied",
                    "requirement": "Проверить расстояние.",
                    "note": "Данные отсутствуют",
                }
            ],
        },
        "objects": [
            {"feature_index": 0, "fit": "correct", "vri_text": "Жилой дом"},
            {
                "feature_index": 1,
                "fit": "wrong",
                "matched_vri_name": "Производство",
                "reason": "Недопустимый вид использования",
            },
            {
                "feature_index": 2,
                "fit": "unclear",
                "verdict": "Нет описания зоны",
                "is_in_correct_zone": False,
            },
        ],
        "parameters": {"density_scenario": "mean", "default_floor_group": "high"},
        "warnings": [],
    }
    text = report_text("Отчёт", "Проверка завершена.", details)
    assert text.startswith("# Отчёт\n")
    assert "## Сводка" in text
    assert "## Не соответствуют ПЗЗ" in text
    assert "## Соответствуют ПЗЗ" in text
    assert "## Требуют ручной проверки" in text
    assert "| Соответствуют ПЗЗ | 1 |" in text
    assert "Жилая зона" in text
    assert "Жилой дом" in text and "Производство" in text
    assert "Нет описания зоны" in text
    assert "не применена" in text and "Данные отсутствуют" in text
    assert "12,5" in text
    assert not re.search(r"[A-Za-z]", text)


def test_report_escapes_table_cells_and_keeps_unknown_values(caplog):
    text = report_text(
        "Отчёт",
        "Ответ",
        {"summary": {"future_metric": 987, "name": "Зона | номер\nодин"}},
    )
    assert "Дополнительный показатель" in text and "987" in text
    assert "future_metric" not in text
    assert "future_metric" in caplog.text
    assert r"Зона \| номер; один" in text


def test_classification_does_not_claim_pzz_compliance():
    text = report_text(
        "Отчёт",
        "Классификация завершена",
        {
            "objects": [
                {"feature_index": 0, "fit": "matched", "matched_vri": "Жилая застройка"}
            ]
        },
    )
    assert "## Подобраны виды использования" in text
    assert "## Соответствуют ПЗЗ" not in text


def test_localization_preserves_numbers_and_source_codes():
    text = report_text(
        "Отчёт",
        "Ответ",
        {
            "summary": {
                "living_area_total": 1234567.89,
                "zone_type_id": "A",
                "vri_code": "6.0",
                "document": "ГОСТ A-1",
                "clause": "B",
                "feature_index": 2,
            }
        },
    )
    assert "1234567,89" in text
    assert "| Код зоны | A |" in text
    assert "| Код ВРИ | 6.0 |" in text
    assert "ГОСТ A-1" in text and "| Пункт | B |" in text
    assert "| Индекс объекта в исходных данных | 2 |" in text
