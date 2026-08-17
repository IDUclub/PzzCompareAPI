"""The building PZZ check has to be submittable outside the SSE chat flow.

Its zone review can end without a task, so both the streaming and the plain
endpoint carry that outcome — these tests pin the two presentations to the same
prepared run.
"""

import asyncio
from datetime import datetime, timezone

import pytest

from service.api.classifier import (
    BuildingPzzRun,
    _building_run_response,
    _run_building_pzz_auto,
)
from service.models import TaskStatus
from service.schemas import TaskOut

SUGGESTIONS = [
    {
        "user_code": "СХ-3",
        "user_name": "сельхоз",
        "suggested_code": "АГ-1",
        "suggested_name": "Сельхоз",
    }
]


def _task() -> TaskOut:
    return TaskOut(
        id=1,
        external_id="a1b2",
        cadastral_data_path="buildings.geojson",
        pzz_zones_data_path="zones.geojson",
        priority=1,
        status=TaskStatus.queued,
        include_pzz_check=True,
        cadastral_vri_col="",
        pzz_zone_code_col="Индекс_зоны",
        pzz_zone_name_col="Код_объекта",
        result_path=None,
        error_text=None,
        celery_task_id=None,
        created_at=datetime.now(timezone.utc),
        started_at=None,
        finished_at=None,
    )


def test_created_run_carries_the_task():
    out = _building_run_response(
        BuildingPzzRun(action="created", narrative="колонки найдены", task=_task())
    )

    assert out.task is not None and out.task.external_id == "a1b2"
    assert "poll" in out.next_step


def test_the_response_carries_our_own_wording_of_the_outcome():
    """A chat client shows this instead of assembling text from action and suggestions."""
    out = _building_run_response(
        BuildingPzzRun(
            action="suggest_upload",
            narrative="колонки найдены",
            detail="d",
            uncovered_codes=("СХ-3", "Т-1"),
        )
    )

    assert "СХ-3, Т-1" in out.chat_message
    assert "pzz_descriptions_file" in out.chat_message


def test_a_started_check_needs_no_review_wording():
    out = _building_run_response(
        BuildingPzzRun(action="created", narrative="n", task=_task())
    )

    assert out.chat_message == ""


def test_confirm_run_carries_suggestions_and_no_task():
    out = _building_run_response(
        BuildingPzzRun(
            action="confirm",
            narrative="колонки найдены",
            detail="подтвердите соответствия",
            suggestions=SUGGESTIONS,
        )
    )

    assert out.task is None
    assert out.suggestions is not None
    assert out.suggestions[0].suggested_code == "АГ-1"
    assert "confirmed_zone_map" in out.next_step


@pytest.mark.parametrize(
    "action, expected",
    [
        ("suggest_upload", "pzz_descriptions_upload_id"),
        ("detection_failed", "resubmit"),
    ],
)
def test_blocked_runs_say_what_the_user_must_provide(action, expected):
    """A response without a task is a request to the user, not a retryable failure."""
    out = _building_run_response(BuildingPzzRun(action=action, narrative="n"))

    assert out.task is None
    assert expected in out.next_step


def _sse_events(response) -> list[str]:
    async def collect() -> list[str]:
        return [event.event async for event in response.body_iterator]

    return asyncio.run(collect())


def _stream(monkeypatch, run: BuildingPzzRun):
    async def fake_prepare(**_kwargs) -> BuildingPzzRun:
        return run

    monkeypatch.setattr(
        "service.api.classifier._prepare_building_pzz_run", fake_prepare
    )
    return asyncio.run(
        _run_building_pzz_auto(
            request=None,
            buildings_file=None,
            zones_file=None,
            descriptions_file=None,
            confirmed_zone_map=None,
            user_query=None,
            chat_id=None,
            group_by="zone",
            model=None,
            temperature=None,
            priority=1,
            force_recompute=False,
            poll_interval=2.0,
            idempotency_key=None,
            user_id="user-1",
            app_settings=None,
            task_repo=None,
            event_repo=None,
            session=None,
        )
    )


def test_stream_reports_a_blocked_review_as_zone_review(monkeypatch):
    response = _stream(
        monkeypatch,
        BuildingPzzRun(
            action="confirm", narrative="n", detail="d", suggestions=SUGGESTIONS
        ),
    )

    events = _sse_events(response)
    assert "zone_review" in events
    assert events[-1] == "done"


def test_stream_reports_undetected_columns_as_error(monkeypatch):
    response = _stream(
        monkeypatch,
        BuildingPzzRun(action="detection_failed", narrative="n", detail="d"),
    )

    events = _sse_events(response)
    assert "error" in events
    assert "zone_review" not in events
    assert events[-1] == "done"
