"""Large results: reports stream the result file, SSE inlines only small ones."""

import asyncio
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from service.api import tasks as tasks_module
from service.api.tasks import (
    _read_inline_result_geojson,
    build_object_zone_fit_response,
)
from service.infrastructure.geo_ingest import GeoIngestError, iter_geojson_features


class _Task:
    status = "finished"
    building_type_col = None
    building_service_col = None
    external_id = "ext-1"

    def __init__(self, result_path: str):
        self.result_path = result_path


def _feature(verdict: str, zone: str | None = "Ж-1", **extra) -> dict:
    props = {"Вердикт_ПЗЗ": verdict, **extra}
    if zone:
        props["Код фактической зоны нахождения кадастра"] = zone
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [142.0, 47.0]},
        "properties": props,
    }


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# --- iter_geojson_features -------------------------------------------------


def test_iter_features_collects_members_around_the_array(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "r.geojson",
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [_feature("Разрешен"), _feature("Не разрешен")],
                "zone_stats": {"zones_count": 4},
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    members: dict = {}
    features = list(iter_geojson_features(path, members))

    assert [f["properties"]["Вердикт_ПЗЗ"] for f in features] == [
        "Разрешен",
        "Не разрешен",
    ]
    assert members == {"type": "FeatureCollection", "zone_stats": {"zones_count": 4}}


def test_iter_features_handles_empty_collections(tmp_path: Path) -> None:
    members: dict = {}
    path = _write(tmp_path / "r.geojson", '{"type": "FeatureCollection", "features": []}')
    assert list(iter_geojson_features(path, members)) == []
    assert members == {"type": "FeatureCollection"}
    assert list(iter_geojson_features(_write(tmp_path / "e.json", "{}"), {})) == []


@pytest.mark.parametrize(
    "text",
    [
        '{"type": "FeatureCollection", "features": [',
        '{"features": [{"a": 1} {"b": 2}]}',
    ],
)
def test_iter_features_rejects_malformed_json(tmp_path: Path, text: str) -> None:
    with pytest.raises(GeoIngestError):
        list(iter_geojson_features(_write(tmp_path / "bad.geojson", text), {}))


# --- object-zone-fit report from the streamed file ---------------------------


def _report(tmp_path: Path, collection: dict) -> dict:
    path = _write(tmp_path / "result.geojson", json.dumps(collection, ensure_ascii=False))
    return build_object_zone_fit_response(
        _Task(str(path)), "ext-1", "object", SimpleNamespace(outputs_dir=str(tmp_path))
    )


def test_report_reads_zone_stats_written_after_features(tmp_path: Path) -> None:
    report = _report(
        tmp_path,
        {
            "type": "FeatureCollection",
            "features": [_feature("Разрешен"), _feature("Нет пересечения с ПЗЗ", None)],
            "zone_stats": {"zones_count": 7},
        },
    )
    assert report["summary"]["total"] == 2
    assert report["summary"]["in_correct_zone"] == 1
    assert report["summary"]["not_in_zone"] == 1
    assert report["summary"]["zone_polygons_count"] == 7
    assert all("category" not in row for row in report["objects"])


def test_report_detects_building_results_by_feature_marker(tmp_path: Path) -> None:
    report = _report(
        tmp_path,
        {
            "type": "FeatureCollection",
            "features": [
                _feature("Разрешен"),
                _feature("Разрешен", **{"Категория_объекта": "Здание"}),
            ],
        },
    )
    assert report["mode"] == "building_pzz_check"
    assert all("category" in row for row in report["objects"])


def test_report_on_malformed_result_is_503(tmp_path: Path) -> None:
    path = _write(tmp_path / "result.geojson", '{"features": [')
    with pytest.raises(HTTPException) as exc:
        build_object_zone_fit_response(
            _Task(str(path)), "ext-1", "object", SimpleNamespace(outputs_dir=str(tmp_path))
        )
    assert exc.value.status_code == 503


def test_report_on_missing_result_is_404(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as exc:
        build_object_zone_fit_response(
            _Task(str(tmp_path / "missing.geojson")),
            "ext-1",
            "object",
            SimpleNamespace(outputs_dir=str(tmp_path)),
        )
    assert exc.value.status_code == 404


# --- SSE inline geojson ------------------------------------------------------


def _settings(tmp_path: Path, max_bytes: int) -> SimpleNamespace:
    return SimpleNamespace(
        outputs_dir=str(tmp_path), sse_inline_geojson_max_bytes=max_bytes
    )


def test_small_result_is_inlined_as_one_valid_line(tmp_path: Path) -> None:
    collection = {
        "type": "FeatureCollection",
        "features": [_feature("Разрешен", reason="a\nb")],
    }
    path = _write(
        tmp_path / "r.geojson",
        json.dumps(collection, ensure_ascii=False, indent=2).replace("\n", "\r\n"),
    )
    text = _read_inline_result_geojson(str(path), _settings(tmp_path, 10_000))

    assert text is not None
    assert "\n" not in text and "\r" not in text
    assert json.loads(text) == collection


def test_large_result_is_not_inlined(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "r.geojson",
        json.dumps({"type": "FeatureCollection", "features": [_feature("Разрешен")]}),
    )
    assert _read_inline_result_geojson(str(path), _settings(tmp_path, 10)) is None


def _stream_events(monkeypatch, tmp_path: Path, max_bytes: int) -> list[str]:
    """Drive ``task_stream_with_report_generator`` over a finished task."""
    from service.models import TaskStatus

    path = _write(
        tmp_path / "r.geojson",
        json.dumps({"type": "FeatureCollection", "features": [_feature("Разрешен")]}),
    )
    task = SimpleNamespace(id=1, status=TaskStatus.finished, result_path=str(path))

    class _Result:
        def scalar_one_or_none(self):
            return task

        def scalars(self):
            return SimpleNamespace(all=lambda: [])

    @contextmanager
    def _session_scope():
        yield SimpleNamespace(execute=lambda stmt: _Result())

    monkeypatch.setattr(tasks_module, "session_scope", _session_scope)
    monkeypatch.setattr(
        tasks_module,
        "TaskOut",
        SimpleNamespace(
            model_validate=lambda t: SimpleNamespace(model_dump=lambda mode: {})
        ),
    )
    monkeypatch.setattr(
        tasks_module,
        "build_result_geo_layers",
        lambda *a, **k: [{"name": "classified_result", "role": "result"}],
    )
    request = SimpleNamespace(is_disconnected=lambda: asyncio.sleep(0, result=False))

    async def run():
        return [
            sse.event
            async for sse in tasks_module.task_stream_with_report_generator(
                "ext-1",
                group_by="object",
                poll_interval=0.01,
                request=request,
                app_settings=_settings(tmp_path, max_bytes),
                initial={"external_id": "ext-1"},
                include_report=False,
            )
        ]

    return asyncio.run(run())


def test_stream_inlines_small_result(monkeypatch, tmp_path: Path) -> None:
    events = _stream_events(monkeypatch, tmp_path, max_bytes=10_000)
    assert events.index("geojson") < events.index("file") < events.index("done")


def test_stream_sends_only_the_link_for_large_result(monkeypatch, tmp_path: Path) -> None:
    events = _stream_events(monkeypatch, tmp_path, max_bytes=10)
    assert "geojson" not in events
    assert "error" not in events
    assert "file" in events and events[-1] == "done"
