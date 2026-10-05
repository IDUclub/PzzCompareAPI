"""``GET /tasks/{id}/overlaps`` and the overlap lines of the object-zone-fit answer."""

import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pytest
from fastapi import HTTPException
from shapely.geometry import box

import service.api.tasks as tasks_api
from pipeline_modules.business.overlap_layer import (
    COL_MO,
    COL_OVERLAP_NOTES,
    OVERLAPS_KEY,
)
from pipeline_modules.business.pipeline_impl import (
    _attach_overlap_checks,
    _load_mo_boundaries,
    _write_overlap_report,
)


class _Task:
    building_type_col = None
    building_service_col = None

    def __init__(self, result_path: str | None, status: str = "finished"):
        self.result_path = result_path
        self.status = status
        self.external_id = "ext-overlaps"


REPORT = {
    "summary": {
        "parcel_overlaps": 1,
        "parcels_with_overlaps": 2,
        "parcels_in_multiple_zones": 0,
        "zone_overlaps": 1,
        "min_area_m2": 1.0,
        "min_share": 0.001,
    },
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": None,
            "properties": {"kind": "parcel_parcel", "feature_index_1": 0},
        }
    ],
}


def _result(tmp_path: Path, **members) -> str:
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": None,
                "properties": {
                    "Вердикт_ПЗЗ": "Разрешен",
                    "Код фактической зоны нахождения кадастра": "Ж-1",
                },
            }
        ],
        **members,
    }
    path = tmp_path / "result.geojson"
    path.write_text(json.dumps(collection, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _get_overlaps(monkeypatch, tmp_path: Path, task: _Task) -> dict:
    monkeypatch.setattr(tasks_api, "get_public_task_or_404", lambda *_: task)
    return tasks_api.get_overlaps_endpoint(
        "ext-overlaps",
        task_repo=None,
        app_settings=SimpleNamespace(outputs_dir=str(tmp_path)),
    )


def test_overlaps_endpoint_returns_report(monkeypatch, tmp_path: Path):
    task = _Task(_result(tmp_path, overlaps=REPORT))

    resp = _get_overlaps(monkeypatch, tmp_path, task)

    assert resp["summary"]["parcel_overlaps"] == 1
    assert resp["summary"]["zone_overlaps"] == 1
    assert resp["summary"]["min_area_m2"] == 1.0
    assert resp["overlaps"]["type"] == "FeatureCollection"
    assert resp["overlaps"]["features"] == REPORT["features"]
    assert "Наложения земельных участков друг на друга: 1" in resp["chat_message"]


def test_overlaps_endpoint_without_findings_says_so(monkeypatch, tmp_path: Path):
    clean = {**REPORT, "summary": {"min_area_m2": 1.0}, "features": []}
    task = _Task(_result(tmp_path, overlaps=clean))

    resp = _get_overlaps(monkeypatch, tmp_path, task)

    assert resp["summary"]["parcel_overlaps"] == 0
    assert resp["chat_message"] == (
        "Наложений земельных участков и территориальных зон не найдено."
    )


def test_old_result_without_report_asks_for_recompute(monkeypatch, tmp_path: Path):
    task = _Task(_result(tmp_path))

    with pytest.raises(HTTPException) as exc:
        _get_overlaps(monkeypatch, tmp_path, task)

    assert exc.value.status_code == 404
    assert "/tasks/ext-overlaps/recompute" in exc.value.detail


def test_unfinished_task_is_409(monkeypatch, tmp_path: Path):
    with pytest.raises(HTTPException) as exc:
        _get_overlaps(monkeypatch, tmp_path, _Task(None, status="running"))
    assert exc.value.status_code == 409


@pytest.mark.parametrize("group_by", ["zone", "object"])
def test_object_zone_fit_mentions_overlaps(tmp_path: Path, group_by: str):
    resp = tasks_api.build_object_zone_fit_response(
        _Task(_result(tmp_path, overlaps=REPORT)),
        "ext-overlaps",
        group_by,
        SimpleNamespace(outputs_dir=str(tmp_path)),
    )

    assert resp["summary"]["overlaps"] == {
        "parcel_overlaps": 1,
        "parcels_with_overlaps": 2,
        "parcels_in_multiple_zones": 0,
        "zone_overlaps": 1,
    }
    assert "Проверка наложений исходных слоёв:" in resp["chat_message"]
    assert "- Наложения территориальных зон друг на друга: 1." in resp["chat_message"]


def test_object_zone_fit_without_report_is_unchanged(tmp_path: Path):
    resp = tasks_api.build_object_zone_fit_response(
        _Task(_result(tmp_path)),
        "ext-overlaps",
        "object",
        SimpleNamespace(outputs_dir=str(tmp_path)),
    )

    assert "overlaps" not in resp["summary"]
    assert "наложений" not in resp["chat_message"]


def test_pipeline_attaches_columns_and_writes_report(tmp_path: Path):
    parcels = gpd.GeoDataFrame(
        {"geometry": [box(0.001, 0.001, 0.002, 0.002), box(0.0015, 0.001, 0.0025, 0.002)]},
        crs="EPSG:4326",
    )
    zones = gpd.GeoDataFrame(
        {"code": ["Ж-1"], "name": ["Жилая"], "geometry": [box(0, 0, 0.01, 0.01)]},
        crs="EPSG:4326",
    )
    classified, report = _attach_overlap_checks(
        parcels.copy(),
        source_gdf=parcels,
        pzz_zones_gdf=zones,
        pzz_zone_code_col="code",
        pzz_zone_name_col="name",
    )
    assert classified[COL_OVERLAP_NOTES].notna().all()
    assert report["summary"]["parcel_overlaps"] == 1

    path = tmp_path / "out.geojson"
    path.write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
    _write_overlap_report(path, report)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored[OVERLAPS_KEY]["summary"]["parcel_overlaps"] == 1
    assert stored[OVERLAPS_KEY]["features"][0]["properties"]["kind"] == "parcel_parcel"


def test_pipeline_skips_check_on_row_mismatch():
    parcels = gpd.GeoDataFrame({"geometry": [box(0, 0, 0.001, 0.001)]}, crs="EPSG:4326")
    zones = gpd.GeoDataFrame({"code": ["Ж-1"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")

    classified, report = _attach_overlap_checks(
        parcels.iloc[0:0],
        source_gdf=parcels,
        pzz_zones_gdf=zones,
        pzz_zone_code_col="code",
        pzz_zone_name_col="code",
    )

    assert report is None
    assert COL_OVERLAP_NOTES not in classified.columns


MO_SUMMARY = {
    "mo_checked": True,
    "mo_overlaps": 0,
    "parcels_in_multiple_mo": 2,
    "parcels_outside_mo": 1,
    "zones_in_multiple_mo": 0,
    "zones_outside_mo": 0,
}


def test_overlaps_endpoint_reports_mo_counts(monkeypatch, tmp_path: Path):
    report = {**REPORT, "summary": {**REPORT["summary"], **MO_SUMMARY}}
    task = _Task(_result(tmp_path, overlaps=report))

    resp = _get_overlaps(monkeypatch, tmp_path, task)

    assert resp["summary"]["parcels_in_multiple_mo"] == 2
    assert resp["summary"]["parcels_outside_mo"] == 1
    assert (
        "Участков, пересекающих границу муниципальных образований: 2."
        in resp["chat_message"]
    )


def test_overlaps_endpoint_without_mo_layer_has_no_mo_keys(monkeypatch, tmp_path: Path):
    resp = _get_overlaps(monkeypatch, tmp_path, _Task(_result(tmp_path, overlaps=REPORT)))

    assert "parcels_outside_mo" not in resp["summary"]
    assert "муниципальных" not in resp["chat_message"]


def test_clean_mo_check_says_mo_were_checked(monkeypatch, tmp_path: Path):
    clean = {**REPORT, "summary": {"mo_checked": True}, "features": []}
    task = _Task(_result(tmp_path, overlaps=clean))

    resp = _get_overlaps(monkeypatch, tmp_path, task)

    assert resp["summary"]["mo_overlaps"] == 0
    assert resp["chat_message"] == (
        "Наложений земельных участков, территориальных зон и границ "
        "муниципальных образований не найдено."
    )


def test_pipeline_checks_parcels_against_mo(tmp_path: Path):
    parcels = gpd.GeoDataFrame(
        {"geometry": [box(0.001, 0.001, 0.002, 0.002)]}, crs="EPSG:4326"
    )
    zones = gpd.GeoDataFrame(
        {"code": ["Ж-1"], "geometry": [box(0, 0, 0.01, 0.01)]}, crs="EPSG:4326"
    )
    mo_path = tmp_path / "mo.geojson"
    gpd.GeoDataFrame(
        {"name": ["Западный"], "geometry": [box(0, 0, 0.00175, 0.01)]},
        crs="EPSG:4326",
    ).to_file(mo_path, driver="GeoJSON")

    classified, report = _attach_overlap_checks(
        parcels.copy(),
        source_gdf=parcels,
        pzz_zones_gdf=zones,
        pzz_zone_code_col="code",
        pzz_zone_name_col="code",
        mo_gdf=_load_mo_boundaries(str(mo_path)),
    )

    assert report["summary"]["mo_checked"] is True
    assert report["summary"]["parcels_outside_mo"] == 1
    assert classified[COL_MO].iloc[0] == "МО «Западный» (75 %), вне МО (25 %)"


def test_unreadable_mo_layer_is_skipped(tmp_path: Path):
    broken = tmp_path / "mo.geojson"
    broken.write_text("not json", encoding="utf-8")

    assert _load_mo_boundaries("") is None
    assert _load_mo_boundaries(str(broken)) is None
