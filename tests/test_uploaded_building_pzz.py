"""Tests for the deterministic uploaded-building PZZ runner."""
from __future__ import annotations

import json
from types import SimpleNamespace

from service.domain import PipelineRequest
from service.infrastructure.runners._deterministic_pzz import (
    COL_MATCHED_VRI_CODE,
    COL_RESOLUTION_BASIS,
    COL_VERDICT,
    COL_ZONE_CODE,
)
from service.infrastructure.runners.uploaded_building_pzz_runner import (
    UploadedBuildingPzzRunner,
)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        physical_object_type_to_vri_path="data/physical_object_type_to_vri.json",
        service_type_to_vri_path="data/service_type_to_vri.json",
        default_fz_to_pzz_mapping_path="data/functional_zones_to_pzz_mapping.json",
    )


def _runner() -> UploadedBuildingPzzRunner:
    return UploadedBuildingPzzRunner(_settings())


# --- pure VRI resolution (no geo) ----------------------------------------------

def test_resolve_residential_by_floor_band() -> None:
    r = _runner()
    # type 4 (жилой дом), 3 floors -> low-rise band 2.1.1
    assert r._resolve_vri(4, True, None, 3)[0] == "2.1.1"
    # 10 floors -> high-rise band 2.6
    assert r._resolve_vri(4, True, None, 10)[0] == "2.6"


def test_resolve_residential_by_text_type() -> None:
    r = _runner()
    # textual "жилое" with no numeric po_type still routes through floor bands
    assert r._resolve_vri(None, True, None, 2)[0] == "2.1.1"


def test_resolve_service_type() -> None:
    r = _runner()
    # service_type_id 22 == school -> 3.5.1 (from service_type_to_vri.json)
    assert r._resolve_vri(None, False, 22, None)[0] == "3.5.1"


def test_extract_resolves_text_service_name() -> None:
    r = _runner()
    req = _req(building_type_col="", building_service_col="service_name")
    po, is_res, svc, floors, label = r._extract({"service_name": "Школа"}, req)
    assert (po, is_res, svc, floors) == (None, False, 22, None)
    assert label == "Школа"
    assert r._resolve_vri(po, is_res, svc, floors)[0] == "3.5.1"


def test_extract_resolves_text_service_alias_prefix() -> None:
    r = _runner()
    req = _req(building_type_col="", building_service_col="service_name")
    po, is_res, svc, floors, label = r._extract(
        {"service_name": "Детский сад 12"}, req
    )
    assert (po, is_res, svc, floors) == (None, False, 21, None)
    assert label == "Детский сад 12"
    assert r._resolve_vri(po, is_res, svc, floors)[0] == "3.5.1"


def test_extract_resolves_text_physical_object_type_name() -> None:
    r = _runner()
    req = _req(building_type_col="type_name", building_service_col="")
    po, is_res, svc, floors, label = r._extract({"type_name": "Склад"}, req)
    assert (po, is_res, svc, floors) == (29, False, None, None)
    assert label == "Склад"
    assert r._resolve_vri(po, is_res, svc, floors)[0] == "6.9"


def test_resolve_residential_wins_over_service() -> None:
    r = _runner()
    # a residential building carrying a stray service id still classifies as жилое
    assert r._resolve_vri(4, True, 22, 3)[0] == "2.1.1"


def test_resolve_unknown_returns_none() -> None:
    r = _runner()
    assert r._resolve_vri(999999, False, None, None) == (None, None, "")


def test_extract_reads_configured_columns() -> None:
    r = _runner()
    req = _req(building_type_col="po", building_service_col="svc", building_floors_col="fl")
    po, is_res, svc, floors, label = r._extract({"po": 4, "svc": 22, "fl": 5}, req)
    assert (po, is_res, svc, floors) == (4, True, 22, 5)
    assert "4" in label and "22" in label


# --- end-to-end run() with a real spatial join ---------------------------------

def _square(x0: float, y0: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x0 + 1, y0], [x0 + 1, y0 + 1], [x0, y0 + 1], [x0, y0]]],
    }


def _point(x: float, y: float) -> dict:
    return {"type": "Point", "coordinates": [x, y]}


def _zones() -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": _square(0, 0), "properties": {"zone_type": 8}},
            {"type": "Feature", "geometry": _square(2, 0), "properties": {"zone_type": 6}},
        ],
    }


def _buildings() -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            # residential, 3 floors, inside zone 8 (allows 2.1.1) -> allowed
            {"type": "Feature", "geometry": _point(0.5, 0.5),
             "properties": {"po": 4, "floors": 3}},
            # residential, 3 floors, inside zone 6 (warehouses, no 2.x) -> not allowed
            {"type": "Feature", "geometry": _point(2.5, 0.5),
             "properties": {"po": 4, "floors": 3}},
        ],
    }


def _req(**overrides) -> PipelineRequest:
    base = dict(
        task_external_id="bld-test",
        cadastral_data_path="",
        pzz_zones_data_path="",
        pzz_zone_vri_labels_path="",
        vri_classifier_path="",
        include_pzz_check=True,
        cadastral_vri_col="",
        pzz_zone_code_col="zone_type",
        pzz_zone_name_col="",
        outputs_dir="",
        is_building_upload=True,
        building_type_col="po",
        building_service_col="",
        building_floors_col="floors",
    )
    base.update(overrides)
    return PipelineRequest(**base)


def _run(tmp_path, request_overrides=None, descriptions=None) -> list[dict]:
    (tmp_path / "b.geojson").write_text(json.dumps(_buildings()), encoding="utf-8")
    (tmp_path / "z.geojson").write_text(json.dumps(_zones()), encoding="utf-8")
    labels_path = ""
    if descriptions is not None:
        p = tmp_path / "desc.json"
        p.write_text(json.dumps(descriptions), encoding="utf-8")
        labels_path = str(p)
    req = _req(
        cadastral_data_path=str(tmp_path / "b.geojson"),
        pzz_zones_data_path=str(tmp_path / "z.geojson"),
        pzz_zone_vri_labels_path=labels_path,
        outputs_dir=str(tmp_path / "out"),
        **(request_overrides or {}),
    )
    out_path = _runner().run(req)
    with open(out_path, encoding="utf-8") as fh:
        return json.load(fh)["features"]


def test_run_residential_verdicts_fallback_mapping(tmp_path) -> None:
    feats = _run(tmp_path)
    props = [f["properties"] for f in feats]
    # feature 0 in zone 8 -> allowed; feature 1 in zone 6 -> not allowed
    assert props[0][COL_VERDICT] == "Разрешен"
    assert props[0][COL_MATCHED_VRI_CODE] == "2.1.1"
    assert props[0][COL_ZONE_CODE] == "8"
    assert props[1][COL_VERDICT] == "Не разрешен"
    # clean whitelist: exactly the 8 result columns, no input passthrough
    assert set(props[0].keys()) == {
        "ВРИ_ЕГРН", COL_ZONE_CODE, "Название фактической зоны нахождения кадастра",
        COL_VERDICT, "Причина", COL_MATCHED_VRI_CODE, "Подобранный_ВРИ",
        COL_RESOLUTION_BASIS,
    }
    # residential building -> basis records the floor-band resolution
    assert "по этажности" in props[0][COL_RESOLUTION_BASIS]


def test_run_uploaded_descriptions_override(tmp_path) -> None:
    # descriptions file that permits only warehouses (6.9) in zone 8 -> the
    # residential building there is no longer allowed, proving the override.
    descriptions = {
        "functional_zone_mappings": [
            {
                "functional_zone_type_id": 8,
                "db_zone_nickname": "Тестовая зона",
                "averaged_pzz_profile": {"main_vri": [{"vri_code": "6.9"}]},
            }
        ]
    }
    feats = _run(tmp_path, descriptions=descriptions)
    assert feats[0]["properties"][COL_VERDICT] == "Не разрешен"
