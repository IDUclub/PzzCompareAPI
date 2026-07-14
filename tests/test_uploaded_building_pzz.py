"""Tests for the deterministic uploaded-building PZZ runner."""
from __future__ import annotations

import json
from types import SimpleNamespace

from service.domain import PipelineRequest
from service.infrastructure.runners._deterministic_pzz import (
    CATEGORY_BUILDING,
    CATEGORY_SERVICE,
    COL_CATEGORY,
    COL_MATCHED_VRI_CODE,
    COL_RESOLUTION_BASIS,
    COL_VERDICT,
    COL_ZONE_CODE,
    COL_ZONE_NAME,
)
from service.infrastructure.runners.uploaded_building_pzz_runner import (
    UploadedBuildingPzzRunner,
)


def _settings(llm_fallback: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        physical_object_type_to_vri_path="data/physical_object_type_to_vri.json",
        service_type_to_vri_path="data/service_type_to_vri.json",
        default_fz_to_pzz_mapping_path="data/functional_zones_to_pzz_mapping.json",
        default_pzz_zone_labels_path="data/pzz_zone_llm_labels_template.json",
        building_llm_name_fallback=llm_fallback,
        building_pzz_zone_suggest_threshold=5,
        ollama_base_url="http://ollama.invalid",
        chat_model="test-model",
        generate_model="test-model",
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
    po, is_res, svc, floors, label, _llm = r._extract({"service_name": "Школа"}, req)
    assert (po, is_res, svc, floors) == (None, False, 22, None)
    assert label == "Школа"
    assert r._resolve_vri(po, is_res, svc, floors)[0] == "3.5.1"


def test_extract_resolves_text_service_alias_prefix() -> None:
    r = _runner()
    req = _req(building_type_col="", building_service_col="service_name")
    po, is_res, svc, floors, label, _llm = r._extract(
        {"service_name": "Детский сад 12"}, req
    )
    assert (po, is_res, svc, floors) == (None, False, 21, None)
    assert label == "Детский сад 12"
    assert r._resolve_vri(po, is_res, svc, floors)[0] == "3.5.1"


def test_extract_resolves_text_physical_object_type_name() -> None:
    r = _runner()
    req = _req(building_type_col="type_name", building_service_col="")
    po, is_res, svc, floors, label, _llm = r._extract({"type_name": "Склад"}, req)
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
    po, is_res, svc, floors, label, _llm = r._extract({"po": 4, "svc": 22, "fl": 5}, req)
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
    # clean whitelist: exactly the 8 result columns + the building-mode category
    assert set(props[0].keys()) == {
        "ВРИ_ЕГРН", COL_ZONE_CODE, "Название фактической зоны нахождения кадастра",
        COL_VERDICT, "Причина", COL_MATCHED_VRI_CODE, "Подобранный_ВРИ",
        COL_RESOLUTION_BASIS, COL_CATEGORY,
    }
    # residential building -> basis records the floor-band resolution
    assert "по этажности" in props[0][COL_RESOLUTION_BASIS]
    # non-service physical objects are categorised as «Здание»
    assert props[0][COL_CATEGORY] == CATEGORY_BUILDING


def test_run_tags_category_for_split(tmp_path) -> None:
    """A mixed layer (physical object + service) is tagged so the API can split
    the result into «здания» / «сервисы» download layers."""
    buildings = {
        "type": "FeatureCollection",
        "features": [
            # physical object (residential building) -> «Здание»
            {"type": "Feature", "geometry": _point(0.5, 0.5),
             "properties": {"po": 4, "floors": 3}},
            # service (school, service_type_id 22) in the same zone -> «Сервис»
            {"type": "Feature", "geometry": _point(0.5, 0.5),
             "properties": {"svc": 22}},
        ],
    }
    (tmp_path / "b.geojson").write_text(json.dumps(buildings), encoding="utf-8")
    (tmp_path / "z.geojson").write_text(json.dumps(_zones()), encoding="utf-8")
    req = _req(
        cadastral_data_path=str(tmp_path / "b.geojson"),
        pzz_zones_data_path=str(tmp_path / "z.geojson"),
        outputs_dir=str(tmp_path / "out"),
        building_type_col="po",
        building_service_col="svc",
        building_floors_col="floors",
    )
    feats = json.load(open(_runner().run(req), encoding="utf-8"))["features"]
    cats = [f["properties"][COL_CATEGORY] for f in feats]
    assert cats == [CATEGORY_BUILDING, CATEGORY_SERVICE]

    # serve-time split (mirrors _serve_result_split in service/api/tasks.py)
    def by_cat(cat):
        return [f for f in feats if f["properties"].get(COL_CATEGORY) == cat]
    assert len(by_cat(CATEGORY_BUILDING)) == 1
    assert len(by_cat(CATEGORY_SERVICE)) == 1
    # every feature lands in exactly one layer
    assert len(by_cat(CATEGORY_BUILDING)) + len(by_cat(CATEGORY_SERVICE)) == len(feats)


def _run_buildings(runner, tmp_path, buildings, **overrides) -> list[dict]:
    (tmp_path / "b.geojson").write_text(json.dumps(buildings), encoding="utf-8")
    (tmp_path / "z.geojson").write_text(json.dumps(_zones()), encoding="utf-8")
    req = _req(
        cadastral_data_path=str(tmp_path / "b.geojson"),
        pzz_zones_data_path=str(tmp_path / "z.geojson"),
        outputs_dir=str(tmp_path / "out"),
        **overrides,
    )
    return json.load(open(runner.run(req), encoding="utf-8"))["features"]


def _unknown_service_layer() -> dict:
    # a service name that matches neither an id nor a catalogue alias
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": _point(0.5, 0.5),
             "properties": {"svc": "Учебный центр «Ромашка»"}},
        ],
    }


def test_needs_llm_only_for_unresolved_text_names() -> None:
    r = _runner()
    assert r._needs_llm("Учебный центр «Ромашка»", r._service_aliases) is True
    assert r._needs_llm("22", r._service_aliases) is False        # numeric id
    assert r._needs_llm("Школа", r._service_aliases) is False      # catalogue alias
    assert r._needs_llm(None, r._service_aliases) is False


def test_llm_fallback_resolves_unknown_service_name(tmp_path) -> None:
    r = _runner()
    calls = {"n": 0}

    def fake_complete(messages, schema):
        calls["n"] += 1
        return {key: "22" for key in schema["required"]}  # -> school 22 -> 3.5.1

    r._llm_complete = fake_complete
    feats = _run_buildings(
        r, tmp_path, _unknown_service_layer(),
        building_type_col="", building_service_col="svc", building_floors_col="",
    )
    p = feats[0]["properties"]
    assert p[COL_MATCHED_VRI_CODE] == "3.5.1"
    assert p[COL_CATEGORY] == CATEGORY_SERVICE
    assert "сопоставлено ИИ" in p[COL_RESOLUTION_BASIS]
    assert calls["n"] == 1  # one batched call for the distinct unknown names


def test_llm_fallback_disabled_keeps_manual_review(tmp_path) -> None:
    r = UploadedBuildingPzzRunner(_settings(llm_fallback=False))
    calls = {"n": 0}
    r._llm_complete = lambda m, s: calls.__setitem__("n", calls["n"] + 1) or {}
    feats = _run_buildings(
        r, tmp_path, _unknown_service_layer(),
        building_type_col="", building_service_col="svc", building_floors_col="",
    )
    assert calls["n"] == 0  # LLM never consulted when the flag is off
    assert feats[0]["properties"][COL_VERDICT] == "Требуется ручная проверка"


def test_no_llm_call_when_everything_resolves(tmp_path) -> None:
    r = _runner()
    calls = {"n": 0}
    r._llm_complete = lambda m, s: calls.__setitem__("n", calls["n"] + 1) or {}
    _run_buildings(r, tmp_path, _buildings(), building_type_col="po", building_floors_col="floors")
    assert calls["n"] == 0  # ids/known names never trigger the LLM


def _pzz_index_zones() -> dict:
    """Zones keyed by a ПЗЗ letter index (no urban_api functional_zone_type_id)."""
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": _square(0, 0),
             "properties": {"Индекс_зоны": "Ж-1"}},
            {"type": "Feature", "geometry": _square(2, 0),
             "properties": {"Индекс_зоны": "П-1"}},
        ],
    }


def _pzz_index_labels() -> list:
    return [
        {"zone_code": "Ж-1", "zone_name": "Жилая зона",
         "main": [{"vri_code": "2.1.1"}], "conditional": [], "auxiliary": []},
        {"zone_code": "П-1", "zone_name": "Производственная зона",
         "main": [{"vri_code": "6.9"}], "conditional": [], "auxiliary": []},
    ]


def test_run_pzz_letter_index_backend(tmp_path) -> None:
    """Real ПЗЗ: zones carry letter indices (Ж-1 / П-1) and permitted ВРИ come from
    an uploaded label file in the pzz_check schema — no urban_api id involved."""
    (tmp_path / "b.geojson").write_text(json.dumps(_buildings()), encoding="utf-8")
    (tmp_path / "z.geojson").write_text(json.dumps(_pzz_index_zones()), encoding="utf-8")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(_pzz_index_labels()), encoding="utf-8")
    req = _req(
        cadastral_data_path=str(tmp_path / "b.geojson"),
        pzz_zones_data_path=str(tmp_path / "z.geojson"),
        pzz_zone_vri_labels_path=str(labels),
        pzz_zone_code_col="Индекс_зоны",
        outputs_dir=str(tmp_path / "out"),
    )
    props = [f["properties"] for f in json.load(open(_runner().run(req), encoding="utf-8"))["features"]]
    # residential 2.1.1 in Ж-1 (permits 2.1.1) -> allowed; in П-1 (only 6.9) -> not
    assert props[0][COL_VERDICT] == "Разрешен"
    assert props[0][COL_ZONE_CODE] == "Ж-1"          # user's verbatim index, not folded
    assert props[0][COL_ZONE_NAME] == "Жилая зона"    # matched template zone name
    assert props[0][COL_MATCHED_VRI_CODE] == "2.1.1"
    assert props[1][COL_VERDICT] == "Не разрешен"
    assert props[1][COL_ZONE_CODE] == "П-1"


def test_run_pzz_letter_index_normalises_spelling(tmp_path) -> None:
    """A folded code («ж 1», en-dash) still matches the «Ж-1» label entry."""
    zones = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": _square(0, 0),
             "properties": {"Индекс_зоны": "ж 1"}},
        ],
    }
    (tmp_path / "b.geojson").write_text(json.dumps(_buildings()), encoding="utf-8")
    (tmp_path / "z.geojson").write_text(json.dumps(zones), encoding="utf-8")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(_pzz_index_labels()), encoding="utf-8")
    req = _req(
        cadastral_data_path=str(tmp_path / "b.geojson"),
        pzz_zones_data_path=str(tmp_path / "z.geojson"),
        pzz_zone_vri_labels_path=str(labels),
        pzz_zone_code_col="Индекс_зоны",
        outputs_dir=str(tmp_path / "out"),
    )
    props = [f["properties"] for f in json.load(open(_runner().run(req), encoding="utf-8"))["features"]]
    assert props[0][COL_VERDICT] == "Разрешен"
    assert props[0][COL_ZONE_CODE] == "ж 1"  # display keeps the user's spelling


def test_confirmed_overlay_resolves_uncovered_zone(tmp_path) -> None:
    """End-to-end of the confirm flow's output: a confirmed {СХ-3 → АГ-1} overlay,
    fed back as the descriptions file, makes the user's СХ-3 zone resolve against
    АГ-1's permitted ВРИ."""
    from service.application.use_cases.building_zone_review import build_confirmed_overlay

    template = tmp_path / "template.json"
    template.write_text(json.dumps([
        {"zone_code": "АГ-1", "zone_name": "Сельхоз-жилая",
         "main": [{"vri_code": "2.1.1"}], "conditional": [], "auxiliary": []},
    ]), encoding="utf-8")
    overlay = build_confirmed_overlay(str(template), {"СХ-3": "АГ-1"})
    labels = tmp_path / "overlay.json"
    labels.write_text(json.dumps(overlay), encoding="utf-8")

    zones = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": _square(0, 0), "properties": {"Индекс_зоны": "СХ-3"}}]}
    (tmp_path / "b.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": _point(0.5, 0.5),
         "properties": {"po": 4, "floors": 3}}]}), encoding="utf-8")
    (tmp_path / "z.geojson").write_text(json.dumps(zones), encoding="utf-8")
    req = _req(
        cadastral_data_path=str(tmp_path / "b.geojson"),
        pzz_zones_data_path=str(tmp_path / "z.geojson"),
        pzz_zone_vri_labels_path=str(labels),
        pzz_zone_code_col="Индекс_зоны",
        outputs_dir=str(tmp_path / "out"),
    )
    props = [f["properties"] for f in json.load(open(_runner().run(req), encoding="utf-8"))["features"]]
    assert props[0][COL_VERDICT] == "Разрешен"       # 2.1.1 permitted via АГ-1
    assert props[0][COL_ZONE_CODE] == "СХ-3"          # user's own code shown


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
