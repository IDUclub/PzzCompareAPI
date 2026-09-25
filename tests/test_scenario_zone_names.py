"""Project (scenario) zones are named by their own ``name`` first, then by a Russian type label."""

import json
from types import SimpleNamespace

from service.api.scenarios import _flatten_functional_zone_features
from service.domain.contracts import PipelineRequest
from service.infrastructure.functional_zone_names import (
    UNKNOWN_ZONE_TYPE_LABEL,
    functional_zone_display_name,
    functional_zone_type_label,
)
from service.infrastructure.runners.deterministic_scenario_runner import (
    DeterministicScenarioRunner,
)

_RESIDENTIAL = {
    "id": 1,
    "name": "residential",
    "nickname": "Жилая зона",
    "description": "Жилой профиль",
}
_BASIC = {"id": 8, "name": "basic", "nickname": "basic", "description": "Без профиля"}
_UNKNOWN = {"id": 14, "name": "unknown", "nickname": "unknown", "description": "--"}


def test_zone_own_name_wins_over_type():
    props = {"name": "  Ж-1 Зона застройки ИЖС ", "functional_zone_type": _RESIDENTIAL}
    assert functional_zone_display_name(props) == "Ж-1 Зона застройки ИЖС"


def test_zone_without_name_falls_back_to_russian_type():
    for name in (None, "  "):
        props = {"name": name, "functional_zone_type": _RESIDENTIAL}
        assert functional_zone_display_name(props) == "Жилая зона"


def test_english_type_nicknames_become_russian():
    assert functional_zone_type_label(_BASIC) == "Зона без профиля"
    assert functional_zone_type_label(_UNKNOWN) == UNKNOWN_ZONE_TYPE_LABEL
    new_type = {
        "name": "new_type",
        "nickname": "new_type",
        "description": "Новый профиль",
    }
    assert functional_zone_type_label(new_type) == "Новый профиль"
    no_label = {"name": "x", "nickname": "x", "description": "--"}
    assert functional_zone_type_label(no_label) == UNKNOWN_ZONE_TYPE_LABEL
    assert functional_zone_type_label(None) == UNKNOWN_ZONE_TYPE_LABEL


def test_flatten_prefers_zone_name_then_russian_type():
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "properties": {
                    "name": "Парк Победы",
                    "functional_zone_type": _RESIDENTIAL,
                }
            },
            {"properties": {"name": None, "functional_zone_type": _RESIDENTIAL}},
            {"properties": {"name": None, "functional_zone_type": _UNKNOWN}},
        ],
    }
    out = _flatten_functional_zone_features(
        fc, code_col="zone_code", name_col="zone_name"
    )
    props = [f["properties"] for f in out["features"]]
    assert [p["zone_code"] for p in props] == ["1", "1", "14"]
    assert [p["zone_name"] for p in props] == [
        "Парк Победы",
        "Жилая зона",
        UNKNOWN_ZONE_TYPE_LABEL,
    ]


def _zone(x0: float, name: str | None, zone_type: dict) -> dict:
    ring = [[x0, 59.0], [x0 + 0.01, 59.0], [x0 + 0.01, 59.01], [x0, 59.01], [x0, 59.0]]
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {"name": name, "functional_zone_type": zone_type},
    }


def _building(x: float) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [x, 59.005]},
        "properties": {
            "physical_object_type": {"physical_object_type_id": 4, "name": "Жилой дом"},
            "properties": {"Количество этажей": 2},
            "vri_text": "Жилой дом, 2 этажей",
        },
    }


def test_scenario_runner_names_each_object_by_its_own_zone(tmp_path):
    zones = {
        "type": "FeatureCollection",
        "features": [
            _zone(30.0, "Квартал А", _RESIDENTIAL),
            _zone(30.1, None, _RESIDENTIAL),
            _zone(30.2, None, _BASIC),
        ],
    }
    zones = _flatten_functional_zone_features(
        zones, code_col="zone_code", name_col="zone_name"
    )
    objects = {
        "type": "FeatureCollection",
        "features": [_building(30.005), _building(30.105), _building(30.205)],
    }
    (tmp_path / "o.geojson").write_text(json.dumps(objects), encoding="utf-8")
    (tmp_path / "z.geojson").write_text(json.dumps(zones), encoding="utf-8")
    runner = DeterministicScenarioRunner(
        SimpleNamespace(
            physical_object_type_to_vri_path="data/physical_object_type_to_vri.json",
            default_fz_to_pzz_mapping_path="data/functional_zones_to_pzz_mapping.json",
            service_type_to_vri_path="data/service_type_to_vri.json",
        )
    )
    request = PipelineRequest(
        task_external_id="ext-zones",
        cadastral_data_path=str(tmp_path / "o.geojson"),
        pzz_zones_data_path=str(tmp_path / "z.geojson"),
        pzz_zone_vri_labels_path="",
        vri_classifier_path="",
        include_pzz_check=True,
        cadastral_vri_col="vri_text",
        pzz_zone_code_col="zone_code",
        pzz_zone_name_col="zone_name",
        outputs_dir=str(tmp_path / "out"),
        is_scenario=True,
    )

    with open(runner.run(request), encoding="utf-8") as fh:
        collection = json.load(fh)
    result = [f["properties"] for f in collection["features"]]

    # Two residential zones are separate zones, not one type bucket.
    assert collection["zone_stats"] == {"zones_count": 3}
    assert all("zone_stats" not in p for p in result)

    assert [p["Название фактической зоны нахождения кадастра"] for p in result] == [
        "Квартал А",
        "Жилая зона",
        "Зона без профиля",
    ]
    assert [p["Код фактической зоны нахождения кадастра"] for p in result] == [
        "1",
        "1",
        "8",
    ]
    assert "«Квартал А»" in result[0]["Причина"]
    assert all("basic" not in p["Причина"] for p in result)


def test_allowed_reason_names_the_section_in_russian():
    from service.infrastructure.runners._deterministic_pzz import verdict

    allowed = {1: {"main": set(), "conditional": {"2.1"}, "auxiliary": set()}}
    machine, reason, _, _ = verdict("2.1", 1, allowed, {1: "Жилая зона"})

    assert machine == "allowed_conditional"
    assert reason == (
        "Тип использования 2.1 разрешён в зоне «Жилая зона» "
        "(условно разрешённый вид использования)."
    )
