"""Scenario building-pzz-check yields the same «здания» / «сервисы» layers as the file check."""

import json
from pathlib import Path
from types import SimpleNamespace

from service.api.scenarios import _flatten_physical_object_features
from service.api.tasks import build_object_zone_fit_response, build_result_geo_layers
from service.infrastructure.runners.deterministic_scenario_runner import (
    DeterministicScenarioRunner,
)


def _runner_settings():
    return SimpleNamespace(
        physical_object_type_to_vri_path="data/physical_object_type_to_vri.json",
        default_fz_to_pzz_mapping_path="data/functional_zones_to_pzz_mapping.json",
        service_type_to_vri_path="data/service_type_to_vri.json",
    )


def test_scenario_runner_resolves_category_and_basis():
    runner = DeterministicScenarioRunner(_runner_settings())

    code, _, basis, category = runner._resolve(
        {
            "physical_object_type": {"physical_object_type_id": 4},
            "properties": {"Количество этажей": 9},
        }
    )
    assert category == "Здание"
    assert code
    assert basis == "жилое здание — тип использования подобран по этажности, 9 эт."

    code, _, basis, category = runner._resolve(
        {"service_type": {"service_type_id": 110, "name": "Гостиница"}}
    )
    assert category == "Сервис"
    assert code == "4.7"
    assert basis == "сервис (service_type_id=110) — тип использования подобран по типу сервиса"

    # Unknown service type → still a service (manual review), not a building.
    code, _, basis, category = runner._resolve(
        {"service_type": {"service_type_id": 999999}}
    )
    assert (code, basis, category) == (None, "", "Сервис")


def test_flatten_labels_scenario_services():
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"properties": {"service_type": {"name": "Школа"}, "name": "Школа №5"}},
            {"properties": {"physical_object_type": {"name": "Жилой дом"}}},
        ],
    }
    out = _flatten_physical_object_features(fc, vri_col="vri_text")
    assert [f["properties"]["vri_text"] for f in out["features"]] == [
        "Школа, Школа №5",
        "Жилой дом",
    ]


class _Task:
    status = "finished"
    building_type_col = None
    building_service_col = None

    def __init__(self, result_path: str):
        self.result_path = result_path


def test_scenario_result_splits_into_building_and_service_layers(tmp_path: Path):
    settings = SimpleNamespace(
        outputs_dir=str(tmp_path), app_name="pzz", public_base_url="http://x"
    )
    layers = build_result_geo_layers(
        _Task("r.geojson"), "ext-sc", settings, None, scenario=True
    )
    assert [layer["name"] for layer in layers] == [
        "buildings_result",
        "services_result",
    ]


def test_scenario_zone_fit_with_categories_keeps_scenario_subject(tmp_path: Path):
    result = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": None,
                "properties": {
                    "Категория_объекта": cat,
                    "Вердикт_ПЗЗ": "Разрешен",
                    "Код фактической зоны нахождения кадастра": "1",
                },
            }
            for cat in ("Здание", "Здание", "Сервис")
        ],
    }
    f = tmp_path / "scenario.geojson"
    f.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    resp = build_object_zone_fit_response(
        _Task(str(f)),
        "ext-sc",
        "object",
        SimpleNamespace(outputs_dir=str(tmp_path)),
        scenario=True,
    )
    assert resp["subject"] == "scenario_object"
    assert resp["mode"] == "building_pzz_check"
    assert resp["summary"]["by_category"] == {"Здание": 2, "Сервис": 1}
    assert "Проверено объектов сценария: 3 (зданий: 2, сервисов: 1)." in resp[
        "chat_message"
    ]
    # Objects have a usage type, not a land parcel's ВРИ.
    assert "ВРИ" not in resp["chat_message"]
    assert "Тип использования допустим" in resp["chat_message"]


def test_problem_objects_use_object_attribute_names_in_object_mode():
    from service.application.use_cases.chat_answer import _extract_problem_objects

    ozf = {
        "objects": [
            {
                "fit": "wrong",
                "verdict": "Не разрешен",
                "matched_vri_code": "4.4",
                "matched_vri_name": "Магазины",
                "resolution_basis": "сервис — тип использования подобран по типу сервиса",
            }
        ]
    }
    obj = _extract_problem_objects(ozf, 5, object_mode=True)[0]
    assert obj["Код_типа_использования"] == "4.4"
    assert obj["Тип_использования"] == "Магазины"
    assert "Основание_подбора_типа_использования" in obj
    assert not any("ВРИ" in k for k in obj)
    parcel = _extract_problem_objects(ozf, 5)[0]
    assert parcel["Код_подобранного_ВРИ"] == "4.4"
    assert obj.get("Исходный_тип_объекта") is None and "Исходный_тип_объекта" in obj
    assert "ВРИ_ЕГРН" in parcel
