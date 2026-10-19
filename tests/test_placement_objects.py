"""Independent actual-object uploads must override only target existence."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from service.application.use_cases.placement_objects import (
    OBJECT_TYPE_TEXT,
    normalize_placement_objects,
)
from service.domain import PipelineRequest
from service.infrastructure.runners.placement_pzz_runner import (
    PlacementPzzRunner,
    PLACEMENT_SUMMARY_KEY,
)
from service.api.tasks import _placement_slots, _build_chat_message_placement


def settings():
    return SimpleNamespace(
        physical_object_type_to_vri_path="data/physical_object_type_to_vri.json",
        service_type_to_vri_path="data/service_type_to_vri.json",
        default_vri_classifier_path="data/rosreestr_vri_classifier_2024_12_24.json",
        default_fz_to_pzz_mapping_path="data/functional_zones_to_pzz_mapping.json",
        default_pzz_zone_labels_path="data/pzz_zone_llm_labels_template.json",
    )


def fc(*features):
    return {"type": "FeatureCollection", "features": list(features)}


def point(x=30.105, y=59.575, **props):
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [x, y]},
        "properties": props,
    }


def test_nested_and_flat_urban_types_keep_kindergarten_distinct_from_school():
    out = normalize_placement_objects(
        fc(point(service_type_id=21.0), point(service_type={"service_type_id": 22})),
        settings(),
    )
    texts = [f["properties"][OBJECT_TYPE_TEXT] for f in out["features"]]
    assert "Детский сад" in texts[0] and "ВРИ 3.5.1" in texts[0]
    assert "Школа" in texts[1] and "ВРИ 3.5.1" in texts[1]


def test_text_type_column_and_explicit_nonstandard_column():
    assert (
        normalize_placement_objects(fc(point(**{"Тип объекта": "Школа"})), settings())[
            "features"
        ][0]["properties"][OBJECT_TYPE_TEXT]
        == "Школа"
    )
    assert (
        normalize_placement_objects(
            fc(point(category="Школа")), settings(), "category"
        )["features"][0]["properties"][OBJECT_TYPE_TEXT]
        == "Школа"
    )
    with pytest.raises(ValueError, match="отсутствует"):
        normalize_placement_objects(fc(point(category="Школа")), settings(), "missing")
    with pytest.raises(ValueError, match="Не найдена"):
        normalize_placement_objects(fc(point(category="Школа")), settings())


def test_non_object_geometries_are_rejected():
    line = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[30.1, 59.57], [30.2, 59.58]],
        },
        "properties": {"тип": "Школа"},
    }
    with pytest.raises(ValueError, match="точками или полигонами"):
        normalize_placement_objects(fc(line), settings())


def run(tmp_path, objects, *, cadastre=False):
    def write(name, value):
        p = tmp_path / name
        p.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return str(p)

    zones = []
    for x, code in [(30.1, "Ж-1"), (30.12, "О-1")]:
        zones.append(
            {
                "type": "Feature",
                "properties": {"code": code},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [x, 59.57],
                            [x + 0.01, 59.57],
                            [x + 0.01, 59.58],
                            [x, 59.58],
                            [x, 59.57],
                        ]
                    ],
                },
            }
        )
    labels = [
        {
            "zone_code": "Ж-1",
            "zone_name": "Жилая",
            "main": [{"vri_code": "3.5.1"}],
            "conditional": [],
            "auxiliary": [],
        },
        {
            "zone_code": "О-1",
            "zone_name": "Общественная",
            "main": [],
            "conditional": [{"vri_code": "3.5.1"}],
            "auxiliary": [],
        },
    ]
    request = PipelineRequest(
        task_external_id="actual",
        cadastral_data_path=(
            write("cad.geojson", fc(point(30.125, 59.575, vri="Школа")))
            if cadastre
            else ""
        ),
        pzz_zones_data_path=write("zones.geojson", fc(*zones)),
        pzz_zone_vri_labels_path=write("labels.json", labels),
        vri_classifier_path=settings().default_vri_classifier_path,
        include_pzz_check=True,
        cadastral_vri_col="vri",
        pzz_zone_code_col="code",
        pzz_zone_name_col="",
        outputs_dir=str(tmp_path),
        is_placement=True,
        placement_vri_code="3.5.1",
        placement_service_name="Школа",
        existing_objects_data_path=write(
            "actual.geojson", normalize_placement_objects(objects, settings())
        ),
    )
    path = PlacementPzzRunner(settings()).run(request)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_actual_objects_work_without_cadastre(tmp_path):
    result = run(
        tmp_path,
        fc(point(service_type_id=22), point(30.125, 59.575, service_type_id=21)),
    )
    summary = result[PLACEMENT_SUMMARY_KEY]
    assert summary["existing_objects_used"] and not summary["cadastre_used"]
    assert summary["existing_objects_count"] == 1
    assert summary["existing_objects_source"] == "upload"
    assert summary["classes"]["A*"]["zones"] == 1
    assert summary["classes"]["B*"]["zones"] == 0
    assert not summary["szz_applied"]
    assert "существующие объекты не учитывались" not in _build_chat_message_placement(
        summary
    )


def test_uploaded_objects_replace_cadastral_guesses(tmp_path):
    summary = run(tmp_path, fc(point(service_type_id=22)), cadastre=True)[
        PLACEMENT_SUMMARY_KEY
    ]
    assert summary["classes"]["A*"]["zones"] == 1
    assert summary["classes"]["B*"]["zones"] == 0
    assert summary["existing_objects_count"] == 1


def test_empty_upload_clears_cadastral_existence_guesses(tmp_path):
    summary = run(tmp_path, fc(), cadastre=True)[PLACEMENT_SUMMARY_KEY]
    assert summary["existing_objects_used"]
    assert summary["existing_objects_count"] == 0
    assert summary["classes"]["A*"]["zones"] == summary["classes"]["B*"]["zones"] == 0


def test_layer_links_include_existing_objects_without_cadastre():
    task = SimpleNamespace(
        cadastral_data_path="",
        existing_objects_data_path="actual",
        placement_context_path="ctx",
        placement_vri_code="3.5.1",
    )
    assert _placement_slots(task) == [
        "placement_classes",
        "placement_norms",
        "placement_reach",
        "placement_existing",
    ]
