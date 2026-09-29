"""The chat answer counts separate zones, not zone codes, when the runner recorded them."""

import json
from pathlib import Path
from types import SimpleNamespace

from service.api.tasks import build_object_zone_fit_response


class _Task:
    status = "finished"
    building_type_col = None
    building_service_col = None

    def __init__(self, result_path: str):
        self.result_path = result_path


def _feature(zone_code: str | None, **extra: str) -> dict:
    props = {"Вердикт_ПЗЗ": "Разрешен" if zone_code else "Нет пересечения с ПЗЗ"}
    if zone_code:
        props["Код фактической зоны нахождения кадастра"] = zone_code
    props.update(extra)
    return {"type": "Feature", "geometry": None, "properties": props}


def _chat(tmp_path: Path, collection: dict, **kwargs) -> dict:
    path = tmp_path / "result.geojson"
    path.write_text(json.dumps(collection, ensure_ascii=False), encoding="utf-8")
    return build_object_zone_fit_response(
        _Task(str(path)),
        "ext-zones",
        "object",
        SimpleNamespace(outputs_dir=str(tmp_path)),
        **kwargs,
    )


def test_scenario_answer_counts_separate_zones_and_their_types(tmp_path: Path):
    features = [
        _feature(code, **{"Категория_объекта": "Здание"})
        for code in ("1", "1", "1", "6")
    ]
    collection = {
        "type": "FeatureCollection",
        "features": features + [_feature(None, **{"Категория_объекта": "Здание"})],
        "zone_stats": {"zones_count": 3},
    }

    resp = _chat(tmp_path, collection, scenario=True)

    assert resp["summary"]["zones_count"] == 2
    assert resp["summary"]["zone_polygons_count"] == 3
    assert (
        "Из них 4 находятся в границах 3 территориальных зон ПЗЗ (2 типов зон), "
        "1 не пересеклись ни с одной зоной ПЗЗ." in resp["chat_message"]
    )


def test_single_zone_is_worded_in_the_singular(tmp_path: Path):
    collection = {
        "type": "FeatureCollection",
        "features": [_feature("1", **{"Категория_объекта": "Здание"})] * 2,
        "zone_stats": {"zones_count": 1},
    }

    resp = _chat(tmp_path, collection, scenario=True)

    assert "Все они находятся в границах 1 территориальной зоны ПЗЗ." in (
        resp["chat_message"]
    )


def test_object_result_without_zone_stats_speaks_of_zone_types(tmp_path: Path):
    collection = {
        "type": "FeatureCollection",
        "features": [
            _feature(code, **{"Категория_объекта": "Здание"}) for code in ("1", "6")
        ],
    }

    resp = _chat(tmp_path, collection, scenario=True)

    assert "zone_polygons_count" not in resp["summary"]
    assert "Все они находятся в границах территориальных зон ПЗЗ 2 типов." in (
        resp["chat_message"]
    )


def test_parcel_answer_counts_zone_codes(tmp_path: Path):
    collection = {
        "type": "FeatureCollection",
        "features": [_feature("Ж-1"), _feature("Ж-1"), _feature("П-1")],
    }

    resp = _chat(tmp_path, collection)

    assert "Все они находятся в границах 2 территориальных зон ПЗЗ." in (
        resp["chat_message"]
    )
