"""ICII-57: ``Вердикт_ПЗЗ`` now holds the Russian label (not machine ``allowed_main``)."""

import json
from pathlib import Path

from service.api.tasks import _classify_verdict, build_object_zone_fit_response


class _Task:
    def __init__(self, result_path: str):
        self.status = "finished"
        self.result_path = result_path


def _settings(tmp_path: Path):
    return type("S", (), {"outputs_dir": str(tmp_path)})()


def test_classify_verdict_maps_russian_labels():
    assert _classify_verdict("Разрешен") == "correct"
    assert _classify_verdict("Условно разрешен") == "correct"
    assert _classify_verdict("Разрешен как вспомогательный") == "correct"
    assert _classify_verdict("Не разрешен") == "wrong"
    assert _classify_verdict("Требуется ручная проверка") == "unclear"
    assert _classify_verdict("Нет пересечения с ПЗЗ") == "unclear"
    assert _classify_verdict("") == "unclear"
    assert _classify_verdict(None) == "unclear"
    # the old machine verdict must NOT be treated as correct anymore
    assert _classify_verdict("allowed_main") == "unclear"


def test_object_zone_fit_reads_status_and_ignores_urban_api(tmp_path: Path):
    result = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": None,
                "properties": {
                    "Вердикт_ПЗЗ": "Разрешен",
                    "ВРИ_ЕГРН": "Ж",
                    "Код фактической зоны нахождения кадастра": "Ж-1",
                    # urban_api passthrough must not affect the report:
                    "PHYSICAL_OBJECT_ID": 411484,
                    "physical_object_type": {"name": "Жилой дом"},
                },
            },
            {
                "type": "Feature",
                "geometry": None,
                "properties": {
                    "Вердикт_ПЗЗ": "Не разрешен",
                    "ВРИ_ЕГРН": "П",
                    "Код фактической зоны нахождения кадастра": "П-1",
                },
            },
            {
                "type": "Feature",
                "geometry": None,
                "properties": {
                    # no zone → counts toward not_in_zone
                    "Вердикт_ПЗЗ": "Нет пересечения с ПЗЗ"
                },
            },
        ],
    }
    f = tmp_path / "r.geojson"
    f.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    resp = build_object_zone_fit_response(
        _Task(str(f)), "ext-1", "object", _settings(tmp_path)
    )

    assert resp["summary"] == {
        "total": 3,
        "in_correct_zone": 1,
        "in_wrong_zone": 1,
        "unclear": 1,
        "zones_count": 2,
        "not_in_zone": 1,
        "by_verdict": {"Разрешен": 1, "Не разрешен": 1, "Нет пересечения с ПЗЗ": 1},
    }
    assert [o["verdict"] for o in resp["objects"]] == [
        "Разрешен",
        "Не разрешен",
        "Нет пересечения с ПЗЗ",
    ]
