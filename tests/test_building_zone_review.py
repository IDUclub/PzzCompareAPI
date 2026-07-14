"""Tests for the building_pzz_check zone-review logic (pure, no SSE)."""
from __future__ import annotations

import json

from service.application.use_cases.building_zone_review import (
    UncoveredZone,
    build_confirmed_overlay,
    build_disclaimer,
    build_suggestion_messages,
    parse_suggestions,
    remaining_uncovered,
    review_building_zones,
    template_candidates,
)


def _zones(*codes: str, name_col: str | None = None) -> dict:
    feats = []
    for c in codes:
        props = {"Индекс_зоны": c}
        if name_col:
            props[name_col] = f"описание {c}"
        feats.append({"type": "Feature", "geometry": {}, "properties": props})
    return {"type": "FeatureCollection", "features": feats}


def _template(tmp_path, *entries) -> str:
    p = tmp_path / "template.json"
    p.write_text(json.dumps(list(entries)), encoding="utf-8")
    return str(p)


def _entry(code, name, main_vri) -> dict:
    return {"zone_code": code, "zone_name": name,
            "main": [{"vri_code": main_vri}], "conditional": [], "auxiliary": []}


def test_numeric_zones_proceed_not_approximate(tmp_path) -> None:
    zones = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {}, "properties": {"zone_code": 3}},
    ]}
    r = review_building_zones(
        zones_fc=zones, code_col="zone_code", name_col=None,
        user_uploaded_descriptions=False, confirmed_map=None,
        template_path=_template(tmp_path, _entry("Ж-1", "Жилая", "2.1")), threshold=5,
    )
    assert (r.numeric, r.approximate, r.action) == (True, False, "proceed")


def test_user_descriptions_proceed_not_approximate(tmp_path) -> None:
    r = review_building_zones(
        zones_fc=_zones("Ж-1"), code_col="Индекс_зоны", name_col=None,
        user_uploaded_descriptions=True, confirmed_map=None,
        template_path=_template(tmp_path, _entry("Ж-1", "Жилая", "2.1")), threshold=5,
    )
    assert (r.approximate, r.action) == (False, "proceed")


def test_template_covers_all_proceed_but_approximate(tmp_path) -> None:
    r = review_building_zones(
        zones_fc=_zones("Ж-1"), code_col="Индекс_зоны", name_col=None,
        user_uploaded_descriptions=False, confirmed_map=None,
        template_path=_template(tmp_path, _entry("Ж-1", "Жилая", "2.1")), threshold=5,
    )
    assert (r.approximate, r.action, r.uncovered) == (True, "proceed", [])


def test_few_uncovered_triggers_confirm(tmp_path) -> None:
    r = review_building_zones(
        zones_fc=_zones("Ж-1", "СХ-3", name_col="Код_объекта"),
        code_col="Индекс_зоны", name_col="Код_объекта",
        user_uploaded_descriptions=False, confirmed_map=None,
        template_path=_template(tmp_path, _entry("Ж-1", "Жилая", "2.1")), threshold=5,
    )
    assert r.action == "confirm"
    assert r.uncovered_codes == ["СХ-3"]
    assert r.uncovered[0].name == "описание СХ-3"  # name carried for the LLM prompt


def test_many_uncovered_triggers_suggest_upload(tmp_path) -> None:
    r = review_building_zones(
        zones_fc=_zones("А-1", "Б-2", "В-3", "Г-4", "Д-5"),
        code_col="Индекс_зоны", name_col=None,
        user_uploaded_descriptions=False, confirmed_map=None,
        template_path=_template(tmp_path, _entry("Ж-1", "Жилая", "2.1")), threshold=5,
    )
    assert r.action == "suggest_upload"
    assert len(r.uncovered) == 5


def test_confirmed_map_proceeds_and_stays_approximate(tmp_path) -> None:
    r = review_building_zones(
        zones_fc=_zones("Ж-1", "СХ-3"), code_col="Индекс_зоны", name_col=None,
        user_uploaded_descriptions=False, confirmed_map={"СХ-3": "Ж-1"},
        template_path=_template(tmp_path, _entry("Ж-1", "Жилая", "2.1")), threshold=5,
    )
    assert (r.approximate, r.action) == (True, "proceed")


def test_remaining_uncovered_excludes_confirmed(tmp_path) -> None:
    r = review_building_zones(
        zones_fc=_zones("СХ-3", "Т-1"), code_col="Индекс_зоны", name_col=None,
        user_uploaded_descriptions=False, confirmed_map=None,
        template_path=_template(tmp_path, _entry("Ж-1", "Жилая", "2.1")), threshold=5,
    )
    assert set(r.uncovered_codes) == {"СХ-3", "Т-1"}
    assert remaining_uncovered(r.uncovered, {"СХ-3": "Ж-1"}) == ["Т-1"]


def test_build_confirmed_overlay_maps_user_code_to_template_vri(tmp_path) -> None:
    path = _template(tmp_path, _entry("Ж-1", "Жилая", "2.1"), _entry("АГ-1", "Сельхоз", "1.15"))
    overlay = build_confirmed_overlay(path, {"СХ-3": "АГ-1"})
    added = [e for e in overlay if e["zone_code"] == "СХ-3"]
    assert len(added) == 1
    assert added[0]["main"] == [{"vri_code": "1.15"}]  # inherits АГ-1's permitted ВРИ
    assert "сопоставлено с АГ-1" in added[0]["zone_name"]
    # original entries preserved
    assert any(e["zone_code"] == "Ж-1" for e in overlay)


def test_template_candidates_lists_code_and_name(tmp_path) -> None:
    path = _template(tmp_path, _entry("Ж-1", "Жилая зона", "2.1"))
    assert template_candidates(path) == [{"code": "Ж-1", "name": "Жилая зона"}]


def test_suggestion_prompt_enums_candidates_and_parses_hits() -> None:
    uncovered = [UncoveredZone("СХ-3", "сельхоз", "сх3"), UncoveredZone("Т-1", "транспорт", "т1")]
    candidates = [{"code": "АГ-1", "name": "Сельхоз"}, {"code": "Ж-1", "name": "Жилая"}]
    messages, schema = build_suggestion_messages(uncovered, candidates)
    # schema enumerates the candidate codes + null, one required key per zone
    assert set(schema["required"]) == {"z0", "z1"}
    assert schema["properties"]["z0"]["enum"] == ["АГ-1", "Ж-1", None]
    assert "АГ-1: Сельхоз" in messages[1]["content"]
    # parsing keeps valid picks, drops null / invented codes
    parsed = {"z0": "АГ-1", "z1": None}
    assert parse_suggestions(uncovered, parsed, candidates) == {"СХ-3": "АГ-1"}
    assert parse_suggestions(uncovered, {"z0": "ZZ-9", "z1": "Ж-1"}, candidates) == {"Т-1": "Ж-1"}


def test_disclaimer_lists_remaining() -> None:
    text = build_disclaimer(["Т-1", "Т-2"])
    assert "приблизительной" in text
    assert "Т-1, Т-2" in text
    assert build_disclaimer([]).count("\n") == 0  # no second line when all covered
