"""Tests for the service placement mode (подбор мест размещения)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from service.application.use_cases import resolve_service_vri as resolve_mod
from service.application.use_cases.resolve_service_vri import (
    resolve_deterministic,
    resolve_service_vri,
)
from service.domain import PipelineRequest
from service.infrastructure.runners.placement_pzz_runner import (
    CATEGORY_CLASSES,
    CATEGORY_EXISTING,
    CATEGORY_NORMS,
    CATEGORY_REACH,
    CATEGORY_SZZ,
    COL_CATEGORY,
    COL_CLASS,
    COL_REACH_SHARE,
    COL_REASON,
    COL_REQUIREMENT,
    COL_SOURCE,
    COL_ZONE_CODE,
    COL_ZONE_LIMITS,
    COL_ZONE_NAME,
    PLACEMENT_SUMMARY_KEY,
    PlacementPzzRunner,
    _excluded_target_regex,
    _target_regex,
    classify_parcel,
    is_szz_sensitive,
)

CLASSIFIER_PATH = "data/rosreestr_vri_classifier_2024_12_24.json"
DEFAULT_LABELS_PATH = "data/pzz_zone_llm_labels_template.json"
SERVICE_MAP_PATH = "data/service_type_to_vri.json"


def _resolver_settings() -> SimpleNamespace:
    return SimpleNamespace(
        default_vri_classifier_path=CLASSIFIER_PATH,
        service_type_to_vri_path=SERVICE_MAP_PATH,
        vectorizer_url="",
        building_semantic_fallback=False,
    )


def _runner_settings() -> SimpleNamespace:
    return SimpleNamespace(
        default_vri_classifier_path=CLASSIFIER_PATH,
        default_fz_to_pzz_mapping_path="data/functional_zones_to_pzz_mapping.json",
        default_pzz_zone_labels_path=DEFAULT_LABELS_PATH,
    )


# --- service name -> ВРИ code ------------------------------------------------


@pytest.mark.parametrize(
    ("query", "code", "source"),
    [
        ("школа", "3.5.1", "catalog"),
        ("где построить детсад", "3.5.1", None),
        ("3.5.1", "3.5.1", "code"),
        ("ВРИ 4.4", "4.4", "code"),
        ("поликлиника", "3.4.1", None),
        ("АЗС", "4.9.1.1", None),
        ("магазин", "4.4", None),
        ("хочу разместить бассейн", "5.1.3", None),
    ],
)
def test_resolve_deterministic(query: str, code: str, source: str | None) -> None:
    resolved = resolve_deterministic(query, _resolver_settings())
    assert resolved is not None
    assert resolved.code == code
    assert resolved.vri_name
    if source is not None:
        assert resolved.source == source
    assert resolved.source_ru


def test_resolve_deterministic_names() -> None:
    assert resolve_deterministic("школа", _resolver_settings()).service_name == "Школа"
    kindergarten = resolve_deterministic("где построить детсад", _resolver_settings())
    assert kindergarten.service_name == "Детский сад"


def test_resolve_deterministic_unknown() -> None:
    assert resolve_deterministic("фывапролд", _resolver_settings()) is None


class _FakeChatClient:
    def __init__(self, answer: dict) -> None:
        self._answer = answer
        self.calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def complete_json(self, messages, **kwargs) -> dict:
        self.calls += 1
        return self._answer


def test_resolve_service_vri_llm_fallback(monkeypatch) -> None:
    client = _FakeChatClient({"vri_code": "4.4"})
    monkeypatch.setattr(resolve_mod, "build_chat_llm_client", lambda s: client)
    resolved = asyncio.run(resolve_service_vri("фывапролд", _resolver_settings()))
    assert resolved is not None
    assert resolved.code == "4.4"
    assert resolved.source == "llm"
    assert client.calls == 1


@pytest.mark.parametrize("answer", [{}, {"vri_code": "99.99"}])
def test_resolve_service_vri_llm_unknown_code(monkeypatch, answer: dict) -> None:
    client = _FakeChatClient(answer)
    monkeypatch.setattr(resolve_mod, "build_chat_llm_client", lambda s: client)
    assert asyncio.run(resolve_service_vri("фывапролд", _resolver_settings())) is None


def test_resolve_service_vri_deterministic_skips_llm(monkeypatch) -> None:
    client = _FakeChatClient({"vri_code": "4.4"})
    monkeypatch.setattr(resolve_mod, "build_chat_llm_client", lambda s: client)
    resolved = asyncio.run(resolve_service_vri("школа", _resolver_settings()))
    assert resolved.code == "3.5.1"
    assert client.calls == 0


def test_resolve_service_vri_empty_query() -> None:
    assert asyncio.run(resolve_service_vri("   ", _resolver_settings())) is None


# --- cadastre parcel helpers --------------------------------------------------


def test_classify_parcel() -> None:
    school_re = re.compile("школ")
    assert classify_parcel("Для размещения школы", "3.5.1", school_re) == "target"
    assert classify_parcel("3.5.1", "3.5.1", None) == "target"
    assert (
        classify_parcel("Для размещения производственных зданий", "3.5.1", school_re)
        == "production"
    )
    assert classify_parcel("Для ИЖС", "3.5.1", school_re) == "housing"
    assert classify_parcel("Магазины", "3.5.1", school_re) == "other"
    assert classify_parcel("Нежилое здание", "3.5.1", school_re) == "other"


def test_a_kindergarten_is_not_an_existing_school() -> None:
    school_re = _target_regex("3.5.1", "Школа")
    preschool = _excluded_target_regex("Школа")
    for text in ("Для размещения детского сада", "Дошкольное образование"):
        assert classify_parcel(text, "3.5.1", school_re, preschool) == "other", text
    for text in ("Для размещения школы", "Школа и детский сад", "Гимназия"):
        assert classify_parcel(text, "3.5.1", school_re, preschool) == "target", text
    assert _excluded_target_regex("Детский сад") is None
    kindergarten_re = _target_regex("3.5.1", "Детский сад")
    assert (
        classify_parcel("Для размещения детского сада", "3.5.1", kindergarten_re)
        == "target"
    )


def test_vocational_and_higher_education_is_not_an_existing_school() -> None:
    school_re = _target_regex("3.5.1", "Школа")
    preschool = _excluded_target_regex("Школа")
    for text in (
        "среднее и высшее профессиональное образование",
        "объекты высшего профессионального образования",
        "под размещение здания Гатчинского педагогического училища (колледжа)",
        "для размещения существующего здания автомобильной школы",
        "образовательные учреждения начального и среднего профессионального "
        "образования",
    ):
        assert classify_parcel(text, "3.5.1", school_re, preschool) == "other", text
    for text in (
        "для размещения средней школы №9",
        "для размещения музыкальной школы",
        "для размещения объектов образования",
        "колледж и общеобразовательная школа",
    ):
        assert classify_parcel(text, "3.5.1", school_re, preschool) == "target", text
    kindergarten_re = _target_regex("3.5.1", "Детский сад")
    assert classify_parcel("Колледж, детский сад", "3.5.1", kindergarten_re) == "target"
    college_re = _target_regex("3.5.2", "Колледж")
    assert classify_parcel("Колледж", "3.5.2", college_re) == "target"


def test_is_szz_sensitive() -> None:
    for code in ("3.5.1", "2.1", "5.1.3", "3.4.1"):
        assert is_szz_sensitive(code), code
    for code in ("4.4", "6.9", "3.7"):
        assert not is_szz_sensitive(code), code


# --- runner -------------------------------------------------------------------

LON, LAT = 30.0, 60.0


def _box(x0: float, y0: float, dx: float, dy: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [[x0, y0], [x0 + dx, y0], [x0 + dx, y0 + dy], [x0, y0 + dy], [x0, y0]]
        ],
    }


def _feature(geometry: dict, **props) -> dict:
    return {"type": "Feature", "properties": props, "geometry": geometry}


def _write(path: Path, data) -> str:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _zone_labels() -> list[dict]:
    return [
        {
            "zone_code": "Ж-1",
            "zone_name": "Жилая",
            "main": [{"vri_code": "3.5.1"}, {"vri_code": "2.1"}],
            "conditional": [],
            "auxiliary": [],
        },
        {
            "zone_code": "П-1",
            "zone_name": "Производственная",
            "main": [{"vri_code": "6.9"}],
            "conditional": [],
            "auxiliary": [],
        },
        {
            "zone_code": "О-1",
            "zone_name": "Общественно-деловая",
            "main": [{"vri_code": "4.4"}],
            "conditional": [{"vri_code": "3.5.1"}],
            "auxiliary": [],
        },
    ]


def _placement_request(
    tmp_path: Path,
    *,
    with_cadastre: bool,
    context: dict | None = None,
    cadastre: list[dict] | None = None,
) -> PipelineRequest:
    zones = {
        "type": "FeatureCollection",
        "features": [
            _feature(_box(LON, LAT, 0.01, 0.01), zone_code="Ж-1", zone_name="Жилая"),
            _feature(
                _box(LON + 0.01, LAT, 0.01, 0.01),
                zone_code="П-1",
                zone_name="Производственная",
            ),
            _feature(
                _box(LON + 0.03, LAT, 0.01, 0.01),
                zone_code="О-1",
                zone_name="Общественно-деловая",
            ),
            _feature(
                _box(LON + 0.04, LAT, 0.01, 0.01),
                zone_code="Х-1",
                zone_name="Без описания",
            ),
        ],
    }
    cadastre_path = ""
    if with_cadastre:
        small = 0.0005
        cadastre_fc = {
            "type": "FeatureCollection",
            "features": cadastre
            or [
                # Production parcel in П-1 ~30 m from Ж-1: its СЗЗ cuts into Ж-1.
                _feature(
                    _box(LON + 0.0105, LAT + 0.005, small, small),
                    ВРИ="Для размещения производственных зданий",
                ),
                # An existing school in О-1 -> B*.
                _feature(
                    _box(LON + 0.035, LAT + 0.005, small, small),
                    ВРИ="Для размещения школы",
                ),
                # Housing in Х-1 right next to О-1.
                _feature(
                    _box(LON + 0.0405, LAT + 0.005, small, small),
                    ВРИ="Для индивидуального жилищного строительства",
                ),
            ],
        }
        cadastre_path = _write(tmp_path / "cadastre.geojson", cadastre_fc)
    context_path = ""
    if context is not None:
        context_path = _write(tmp_path / "placement_context.json", context)
    return PipelineRequest(
        task_external_id="plc-test",
        cadastral_data_path=cadastre_path,
        pzz_zones_data_path=_write(tmp_path / "zones.geojson", zones),
        pzz_zone_vri_labels_path=(
            DEFAULT_LABELS_PATH
            if context is not None
            else _write(tmp_path / "labels.json", _zone_labels())
        ),
        vri_classifier_path=CLASSIFIER_PATH,
        include_pzz_check=True,
        cadastral_vri_col="ВРИ" if with_cadastre else "",
        pzz_zone_code_col="zone_code",
        pzz_zone_name_col="zone_name",
        outputs_dir=str(tmp_path / "out"),
        is_placement=True,
        placement_vri_code="3.5.1",
        placement_service_name="Школа",
        placement_context_path=context_path,
    )


def _result_path(tmp_path: Path) -> Path:
    return tmp_path / "out" / "pzz_compare_spatial_first_plc-test.geojson"


def _run(tmp_path: Path, *, with_cadastre: bool, **kwargs) -> dict:
    request = _placement_request(tmp_path, with_cadastre=with_cadastre, **kwargs)
    PlacementPzzRunner(_runner_settings()).run(request)
    return json.loads(_result_path(tmp_path).read_text(encoding="utf-8"))


def _class_features(result: dict) -> list[dict]:
    return [
        f["properties"]
        for f in result["features"]
        if f["properties"][COL_CATEGORY] == CATEGORY_CLASSES
    ]


def test_runner_with_cadastre(tmp_path: Path) -> None:
    result = _run(tmp_path, with_cadastre=True)
    summary = result[PLACEMENT_SUMMARY_KEY]
    classes = _class_features(result)
    by_zone: dict[str, set[str]] = {}
    for props in classes:
        by_zone.setdefault(props[COL_ZONE_CODE], set()).add(props[COL_CLASS])

    # Ж-1: main use, but the production СЗЗ turns a strip of it into C.
    assert by_zone["Ж-1"] == {"A", "C"}
    assert by_zone["П-1"] == {"C"}
    # О-1: conditional use with an existing school -> B*.
    assert by_zone["О-1"] == {"B*"}
    assert by_zone["Х-1"] == {"D"}

    # Without NormGraph norms there is no housing reach to measure.
    assert all(COL_REACH_SHARE not in p for p in classes)

    assert summary["vri_code"] == "3.5.1"
    assert summary["vri_name"]
    assert summary["service_name"] == "Школа"
    assert summary["zones_total"] == 4
    assert summary["cadastre_used"] is True
    assert summary["szz_applied"] is True
    assert summary["szz_ha"] > 0
    # «Производственная» names no hazard class -> IV, 100 m.
    assert summary["szz_by_class"] == [
        {"hazard_class": "IV", "width_m": 100.0, "parcels": 1}
    ]
    assert summary["szz_default_class_parcels"] == 1
    assert summary["reach_m"] is None
    assert summary["reach_ha"] == 0.0
    assert summary["norms"] == []
    assert summary["regulations"]["source"] == "user"
    assert summary["used_default_mapping"] is False
    assert summary["uncovered_zones"] == ["Х-1"]
    assert summary["parcels"] == {"production": 1, "target": 1, "housing": 1}
    assert list(summary["classes"]) == ["A", "A*", "B", "B*", "C", "D"]
    assert summary["classes"]["A"]["zones"] == 1
    assert summary["classes"]["B*"]["zones"] == 1
    assert summary["classes"]["C"]["zones"] == 2
    assert summary["classes"]["D"]["zones"] == 1
    assert [z["code"] for z in summary["best_zones"]] == ["Ж-1"]

    categories = [f["properties"][COL_CATEGORY] for f in result["features"]]
    assert categories.count(CATEGORY_SZZ) == 1
    assert categories.count(CATEGORY_NORMS) == 0
    assert categories.count(CATEGORY_REACH) == 0
    assert categories.count(CATEGORY_EXISTING) == 1


def _ng_zone(code: str, name: str, uses: dict, parameters=()) -> dict:
    return {
        "code": code,
        "name": name,
        "uses": [
            {"section": section, "name": name, "codes": codes}
            for section, codes in uses.items()
        ],
        "parameters": list(parameters),
    }


def _context() -> dict:
    return {
        "version": 1,
        "service_name": "Школа",
        "regulations": {
            "document": {"doc_id": "pzz-1", "name": "ПЗЗ Гатчины", "version": "2023"},
            "zone_code_overlap": 0.75,
            "zones": [
                _ng_zone(
                    "Ж-1",
                    "Жилая",
                    {"main": ["2.1", "3.5.1"]},
                    [{"kind": "max_floors", "value": 3, "number": "1"}],
                ),
                _ng_zone("П-1", "Производственная", {"main": ["6.9"]}),
                _ng_zone("О-1", "Общественно-деловая", {"conditional": ["3.5.1"]}),
            ],
        },
        "entities": ["школа"],
        "rules": [
            {
                "restriction_id": "r-1",
                "strictness": "strict",
                "effect": "exclude_on",
                "requirement": "не размещать на жилых участках",
                "text": "…",
                "document": "СП 2.4.2.4283-26",
                "clause": "3.1",
                "reviewed": False,
                "layer": "housing",
            },
            {
                "restriction_id": "r-2",
                "strictness": "soft",
                "effect": "prefer_near",
                "requirement": "в пределах 500 м от жилых участков",
                "text": "…",
                "document": "СП 2.4.2.4283-26",
                "clause": "3.2",
                "reviewed": False,
                "layer": "housing",
                "distance_m": 500.0,
            },
            {
                "restriction_id": "r-3",
                "strictness": "unchecked",
                "effect": "none",
                "requirement": "норма текстом",
                "text": "норма текстом",
                "document": "СП 1",
                "clause": "",
                "reviewed": True,
                "reason": "норма не формализована для автоматической проверки",
            },
        ],
        "notes": ["Не учтены нормы исключённых (тестовых) документов: 4."],
    }


def _context_cadastre() -> list[dict]:
    small = 0.0005
    return [
        # In a zone of hazard class III -> a 300 m СЗЗ reaching deep into Ж-1.
        _feature(
            _box(LON + 0.0105, LAT + 0.005, small, small),
            ВРИ="Для размещения производственных зданий",
        ),
        # Housing at the far edge of Ж-1: cut out by the strict norm.
        _feature(
            _box(LON + 0.001, LAT + 0.005, small, small),
            ВРИ="Для индивидуального жилищного строительства",
        ),
        # A kindergarten is not a school: Ж-1 stays A, not A*.
        _feature(
            _box(LON + 0.002, LAT + 0.001, small, small),
            ВРИ="Для размещения детского сада",
        ),
        _feature(
            _box(LON + 0.035, LAT + 0.005, small, small),
            ВРИ="Для размещения школы",
        ),
    ]


def _run_with_context(tmp_path: Path) -> dict:
    request = _placement_request(
        tmp_path, with_cadastre=True, context=_context(), cadastre=_context_cadastre()
    )
    zones_path = Path(request.pzz_zones_data_path)
    zones = json.loads(zones_path.read_text(encoding="utf-8"))
    zones["features"][1]["properties"][
        "zone_name"
    ] = "Зона производственных объектов III класса опасности"
    _write(zones_path, zones)
    PlacementPzzRunner(_runner_settings()).run(request)
    return json.loads(_result_path(tmp_path).read_text(encoding="utf-8"))


def test_runner_with_normgraph_context(tmp_path: Path) -> None:
    result = _run_with_context(tmp_path)
    summary = result[PLACEMENT_SUMMARY_KEY]
    classes = _class_features(result)

    by_zone: dict[str, set[str]] = {}
    for props in classes:
        by_zone.setdefault(props[COL_ZONE_CODE], set()).add(props[COL_CLASS])
    assert by_zone == {"Ж-1": {"A", "C"}, "П-1": {"C"}, "О-1": {"B*"}, "Х-1": {"D"}}
    # NormGraph ПЗЗ names the zones; the uploaded name only feeds the hazard class.
    names = {p[COL_ZONE_CODE]: p[COL_ZONE_NAME] for p in classes}
    assert names["П-1"] == "Производственная"
    assert names["Х-1"] == "Без описания"
    zh1_c_reasons = {
        p[COL_REASON]
        for p in classes
        if p[COL_ZONE_CODE] == "Ж-1" and p[COL_CLASS] == "C"
    }
    assert any("санитарно-защитную" in r for r in zh1_c_reasons)
    assert any("не размещать на жилых участках" in r for r in zh1_c_reasons)

    suitable = {p[COL_ZONE_CODE]: p for p in classes if p[COL_CLASS] in ("A", "B*")}
    assert suitable["Ж-1"][COL_ZONE_LIMITS] == "предельная этажность 3 эт."
    assert 0 < suitable["Ж-1"][COL_REACH_SHARE] < 100
    assert suitable["О-1"][COL_REACH_SHARE] == 0

    assert summary["regulations"] == {
        "source": "normgraph",
        "document": "ПЗЗ Гатчины",
        "version": "2023",
    }
    assert summary["used_default_mapping"] is False
    assert summary["uncovered_zones"] == ["Х-1"]
    assert summary["szz_by_class"] == [
        {"hazard_class": "III", "width_m": 300.0, "parcels": 1}
    ]
    assert summary["szz_default_class_parcels"] == 0
    assert summary["parcels"] == {
        "production": 1,
        "housing": 1,
        "other": 1,
        "target": 1,
    }
    assert summary["best_zones"][0]["limits"] == "предельная этажность 3 эт."
    assert summary["reach_m"] == 500.0 and summary["reach_ha"] > 0
    assert summary["reach_zones"] == 1
    statuses = {n["requirement"]: n["status"] for n in summary["norms"]}
    assert statuses == {
        "не размещать на жилых участках": "applied",
        "в пределах 500 м от жилых участков": "applied",
        "норма текстом": "listed",
    }
    strict = summary["norms"][0]
    assert strict["ha"] > 0 and strict["document"] == "СП 2.4.2.4283-26"
    assert summary["notes"] == _context()["notes"]

    overlays = [
        f["properties"]
        for f in result["features"]
        if f["properties"][COL_CATEGORY] != CATEGORY_CLASSES
    ]
    norm_layer = [p for p in overlays if p[COL_CATEGORY] == CATEGORY_NORMS]
    assert len(norm_layer) == 1
    assert norm_layer[0][COL_REQUIREMENT] == "не размещать на жилых участках"
    assert norm_layer[0][COL_SOURCE] == "СП 2.4.2.4283-26, п. 3.1"
    reach_layer = [p for p in overlays if p[COL_CATEGORY] == CATEGORY_REACH]
    assert reach_layer[0][COL_REQUIREMENT] == "в пределах 500 м от жилых участков"
    assert [p[COL_CATEGORY] for p in overlays].count(CATEGORY_EXISTING) == 1


def test_strict_norms_without_cadastre_data_are_not_applied(tmp_path: Path) -> None:
    school = _feature(_box(LON + 0.035, LAT + 0.005, 0.0005, 0.0005), ВРИ="Школа")
    result = _run(tmp_path, with_cadastre=True, context=_context(), cadastre=[school])
    summary = result[PLACEMENT_SUMMARY_KEY]
    assert [n["status"] for n in summary["norms"][:2]] == [
        "not_applied",
        "not_applied",
    ]
    assert summary["reach_m"] is None
    assert summary["classes"]["A"]["zones"] == 1
    assert summary["classes"]["B*"]["zones"] == 1


def test_default_labels_are_not_the_users_descriptions(tmp_path: Path) -> None:
    request = dataclasses.replace(
        _placement_request(tmp_path, with_cadastre=False),
        pzz_zone_vri_labels_path=DEFAULT_LABELS_PATH,
    )
    PlacementPzzRunner(_runner_settings()).run(request)
    result = json.loads(_result_path(tmp_path).read_text(encoding="utf-8"))
    summary = result[PLACEMENT_SUMMARY_KEY]
    assert summary["used_default_mapping"] is True
    assert summary["regulations"]["source"] == "default"


def test_runner_without_cadastre(tmp_path: Path) -> None:
    result = _run(tmp_path, with_cadastre=False)
    summary = result[PLACEMENT_SUMMARY_KEY]
    classes = _class_features(result)

    assert len(classes) == len(result["features"])
    assert {p[COL_ZONE_CODE]: p[COL_CLASS] for p in classes} == {
        "Ж-1": "A",
        "П-1": "C",
        "О-1": "B",
        "Х-1": "D",
    }
    assert all(COL_REACH_SHARE not in p for p in classes)
    assert summary["cadastre_used"] is False
    assert summary["szz_applied"] is False
    assert summary["parcels"] == {}


# --- routing ------------------------------------------------------------------


def test_factory_picks_placement_runner(monkeypatch) -> None:
    from service.infrastructure.runners import pipeline_runner as runner_mod

    monkeypatch.setattr(runner_mod, "get_object_storage", lambda: object())
    request = PipelineRequest(
        task_external_id="x",
        cadastral_data_path="",
        pzz_zones_data_path="z.geojson",
        pzz_zone_vri_labels_path="",
        vri_classifier_path="",
        include_pzz_check=True,
        cadastral_vri_col="",
        pzz_zone_code_col="zone_code",
        pzz_zone_name_col="",
        outputs_dir="out",
        is_placement=True,
        placement_vri_code="3.5.1",
    )
    runner = runner_mod.PipelineRunnerFactory.create(_runner_settings(), request)
    assert isinstance(runner._inner, PlacementPzzRunner)


def test_enqueue_placement_goes_to_default_queue(monkeypatch) -> None:
    from service import tasks as tasks_mod

    captured: dict = {}

    def _fake_apply_async(*, args, queue):
        captured.update(args=args, queue=queue)

    monkeypatch.setattr(
        tasks_mod.execute_pipeline_task, "apply_async", _fake_apply_async
    )
    tasks_mod.enqueue_pipeline_task(7, is_scenario=False, is_placement=True)
    assert captured == {"args": [7], "queue": "default"}


# --- API helpers ----------------------------------------------------------------


def _task(**overrides) -> SimpleNamespace:
    base = {
        "external_id": "plc-test",
        "status": "finished",
        "result_path": "",
        "placement_vri_code": "3.5.1",
        "cadastral_data_path": "cad.geojson",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_placement_slots() -> None:
    from service.api.tasks import _placement_slots

    assert _placement_slots(_task()) == [
        "placement_classes",
        "placement_szz",
        "placement_existing",
    ]
    assert _placement_slots(_task(placement_context_path="ctx.json")) == [
        "placement_classes",
        "placement_szz",
        "placement_norms",
        "placement_reach",
        "placement_existing",
    ]
    assert _placement_slots(_task(placement_vri_code="4.4")) == [
        "placement_classes",
        "placement_existing",
    ]
    assert _placement_slots(_task(cadastral_data_path=None)) == ["placement_classes"]


def _summary() -> dict:
    return {
        "vri_code": "3.5.1",
        "vri_name": "Дошкольное, начальное и среднее общее образование",
        "service_name": "Школа",
        "classes": {
            "A": {"zones": 1, "ha": 12.5},
            "A*": {"zones": 1, "ha": 2.0},
            "B": {"zones": 0, "ha": 0.0},
            "B*": {"zones": 1, "ha": 3.0},
            "C": {"zones": 2, "ha": 20.0},
            "D": {"zones": 1, "ha": 6.2},
        },
        "zones_total": 4,
        "best_zones": [
            {
                "code": "Ж-1",
                "name": "Жилая",
                "ha": 12.5,
                "limits": "предельная этажность 3 эт.",
            }
        ],
        "regulations": {"source": "default", "document": "", "version": ""},
        "szz_applied": True,
        "szz_ha": 4.1,
        "szz_by_class": [{"hazard_class": "III", "width_m": 300.0, "parcels": 2}],
        "szz_default_class_parcels": 1,
        "norms": [
            {
                "strictness": "strict",
                "requirement": "не размещать на жилых участках",
                "document": "СП 2.4.2.4283-26",
                "clause": "3.1",
                "reviewed": False,
                "status": "applied",
                "note": "",
                "ha": 1.5,
            },
            {
                "strictness": "soft",
                "requirement": "в пределах 500 м от жилых участков",
                "document": "СП 2.4.2.4283-26",
                "clause": "",
                "reviewed": False,
                "status": "applied",
                "note": "",
                "ha": 0.0,
            },
            {
                "strictness": "unchecked",
                "requirement": "длинный текст нормы " * 20,
                "document": "СП 1",
                "clause": "",
                "reviewed": True,
                "status": "listed",
                "note": "норма не формализована для автоматической проверки",
                "ha": 0.0,
            },
        ],
        "reach_m": 500.0,
        "reach_ha": 80.0,
        "reach_zones": 2,
        "parcels": {"production": 1, "target": 1, "housing": 1},
        "cadastre_used": True,
        "used_default_mapping": True,
        "uncovered_zones": ["Х-1"],
        "notes": ["Не учтены нормы исключённых (тестовых) документов: 4."],
    }


def test_chat_message_placement_is_russian() -> None:
    from service.api.tasks import _build_chat_message_placement

    text = _build_chat_message_placement(_summary())
    assert text.startswith(
        "Подбор мест размещения: «Школа» (ВРИ 3.5.1 "
        "«Дошкольное, начальное и среднее общее образование»)."
    )
    assert "- Ж-1 — Жилая: 12,50 га; параметры зоны: предельная этажность 3 эт." in text
    assert "- A* — Подходит, но объект этого типа в зоне уже есть" in text
    assert "Х-1" in text
    assert "ориентировочный" in text
    assert "III класс — 300 м (участков 2)" in text
    assert "принят IV класс (100 м)" in text
    assert "Зона доступности жилья (500 м от жилых участков): 80,00 га" in text
    assert (
        "- не размещать на жилых участках (СП 2.4.2.4283-26, п. 3.1): "
        "применена, исключено 1,50 га." in text
    )
    assert "Нестрогие нормы" in text and "Нормы, не проверенные автоматически" in text
    assert "…" in text
    assert "Не прошли экспертную проверку: 2 из 3 норм" in text
    assert text.endswith("Не учтены нормы исключённых (тестовых) документов: 4.")
    # No internal English keys or parcel kinds leak into the user-facing text.
    assert not re.search(
        r"vri_code|best_zones|szz_applied|cadastre_used|uncovered_zones"
        r"|production|housing|target|strict|soft|unchecked|applied|listed"
        r"|normgraph|default|True|False|None",
        text,
    )


def test_chat_message_placement_names_the_normgraph_pzz() -> None:
    from service.api.tasks import _build_chat_message_placement

    summary = _summary()
    summary["regulations"] = {
        "source": "normgraph",
        "document": "ПЗЗ Гатчины",
        "version": "2019, ред. от 20.11.2023",
    }
    summary["used_default_mapping"] = False
    text = _build_chat_message_placement(summary)
    assert "из ПЗЗ в NormGraph: «ПЗЗ Гатчины» (2019, ред. от 20.11.2023)." in text
    assert "ориентировочный" not in text


def test_chat_message_placement_reads_an_old_summary() -> None:
    from service.api.tasks import _build_chat_message_placement

    summary = _summary()
    for key in ("regulations", "norms", "szz_by_class", "reach_m", "notes"):
        summary.pop(key)
    text = _build_chat_message_placement(summary)
    assert "ориентировочный" in text
    assert "Нормы NormGraph" not in text


def test_chat_message_placement_without_main_zones() -> None:
    from service.api.tasks import _build_chat_message_placement

    summary = _summary()
    summary["best_zones"] = []
    summary["classes"]["A"] = {"zones": 0, "ha": 0.0}
    summary["cadastre_used"] = False
    text = _build_chat_message_placement(summary)
    assert "Зон, где вид разрешён как основной, не найдено." in text
    assert "Кадастровый слой не передан" in text


def test_build_placement_summary_response(tmp_path: Path) -> None:
    from service.api.tasks import build_placement_summary_response

    result = {
        "type": "FeatureCollection",
        "features": [],
        PLACEMENT_SUMMARY_KEY: _summary(),
    }
    path = _write(tmp_path / "result.geojson", result)
    settings = SimpleNamespace(outputs_dir=str(tmp_path))

    payload = build_placement_summary_response(
        _task(result_path=path), "plc-test", settings
    )
    assert payload["mode"] == "placement"
    assert payload["task_external_id"] == "plc-test"
    assert payload["summary"]["vri_code"] == "3.5.1"
    assert payload["chat_message"].startswith("Подбор мест размещения")

    with pytest.raises(HTTPException) as exc:
        build_placement_summary_response(
            _task(status="running", result_path=path), "plc-test", settings
        )
    assert exc.value.status_code == 409
    with pytest.raises(HTTPException) as exc:
        build_placement_summary_response(
            _task(result_path=path, placement_vri_code=None), "plc-test", settings
        )
    assert exc.value.status_code == 404

    plain = _write(
        tmp_path / "plain.geojson", {"type": "FeatureCollection", "features": []}
    )
    with pytest.raises(HTTPException) as exc:
        build_placement_summary_response(_task(result_path=plain), "plc-test", settings)
    assert exc.value.status_code == 404


def test_serve_placement_split(tmp_path: Path) -> None:
    from service.api.tasks import _serve_placement_split

    result = _run_with_context(tmp_path)
    path = str(_result_path(tmp_path))
    settings = SimpleNamespace(outputs_dir=str(tmp_path / "out"))
    task = _task(result_path=path)

    expected = {
        "placement_classes": len(_class_features(result)),
        "placement_szz": 1,
        "placement_norms": 1,
        "placement_reach": 1,
        "placement_existing": 1,
    }
    for slot, count in expected.items():
        resp = _serve_placement_split(task, slot, settings)
        assert resp.media_type == "application/geo+json"
        assert len(json.loads(resp.body)["features"]) == count, slot

    with pytest.raises(HTTPException) as exc:
        _serve_placement_split(
            _task(status="running", result_path=path), "placement_classes", settings
        )
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        _serve_placement_split(
            _task(result_path=path, placement_vri_code=None),
            "placement_classes",
            settings,
        )
    assert exc.value.status_code == 404


# --- auto chat endpoint validation --------------------------------------------


def _post_auto_stream(data: dict, files: dict):
    from fastapi.testclient import TestClient

    from service import app as app_module
    from service.api.security import AuthUser, get_current_user

    app = app_module.app
    app.dependency_overrides[get_current_user] = lambda: AuthUser(
        token="t", user_id="u"
    )
    try:
        return TestClient(app).post("/tasks/auto/chat/stream", data=data, files=files)
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def _zones_upload() -> tuple:
    return (
        "zones.geojson",
        json.dumps({"type": "FeatureCollection", "features": []}),
        "application/geo+json",
    )


def test_auto_stream_placement_requires_zones() -> None:
    resp = _post_auto_stream(
        {"mode": "placement", "service_name": "школа"},
        {"cadastral_feature_collection_file": _zones_upload()},
    )
    assert resp.status_code == 422
    assert "pzz_zones_feature_collection_file" in resp.text


def test_auto_stream_placement_requires_service_name() -> None:
    resp = _post_auto_stream(
        {"mode": "placement"},
        {"pzz_zones_feature_collection_file": _zones_upload()},
    )
    assert resp.status_code == 422
    assert "service_name" in resp.text


def test_auto_stream_pzz_check_still_requires_cadastre() -> None:
    resp = _post_auto_stream(
        {"mode": "pzz_check"},
        {"pzz_zones_feature_collection_file": _zones_upload()},
    )
    assert resp.status_code == 422
    assert "cadastral_feature_collection_file" in resp.text


# --- detection narrative --------------------------------------------------------


def test_render_detection_narrative_custom_header() -> None:
    from service.api.classifier import _PLACEMENT_DETECTION_HEADER
    from service.application.use_cases.detect_columns import (
        PZZ_ZONE_TARGETS,
        ColumnSuggestion,
        render_detection_narrative,
    )

    suggestions = {
        "pzz_zone_code_col": ColumnSuggestion(
            value="zone_code", confidence=1.0, source="heuristic", reason=""
        )
    }
    default = render_detection_narrative(suggestions, PZZ_ZONE_TARGETS)
    custom = render_detection_narrative(
        suggestions, PZZ_ZONE_TARGETS, header=_PLACEMENT_DETECTION_HEADER
    )
    assert _PLACEMENT_DETECTION_HEADER in custom
    assert _PLACEMENT_DETECTION_HEADER not in default


# --- Scenario mode: the user's own ПЗЗ descriptions / ВРИ classifier ---------

import io  # noqa: E402

from fastapi import UploadFile  # noqa: E402

from service.api import classifier as classifier_api  # noqa: E402
from service.api import scenarios as scenarios_api  # noqa: E402

_SCENARIO_DESCRIPTIONS = {
    "functional_zone_mappings": [
        {
            "functional_zone_type_id": 4,
            "averaged_pzz_profile": {"main_vri": [{"vri_code": "3.5.1"}]},
        }
    ]
}


def _upload(data: bytes, filename: str = "file.json") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename)


def _upload_settings(tmp_path: Path, max_bytes: int = 1_000_000) -> SimpleNamespace:
    return SimpleNamespace(
        uploads_dir=str(tmp_path / "uploads"), max_upload_bytes=max_bytes
    )


def _read_files(tmp_path: Path, **kwargs):
    slots = {
        "pzz_descriptions_file": None,
        "pzz_descriptions_upload_id": None,
        "vri_classifier_file": None,
        "vri_classifier_upload_id": None,
    }
    slots.update(kwargs)
    return asyncio.run(
        scenarios_api.read_scenario_regulation_files(
            **slots, owner_id="u1", app_settings=_upload_settings(tmp_path)
        )
    )


def test_scenario_regulation_files_are_optional(tmp_path: Path) -> None:
    assert _read_files(tmp_path) == (None, None)


def test_scenario_regulation_files_are_parsed(tmp_path: Path) -> None:
    descriptions, classifier = _read_files(
        tmp_path,
        pzz_descriptions_file=_upload(
            json.dumps(_SCENARIO_DESCRIPTIONS).encode("utf-8")
        ),
        vri_classifier_file=_upload(b'\xef\xbb\xbf[{"code": "3.5.1"}]'),
    )
    assert descriptions == _SCENARIO_DESCRIPTIONS
    assert classifier == [{"code": "3.5.1"}]


def test_scenario_regulation_files_by_upload_id(tmp_path: Path, monkeypatch) -> None:
    stored = tmp_path / "stored" / "descriptions.json"
    stored.parent.mkdir()
    stored.write_text(json.dumps(_SCENARIO_DESCRIPTIONS), encoding="utf-8")
    monkeypatch.setattr(classifier_api, "resolve_upload", lambda *a, **k: stored)
    descriptions, _ = _read_files(tmp_path, pzz_descriptions_upload_id="abc")
    assert descriptions == _SCENARIO_DESCRIPTIONS
    # The scratch copy is cleaned up; the stored upload is kept.
    assert stored.exists()
    assert not any((tmp_path / "uploads").glob("resolve-*"))


@pytest.mark.parametrize(
    "doc",
    [
        {"zones": []},  # file-mode label JSON — can't match numeric scenario codes
        {"functional_zone_mappings": []},
        {"functional_zone_mappings": [{"functional_zone_type_id": "Ж1"}]},
        [1, 2],
    ],
)
def test_scenario_descriptions_wrong_schema_is_422(tmp_path: Path, doc) -> None:
    with pytest.raises(HTTPException) as exc:
        _read_files(
            tmp_path, pzz_descriptions_file=_upload(json.dumps(doc).encode("utf-8"))
        )
    assert exc.value.status_code == 422
    assert "functional_zone_mappings" in exc.value.detail


@pytest.mark.parametrize(
    "kwargs, status",
    [
        ({"vri_classifier_file": _upload(b"{}")}, 422),
        ({"vri_classifier_file": _upload(b"not json")}, 422),
        ({"vri_classifier_file": _upload(b"a;b", "classifier.csv")}, 415),
        ({"pzz_descriptions_file": _upload(b"x", "descriptions.xlsx")}, 415),
    ],
)
def test_scenario_regulation_files_rejected(tmp_path: Path, kwargs, status) -> None:
    kwargs = {k: _upload(v.file.getvalue(), v.filename) for k, v in kwargs.items()}
    with pytest.raises(HTTPException) as exc:
        _read_files(tmp_path, **kwargs)
    assert exc.value.status_code == status


def test_scenario_regulation_file_too_big_is_413(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            scenarios_api.read_scenario_regulation_files(
                pzz_descriptions_file=None,
                pzz_descriptions_upload_id=None,
                vri_classifier_file=_upload(b'["' + b"a" * 100 + b'"]'),
                vri_classifier_upload_id=None,
                owner_id="u1",
                app_settings=_upload_settings(tmp_path, max_bytes=10),
            )
        )
    assert exc.value.status_code == 413


def test_json_digest_ignores_key_order() -> None:
    a = scenarios_api._json_digest({"a": 1, "b": [1, 2]})
    assert a == scenarios_api._json_digest({"b": [1, 2], "a": 1})
    assert a != scenarios_api._json_digest({"a": 2, "b": [1, 2]})


def _persist_scenario_task(tmp_path: Path, monkeypatch, **kwargs) -> dict:
    persisted: dict[str, object] = {}

    def fake_persist(doc, task_dir, name, external_id, storage):
        persisted[name] = doc
        return f"s3://{external_id}/{name}"

    captured: dict = {}

    def fake_create_task(**kw):
        captured.update(kw)
        return SimpleNamespace(id=1, external_id="x")

    monkeypatch.setattr(scenarios_api, "persist_geojson_dict", fake_persist)
    monkeypatch.setattr(scenarios_api, "get_object_storage", lambda: None)
    monkeypatch.setattr(scenarios_api, "create_task", fake_create_task)
    monkeypatch.setattr(scenarios_api, "api_log", lambda *a, **k: None)
    settings = SimpleNamespace(
        task_inputs_dir=str(tmp_path / "inputs"),
        default_vri_classifier_path=str(tmp_path / "default_classifier.json"),
    )
    session = SimpleNamespace(flush=lambda: None, refresh=lambda task: None)
    scenarios_api._persist_and_create_scenario_task(
        objects_fc={"type": "FeatureCollection", "features": []},
        zones_fc={"type": "FeatureCollection", "features": []},
        scenario_id=1,
        year=2024,
        source="PZZ",
        physical_object_type_id=0,
        priority=5,
        namespaced_key="k",
        force_recompute=False,
        app_settings=settings,
        task_repo=None,
        event_repo=None,
        session=session,
        placement_vri_code="3.5.1",
        placement_service_name="школа",
        **kwargs,
    )
    return {"persisted": persisted, "input_paths": captured["input_paths"]}


def test_scenario_task_uses_uploaded_regulation_files(
    tmp_path: Path, monkeypatch
) -> None:
    out = _persist_scenario_task(
        tmp_path,
        monkeypatch,
        zone_descriptions=_SCENARIO_DESCRIPTIONS,
        vri_classifier=[{"code": "3.5.1"}],
    )
    paths = out["input_paths"]
    assert paths["pzz_zone_vri_labels_path"].endswith("/pzz_zone_descriptions.json")
    assert paths["vri_classifier_path"].endswith("/vri_classifier.json")
    assert out["persisted"]["pzz_zone_descriptions.json"] == _SCENARIO_DESCRIPTIONS
    assert out["persisted"]["vri_classifier.json"] == [{"code": "3.5.1"}]


def test_scenario_task_defaults_without_uploads(tmp_path: Path, monkeypatch) -> None:
    out = _persist_scenario_task(tmp_path, monkeypatch)
    paths = out["input_paths"]
    assert paths["pzz_zone_vri_labels_path"] == ""
    assert paths["vri_classifier_path"] == str(
        (tmp_path / "default_classifier.json").resolve()
    )
    assert "vri_classifier.json" not in out["persisted"]
