"""Placement norms: СЗЗ by hazard class, NormGraph restrictions as placement rules,
the NormGraph context of a placement and the client calls behind it."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from service.application.use_cases.check_vri import zone_limits
from service.application.use_cases.placement_context import (
    build_placement_context,
    parse_excluded_documents,
    zone_code_overlap,
)
from service.domain.placement import (
    EFFECT_EXCLUDE_NEAR,
    EFFECT_EXCLUDE_ON,
    EFFECT_NONE,
    EFFECT_PREFER_NEAR,
    LAYER_HOUSING,
    SOFT,
    STRICT,
    UNCHECKED,
    PlacementRule,
    hazard_class,
    rule_from_restriction,
    szz_width_m,
    with_existing,
)
from service.infrastructure.normgraph_client import NormGraphClient, NormGraphError
from service.infrastructure.runners._deterministic_pzz import normalise_zone_code

SCHOOL = frozenset({"школа", "школ"})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Зона производственных объектов III класса опасности", 3),
        ("Производственная зона объектов IV-V классов опасности", 4),
        ("Зона предприятий 2-го класса", 2),
        ("Зона объектов I, II и III класса", 1),
        ("ЗОНА ПРОИЗВОДСТВЕННЫХ ОБЪЕКТОВ III-V КЛАССОВ ОПАСНОСТИ", 3),
        ("ЗОНА КОММУНАЛЬНО-СКЛАДСКИХ ОБЪЕКТОВ IV-V КЛАССА ОПАСНОСТИ", 4),
        ("Производственная зона", None),
        ("", None),
        (None, None),
    ],
)
def test_hazard_class_takes_the_most_dangerous_class(text, expected):
    assert hazard_class(text) == expected


def test_szz_width_follows_the_hazard_class_with_iv_by_default():
    assert [szz_width_m(c) for c in (1, 2, 3, 4, 5)] == [1000, 500, 300, 100, 50]
    assert szz_width_m(None) == 100


def test_an_existing_object_marks_only_suitable_classes():
    assert with_existing("A") == "A*"
    assert with_existing("B") == "B*"
    assert with_existing("C") == "C"


def _hit(template, params=None, layers=(), **extra):
    return {
        "id": extra.pop("id", "r-1"),
        "extraction_text": extra.pop("text", "текст нормы"),
        "provenance": {"name": "СП 2.4.2.4283-26", "numbering": "2.1"},
        "check_plan_review_status": extra.pop("review", "pending"),
        "check_plan": {
            "template": template,
            "params": params or {},
            "declared_requirements": {
                "layers": [
                    {"role": role, "entity": entity, "entity_type": "service"}
                    for role, entity in layers
                ]
            },
        },
        **extra,
    }


def test_not_on_housing_is_a_strict_cut_of_the_housing_layer():
    rule = rule_from_restriction(
        _hit(
            "distance_from_source",
            {
                "source_layer": "src",
                "targets": ["dst"],
                "geometry_mode": "source_geometry",
            },
            [("src", "школа"), ("dst", "жилой дом")],
        ),
        SCHOOL,
    )
    assert (rule.strictness, rule.effect, rule.layer) == (
        STRICT,
        EFFECT_EXCLUDE_ON,
        LAYER_HOUSING,
    )
    assert rule.requirement == "не размещать на жилых участках"
    assert rule.document == "СП 2.4.2.4283-26" and rule.clause == "2.1"
    assert rule.reviewed is False


def test_not_closer_than_is_a_strict_buffer():
    rule = rule_from_restriction(
        _hit(
            "distance_from_source",
            {
                "source_layer": "src",
                "targets": ["dst"],
                "geometry_mode": "buffered",
                "distance_m": 50,
            },
            [("src", "многоквартирный дом"), ("dst", "школ")],
            review="approved",
        ),
        SCHOOL,
    )
    assert (rule.strictness, rule.effect, rule.distance_m) == (
        STRICT,
        EFFECT_EXCLUDE_NEAR,
        50,
    )
    assert rule.requirement == "не ближе 50 м от жилых участков"
    assert rule.reviewed is True


def test_within_reach_of_housing_is_soft():
    rule = rule_from_restriction(
        _hit(
            "accessibility_within",
            {
                "objects_layer": "obj",
                "required_neighbor_layers": ["home"],
                "limit": {"kind": "distance", "meters": 500},
            },
            [("obj", "школа"), ("home", "жилая застройка")],
        ),
        SCHOOL,
    )
    assert (rule.strictness, rule.effect, rule.distance_m) == (
        SOFT,
        EFFECT_PREFER_NEAR,
        500,
    )
    assert rule.requirement == "в пределах 500 м от жилых участков"


def test_a_threshold_on_the_object_is_soft_and_only_listed():
    rule = rule_from_restriction(_hit("object_attribute_threshold"), SCHOOL)
    assert (rule.strictness, rule.effect) == (SOFT, EFFECT_NONE)
    assert "проектировании" in rule.reason


@pytest.mark.parametrize(
    ("hit", "reason"),
    [
        (_hit("unsupported"), "не формализована"),
        (_hit("geometry_magic"), "не поддерживается"),
        (
            _hit(
                "distance_from_source",
                {"source_layer": "a", "targets": ["b"], "distance_m": 10},
                [("a", "гараж"), ("b", "жилой дом")],
            ),
            "не относится к размещаемому объекту",
        ),
        (
            _hit(
                "distance_from_source",
                {"source_layer": "a", "targets": ["b"], "distance_m": 10},
                [("a", "школа"), ("b", "автодорога")],
            ),
            "нет данных о слое «автодорога»",
        ),
        (
            _hit(
                "distance_from_source",
                {"source_layer": "a", "targets": ["b"], "geometry_mode": "buffered"},
                [("a", "школа"), ("b", "жилой дом")],
            ),
            "не указано расстояние",
        ),
        (
            _hit(
                "presence_within",
                {
                    "objects_layer": "a",
                    "required_neighbor_layers": ["b"],
                    "limit": {"kind": "time", "minutes": 15},
                },
                [("a", "школа"), ("b", "жилой дом")],
            ),
            "не расстоянием",
        ),
    ],
)
def test_norms_the_placement_cannot_check_are_listed_with_a_reason(hit, reason):
    rule = rule_from_restriction(hit, SCHOOL)
    assert rule.strictness == UNCHECKED and rule.effect == EFFECT_NONE
    assert reason in rule.reason
    assert rule.requirement == "текст нормы"


def test_a_rule_survives_the_json_round_trip():
    rule = rule_from_restriction(_hit("object_attribute_threshold"), SCHOOL)
    raw = json.loads(json.dumps(rule.to_dict(), ensure_ascii=False))
    assert PlacementRule.from_dict({**raw, "unknown": 1}) == rule


def test_zone_limits_prefer_the_rows_of_the_use_and_the_tightest_value():
    zone = {
        "parameters": [
            {"kind": "max_floors", "value": 9, "number": "1"},
            {"kind": "max_floors", "value": 5, "number": "2", "vri_codes": ["3.5.1"]},
            {"kind": "max_floors", "value": 4, "number": "3", "vri_codes": ["3.5.1"]},
            {"kind": "max_height", "not_set": True},
            {"kind": "max_coverage", "maximum": 40, "number": "4"},
            {"kind": "max_height", "value": 12, "building": "residential"},
        ]
    }
    limits = {item["kind"]: item for item in zone_limits(zone, "3.5.1")}
    assert limits["max_floors"]["value"] == 4 and limits["max_floors"]["clause"] == "3"
    assert limits["max_coverage"]["value"] == 40
    assert "max_height" not in limits
    general = {item["kind"]: item["value"] for item in zone_limits(zone, "4.4")}
    assert general["max_floors"] == 9
    residential = {
        item["kind"]: item["value"]
        for item in zone_limits(zone, "2.1", residential=True)
    }
    assert residential["max_height"] == 12


# --- the NormGraph context ----------------------------------------------------

GATCHINA = {
    "doc_id": "pzz-gatchina",
    "name": "ПЗЗ Гатчины",
    "version": "2023",
    "territory_id": 73,
    "effective_date": "2023-02-13",
    "zones": 3,
}
OTHER = {**GATCHINA, "doc_id": "pzz-other", "name": "ПЗЗ другого МО"}
ELSEWHERE = {**GATCHINA, "doc_id": "pzz-elsewhere", "territory_id": 99}
ZONES = {
    "pzz-gatchina": [
        {"code": "Ж-1", "name": "Жилая", "uses": [], "document": GATCHINA},
        {"code": "О-1", "name": "Общественная", "uses": []},
        {"code": "П-1", "name": "Производственная", "uses": []},
    ],
    "pzz-other": [{"code": "Ж1", "name": "Жилая", "uses": []}],
    "pzz-elsewhere": [
        {"code": "Ж-1", "name": "Жилая", "uses": []},
        {"code": "О-1", "name": "Общественная", "uses": []},
        {"code": "П-1", "name": "Производственная", "uses": []},
        {"code": "Х-9", "name": "Иная", "uses": []},
    ],
}
RESOLVED = [
    {
        "term": "Школа",
        "candidates": [
            {"normalized": "школ", "match": "alias", "aliases": ["школа"]},
            {"normalized": "спортивная школа", "match": "text"},
            {"normalized": "школа", "match": "layer"},
        ],
    }
]
HITS = [
    _hit(
        "distance_from_source",
        {"source_layer": "a", "targets": ["b"], "geometry_mode": "source_geometry"},
        [("a", "школа"), ("b", "жилой дом")],
        id="r-1",
        duplicate_group="g-1",
    ),
    _hit("unsupported", id="r-2", duplicate_group="g-1"),
    {**_hit("unsupported", id="r-3"), "provenance": {"name": "Тестовые нормы"}},
    {**_hit("unsupported", id="r-4"), "provenance": {"doc_id": "pzz-gatchina"}},
    {**_hit("unsupported", id="r-5"), "provenance": {"doc_id": "pzz-elsewhere"}},
]


class FakeNormGraph:
    def __init__(self, *, fail_regulations=False, fail_norms=False, resolved=None):
        self.fail_regulations = fail_regulations
        self.fail_norms = fail_norms
        self.resolved = RESOLVED if resolved is None else resolved
        self.calls: list[tuple] = []

    async def regulation_documents(self, territory_ids):
        self.calls.append(("documents", tuple(territory_ids)))
        if self.fail_regulations:
            raise NormGraphError(503, "down")
        return [
            OTHER,
            GATCHINA,
            ELSEWHERE,
            {**GATCHINA, "doc_id": "empty", "zones": 0},
        ]

    async def zones(self, doc_id):
        self.calls.append(("zones", doc_id))
        return ZONES[doc_id]

    async def resolve_entities(self, terms, *, limit=10):
        self.calls.append(("resolve", tuple(terms)))
        return self.resolved

    async def list_restrictions(self, entities, *, executable_only=False):
        self.calls.append(("restrictions", tuple(entities)))
        if self.fail_norms:
            raise httpx.ConnectError("no route")
        return HITS


def _context(normgraph, **kwargs):
    return asyncio.run(
        build_placement_context(
            normgraph,
            service_name="Школа",
            zone_codes=["Ж-1", "О-1", "П-1", "Х-9"],
            search_terms=["школу", "Школа"],
            excluded_documents=parse_excluded_documents("Тестовые Нормы; "),
            **kwargs,
        )
    )


def test_the_context_takes_the_pzz_covering_the_zones_and_the_service_norms():
    normgraph = FakeNormGraph()
    context = _context(normgraph, territory_id=73)

    regulations = context["regulations"]
    assert regulations["document"]["doc_id"] == "pzz-gatchina"
    assert regulations["zone_code_overlap"] == 0.75
    assert all("document" not in zone for zone in regulations["zones"])
    assert ("documents", ()) in normgraph.calls
    assert ("zones", "empty") not in normgraph.calls
    assert ("zones", "pzz-elsewhere") not in normgraph.calls

    assert ("resolve", ("Школа", "школу")) in normgraph.calls
    assert context["entities"] == ["школ", "школа"]
    assert [r["restriction_id"] for r in context["rules"]] == ["r-1"]
    assert context["rules"][0]["strictness"] == STRICT
    assert context["notes"] == [
        "Положения ПЗЗ (2) не учтены как нормы: ПЗЗ учтены через разрешённые виды "
        "использования зон.",
        "Не учтены нормы исключённых (тестовых) документов: 1.",
    ]
    json.dumps(context, ensure_ascii=False)


def test_without_a_territory_the_best_covering_pzz_of_any_territory_is_taken():
    context = _context(FakeNormGraph())
    assert context["regulations"]["document"]["doc_id"] == "pzz-elsewhere"


def test_pzz_provisions_are_not_norms_even_when_no_pzz_matches_the_zones():
    context = asyncio.run(
        build_placement_context(
            FakeNormGraph(), service_name="Школа", zone_codes=["А", "Б", "В"]
        )
    )
    assert context["regulations"] is None
    assert [r["restriction_id"] for r in context["rules"]] == ["r-1", "r-3"]
    assert any(note.startswith("Положения ПЗЗ (2)") for note in context["notes"])


def test_a_pzz_with_other_zone_codes_is_not_taken():
    context = asyncio.run(
        build_placement_context(
            FakeNormGraph(), service_name="Школа", zone_codes=["А", "Б", "В"]
        )
    )
    assert context["regulations"] is None
    assert any("не найден документ ПЗЗ" in note for note in context["notes"])


def test_normgraph_failures_become_notes():
    context = _context(FakeNormGraph(fail_regulations=True, fail_norms=True))
    assert context["regulations"] is None
    assert context["rules"] == []
    assert any("ответ 503" in note for note in context["notes"])
    assert any("ConnectError" in note for note in context["notes"])


def test_a_service_unknown_to_normgraph_has_no_norms():
    normgraph = FakeNormGraph(resolved=[{"term": "Школа", "candidates": []}])
    context = _context(normgraph)
    assert context["rules"] == []
    assert not any(call[0] == "restrictions" for call in normgraph.calls)
    assert "В NormGraph нет норм для объекта «Школа»." in context["notes"]


def test_without_normgraph_the_context_says_so():
    context = _context(None)
    assert context["regulations"] is None and context["rules"] == []
    assert context["notes"][0].startswith("NormGraph не подключён")


def test_zone_code_overlap_compares_normalised_codes():
    uploaded = {normalise_zone_code(code) for code in ("ж 1", "X")}
    assert zone_code_overlap(set(), ZONES["pzz-gatchina"]) == 0.0
    assert zone_code_overlap(uploaded, ZONES["pzz-gatchina"]) == 0.5


# --- the NormGraph client ---------------------------------------------------


class FakeTokenClient:
    async def get_authorization_headers(self, *, force_refresh=False):
        return {"Authorization": "Bearer service-tok"}


def _client(handler) -> NormGraphClient:
    client = NormGraphClient("http://ng.local", FakeTokenClient())
    client._client = httpx.AsyncClient(
        base_url="http://ng.local", transport=httpx.MockTransport(handler)
    )
    return client


def test_restrictions_are_read_page_by_page():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        assert request.headers["Authorization"] == "Bearer service-tok"
        if "after_id" not in body:
            return httpx.Response(
                200, json={"hits": [{"id": "1"}, {"id": "2"}], "next_after_id": "2"}
            )
        return httpx.Response(200, json={"hits": [{"id": "3"}], "next_after_id": None})

    async def go():
        async with _client(handler) as client:
            return await client.list_restrictions([f"e{i}" for i in range(60)])

    hits = asyncio.run(go())
    assert [h["id"] for h in hits] == ["1", "2", "3"]
    assert len(bodies[0]["entities"]) == 50
    assert bodies[0]["executable_only"] is False
    assert bodies[1]["after_id"] == "2"


def test_restrictions_stop_at_the_cap():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        hits = [{"id": str(i)} for i in range(body["limit"])]
        return httpx.Response(200, json={"hits": hits, "next_after_id": "more"})

    async def go():
        async with _client(handler) as client:
            return await client.list_restrictions(["школа"], max_items=250)

    assert len(asyncio.run(go())) == 250


def test_entities_are_resolved_and_errors_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["terms"] == ["школа"]:
            assert body["limit"] == 10
            return httpx.Response(200, json=RESOLVED)
        return httpx.Response(422, json={"detail": "bad"})

    async def go(terms):
        async with _client(handler) as client:
            return await client.resolve_entities(terms)

    assert asyncio.run(go(["школа"])) == RESOLVED
    with pytest.raises(NormGraphError) as caught:
        asyncio.run(go(["x"]))
    assert caught.value.status == 422


# --- the API builds the context ---------------------------------------------


class _ClosableNormGraph(FakeNormGraph):
    closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        self.closed = True


def _api_context(monkeypatch, normgraph):
    from types import SimpleNamespace

    from service.api import classifier

    monkeypatch.setattr(classifier, "build_normgraph_client", lambda s: normgraph)
    settings = SimpleNamespace(placement_excluded_norm_documents="Тестовые Нормы")
    zones_fc = {
        "features": [
            {"properties": {"code": code}}
            for code in ("Ж-1", "О-1", "П-1", "Ж-1", "", None)
        ]
        + [{"properties": None}]
    }
    return asyncio.run(
        classifier._placement_context(
            settings,
            service_name="Школа",
            service_query="где поставить школу",
            zones_fc=zones_fc,
            zone_code_col="code",
            territory_id=73,
        )
    )


def test_the_api_builds_the_context_from_the_uploaded_zones(monkeypatch):
    normgraph = _ClosableNormGraph()
    context = _api_context(monkeypatch, normgraph)
    assert normgraph.closed
    assert ("resolve", ("Школа", "где поставить школу")) in normgraph.calls
    assert context["regulations"]["document"]["doc_id"] == "pzz-gatchina"
    assert ("zones", "pzz-elsewhere") not in normgraph.calls
    assert context["regulations"]["zone_code_overlap"] == 1.0
    assert [r["restriction_id"] for r in context["rules"]] == ["r-1"]


def test_the_api_context_without_normgraph(monkeypatch):
    context = _api_context(monkeypatch, None)
    assert context["regulations"] is None
    assert context["notes"][0].startswith("NormGraph не подключён")


def test_the_same_clause_extracted_twice_is_one_rule():
    twice = [
        _hit("unsupported", id="a", text="допускается размещать  магазин"),
        _hit("unsupported", id="b", text="допускается размещать магазин"),
        _hit("unsupported", id="c", text=""),
        _hit("unsupported", id="d", text=""),
        _hit("unsupported", id="a", text="другой текст"),
    ]
    normgraph = FakeNormGraph()
    normgraph.list_restrictions = lambda entities, **kw: _done(twice)
    context = _context(normgraph)
    assert [r["restriction_id"] for r in context["rules"]] == ["a", "c", "d"]


async def _done(value):
    return value
