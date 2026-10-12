"""Point check of a ВРИ against the ПЗЗ of a territory (NormGraph regulations)."""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from service.api import regulations as regulations_api
from service.application.use_cases.check_vri import (
    DEPENDS_ON_ZONE,
    PARAMS_EXCEEDED,
    PARAMS_NOT_CHECKED,
    PARAMS_OK,
    Regulations,
    check,
    resolve_vri,
)
from service.application.use_cases.pzz_zone_types import (
    METHOD_CODE_PREFIX,
    METHOD_LLM,
    base_code,
    base_zones,
    code_prefix,
    mapping_messages,
    mapping_schema,
    parse_mapping,
    prefix_mapping,
    vote,
    zone_kind,
    zone_types,
)
from service.infrastructure import pzz_regulations
from service.infrastructure.chat_llm_client import ChatLlmError
from service.infrastructure.normgraph_client import NormGraphClient, NormGraphError
from service.infrastructure.pzz_regulations import (
    PzzNotFound,
    edition_key,
    resolve_territory_pzz,
    zone_type_mapping,
)
from service.infrastructure.urban_api_client import UrbanApiError

DOCUMENT = {
    "doc_id": "pzz-1",
    "name": "Правила землепользования и застройки МО «Город Гатчина»",
    "version": "2019 (ред. от 13.02.2023)",
    "territory_id": 73,
    "territory_name": "Гатчинское городское поселение",
}


def _param(kind, value=None, **extra):
    return {"kind": kind, "name": kind, "value": value, "number": "1", **extra}


def _zone(code, name, uses, parameters=(), group=""):
    return {
        "code": code,
        "name": name,
        "group": group,
        "uses": [
            {"section": section, "name": f"вид {code}", "codes": list(codes)}
            for section, codes in uses.items()
        ],
        "parameters": list(parameters),
    }


ZONES = [
    _zone(
        "Ж-1",
        "Зона застройки индивидуальными жилыми домами",
        {"main": ["2.1"], "conditional": ["4.4"]},
        [_param("max_floors", 3), _param("max_height", 20)],
        group="ЖИЛЫЕ ЗОНЫ",
    ),
    _zone(
        "Ж-1.10",
        "Зона застройки индивидуальными жилыми домами ОЗ-1",
        {"main": ["2.1"]},
        [_param("max_floors", 2)],
        group="ЖИЛЫЕ ЗОНЫ",
    ),
    _zone(
        "Ж-4",
        "Зона застройки многоэтажными жилыми домами",
        {"main": ["2.6"], "auxiliary": ["4.9"]},
        [
            _param("max_height", 40, building="residential"),
            _param("max_floors", 12, building="residential"),
            _param("max_height", 30, building="non_residential"),
            _param("max_floors", 5, vri_codes=["2.6"], number="2.1"),
        ],
        group="ЖИЛЫЕ ЗОНЫ",
    ),
    _zone("О-1", "Зона делового назначения", {"main": ["4.1", "4.4"]}),
    _zone(
        "Р-1", "Зона парков", {"main": ["5.1"]}, [_param("max_height", not_set=True)]
    ),
    _zone("И", "Зона объектов инженерной инфраструктуры", {"main": ["3.1.1"]}),
    _zone("Т", "Зона железнодорожного транспорта", {}),
]
PAYLOAD = {
    "document": DOCUMENT,
    "territory": {"id": 73, "name": "Гатчинское городское поселение"},
    "requested_territory": {"id": 73, "name": "Гатчинское городское поселение"},
    "zones": ZONES,
}
TEMPLATE = {
    "functional_zone_mappings": [
        {
            "functional_zone_type_id": type_id,
            "db_name": name,
            "db_zone_nickname": nickname,
            "db_description": "",
            "averaged_pzz_profile": {
                "main_vri": [{"vri_code": c, "vri_name": f"вид {c}"} for c in codes]
            },
        }
        for type_id, name, nickname, codes in (
            (1, "residential", "Жилая зона", ["2.1", "2.6", "3.1.1"]),
            (4, "industrial", "Промышленная зона", ["6.0", "3.1.1"]),
            (6, "transport", "Транспортная зона", ["7.1", "3.1.1"]),
            (7, "business", "Общественно-деловая зона", ["4.1", "3.1.1"]),
            (11, "residential_lowrise", "Малоэтажная жилая зона", ["2.1.1"]),
            (14, "unknown", "Тип зоны не определён", []),
        )
    ]
}
TYPES = zone_types(TEMPLATE)


def _regs():
    return Regulations(PAYLOAD)


def _zones(regs, *codes):
    return [regs.zone(code) for code in codes]


# --- verdicts in one zone ----------------------------------------------------


def test_zone_verdict_uses_the_sections_of_the_zone():
    regs = _regs()
    zone = regs.zone("ж 1")
    assert zone["code"] == "Ж-1"
    verdict, reason = regs.zone_verdict("2.1", zone)
    assert verdict == "allowed_main" and "зоне Ж-1" in reason
    assert regs.zone_verdict("4.4", zone)[0] == "allowed_conditional"
    assert regs.zone_verdict("6.1", zone)[0] == "not_allowed"
    assert regs.zone_verdict("2.1", regs.zone("Т"))[0] == "no_zone_metadata"


def test_floors_over_the_limit_are_reported():
    regs = _regs()
    label, reason = regs.check_parameters(
        regs.zone("Ж-1"), "2.1", floors=5, height=None, residential=True
    )
    assert label == PARAMS_EXCEEDED
    assert reason.startswith("Этажность 5 эт. больше предельной 3 эт.")
    assert "высота объекта неизвестна (предельная 20 м)" in reason
    assert reason.endswith("Зона Ж-1.")


def test_limits_for_residential_and_other_buildings_differ():
    regs = _regs()
    zone = regs.zone("Ж-4")
    label, reason = regs.check_parameters(
        zone, "2.6", floors=5, height=35, residential=True
    )
    assert label == PARAMS_OK
    assert "строка 2.1" in reason
    label, reason = regs.check_parameters(
        zone, "4.9", floors=None, height=35, residential=False
    )
    assert label == PARAMS_EXCEEDED
    assert "высота 35 м больше предельной 30 м" in reason.lower()


def test_a_row_for_one_use_wins_over_the_general_row():
    regs = _regs()
    zone = regs.zone("Ж-4")
    assert (
        regs.check_parameters(zone, "2.6", floors=9, height=None, residential=True)[0]
        == PARAMS_EXCEEDED
    )
    assert (
        regs.check_parameters(zone, "2.5", floors=9, height=None, residential=True)[0]
        == PARAMS_OK
    )


def test_unset_limits_and_unknown_values_are_not_checked():
    regs = _regs()
    label, reason = regs.check_parameters(
        regs.zone("Р-1"), "5.1", floors=2, height=8, residential=False
    )
    assert label == PARAMS_NOT_CHECKED
    assert "не подлежит установлению" in reason
    label, reason = regs.check_parameters(
        regs.zone("О-1"), "4.1", floors=2, height=8, residential=False
    )
    assert label == PARAMS_NOT_CHECKED
    assert "нет предельной этажности и высоты" in reason


# --- the verdict over the zones of a type ------------------------------------


def _check(vri, codes, floors=None):
    regs = _regs()
    return check(
        regs,
        {"code": vri},
        _zones(regs, *codes),
        where="ПЗЗ типа «Жилая зона»",
        floors=floors,
        height=None,
        residential=True,
    )


def test_a_use_allowed_in_every_zone_of_the_type_is_allowed():
    result = _check("2.1", ["Ж-1", "Ж-1.10"])
    assert result["verdict"] == "allowed_main"
    assert result["verdict_label"] == "Разрешен"
    assert "во всех зонах ПЗЗ типа «Жилая зона» (Ж-1, Ж-1.10)" in result["reason"]


def test_the_weakest_section_of_the_zones_wins():
    result = _check("4.4", ["Ж-1", "О-1"])
    assert result["verdict"] == "allowed_conditional"
    assert "только как условно разрешённый" in result["reason"]


def test_a_use_allowed_in_some_zones_depends_on_the_zone():
    result = _check("2.1", ["Ж-1", "Ж-4"], floors=3)
    assert result["verdict"] == DEPENDS_ON_ZONE
    assert result["verdict_label"] == "Зависит от зоны ПЗЗ"
    assert "разрешён в зонах Ж-1, не разрешён в Ж-4" in result["reason"]
    by_zone = {z["code"]: z for z in result["zones"]}
    assert by_zone["Ж-1"]["verdict"] == "allowed_main"
    assert by_zone["Ж-1"]["parameters"]["status"] == PARAMS_OK
    assert by_zone["Ж-4"]["verdict"] == "not_allowed"


def test_a_use_allowed_nowhere_is_not_allowed():
    result = _check("6.1", ["Ж-1", "Ж-4"])
    assert result["verdict"] == "not_allowed"
    assert "ни в одной из зон" in result["reason"]


def test_zones_without_uses_or_no_zones_leave_it_unclear():
    assert _check("2.1", ["Т"])["verdict"] == "no_zone_metadata"
    assert _check("2.1", ["Ж-4", "Т"])["verdict"] == "unclear"
    assert _check("2.1", ["Ж-1", "Т"])["verdict"] == DEPENDS_ON_ZONE
    empty = _check("2.1", [])
    assert empty["verdict"] == "unclear" and empty["zones"] == []
    regs = _regs()
    no_vri = check(
        regs,
        {"code": None},
        [regs.zone("Ж-1")],
        where="",
        floors=None,
        height=None,
        residential=False,
    )
    assert no_vri["verdict"] == "unclear"


def test_the_use_comes_from_a_code_an_object_type_or_a_service_type():
    po2vri = json.loads(open("data/physical_object_type_to_vri.json").read())
    kwargs = dict(po2vri=po2vri, service_map={"110": {"vri_code": "4.7"}})
    assert (
        resolve_vri(
            vri_code=" 2.1 ",
            physical_object_type_id=None,
            service_type_id=None,
            floors=None,
            **kwargs,
        )["code"]
        == "2.1"
    )
    house = resolve_vri(
        vri_code=None,
        physical_object_type_id=4,
        service_type_id=None,
        floors=6,
        **kwargs,
    )
    assert house["code"] == "2.5" and "по этажности, 6 эт." in house["basis"]
    hotel = resolve_vri(
        vri_code=None,
        physical_object_type_id=None,
        service_type_id=110,
        floors=None,
        **kwargs,
    )
    assert hotel["code"] == "4.7"
    unknown = resolve_vri(
        vri_code=None,
        physical_object_type_id=None,
        service_type_id=999,
        floors=None,
        **kwargs,
    )
    assert unknown["code"] is None


# --- functional zone types -> ПЗЗ zones --------------------------------------


def test_codes_prefixes_and_kinds():
    assert [code_prefix(c) for c in ("Ж-2.15", "ТД.10", "ПП", "ОИ-1")] == [
        "Ж",
        "ТД",
        "ПП",
        "ОИ",
    ]
    assert [base_code(c) for c in ("Ж-1.10.2", "ТД.10", "И")] == ["Ж-1", "ТД", "И"]
    assert zone_kind("residential_lowrise") == "residential"
    assert zone_kind("unknown") is None


def test_the_model_sees_base_zones_and_characteristic_uses():
    bases = base_zones(ZONES)
    assert [b["code"] for b in bases] == ["Ж-1", "Ж-4", "О-1", "Р-1", "И", "Т"]
    assert bases[0]["name"] == "Зона застройки индивидуальными жилыми домами"
    residential = next(t for t in TYPES if t["id"] == 1)
    # 3.1.1 is permitted in most types: it says nothing about a type.
    assert [code for code, _ in residential["vri"]] == ["2.1", "2.6"]
    messages = mapping_messages(TYPES, bases, "ПЗЗ Гатчины")
    assert "- Ж-1 [ЖИЛЫЕ ЗОНЫ] Зона застройки" in messages[1]["content"]
    assert (
        "- 11 (residential_lowrise) «Малоэтажная жилая зона»" in messages[1]["content"]
    )
    assert "(unknown)" not in messages[1]["content"]
    schema = mapping_schema(TYPES, bases)
    item = schema["properties"]["zones"]["items"]["properties"]
    assert item["code"]["enum"] == [b["code"] for b in bases]
    assert item["functional_zone_type_ids"]["items"]["enum"] == [1, 4, 6, 7, 11]


def test_the_answer_of_the_model_covers_subzones_and_drops_unknown_codes():
    answer = {
        "zones": [
            {"code": "Ж-4", "functional_zone_type_ids": [1]},
            {"code": "Ж-1", "functional_zone_type_ids": [11]},
            {"code": "Ж-9", "functional_zone_type_ids": [1]},
            {"code": "О-1", "functional_zone_type_ids": [99, 14]},
        ]
    }
    mapping = parse_mapping(answer, TYPES, ZONES)
    assert mapping.method == METHOD_LLM
    assert mapping.codes(11) == ["Ж-1", "Ж-1.10"]
    # A zone of a residential subtype is a residential zone too.
    assert mapping.codes(1) == ["Ж-1", "Ж-1.10", "Ж-4"]
    assert mapping.codes(7) == [] and mapping.codes(14) == []
    assert 99 not in mapping.by_type
    assert parse_mapping({"zones": []}, TYPES, ZONES) is None


def test_answers_of_the_model_vote():
    def answer(*pairs):
        return parse_mapping(
            {"zones": [{"code": c, "functional_zone_type_ids": [t]} for c, t in pairs]},
            TYPES,
            ZONES,
        )

    voted = vote(
        [
            answer(("О-1", 7), ("Т", 6)),
            answer(("О-1", 7), ("И", 6)),
            answer(("О-1", 7), ("Т", 6), ("И", 4)),
        ],
        ZONES,
    )
    assert voted.method == METHOD_LLM
    assert voted.codes(7) == ["О-1"]
    assert voted.codes(6) == ["Т"]  # И: one vote of three
    assert voted.codes(4) == []
    assert vote([], ZONES) is None


def test_without_the_model_zones_follow_the_code_prefix():
    mapping = prefix_mapping(TYPES, ZONES)
    assert mapping.method == METHOD_CODE_PREFIX
    assert mapping.codes(11) == ["Ж-1", "Ж-1.10", "Ж-4"]
    # Engineering infrastructure belongs to both industrial and transport zones.
    assert mapping.codes(4) == ["И"]
    assert mapping.codes(6) == ["И", "Т"]
    assert mapping.codes(14) == []


class FakeLlm:
    def __init__(self, answer=None, error=None):
        self.answer = answer
        self.error = error
        self.calls = 0

    def __call__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    async def complete_json(
        self, messages, *, schema, temperature=0.0, reasoning_effort="low"
    ):
        self.calls += 1
        self.reasoning_effort = reasoning_effort
        if self.error:
            raise self.error
        return self.answer


@pytest.fixture(autouse=True)
def _fresh_caches():
    caches = (
        pzz_regulations._TERRITORIES,
        pzz_regulations._DOCUMENTS,
        pzz_regulations._ZONES,
        pzz_regulations._MAPPINGS,
        pzz_regulations._FALLBACK_MAPPINGS,
    )
    for cache in caches:
        cache.clear()
    yield
    for cache in caches:
        cache.clear()


def _map(llm, payload=PAYLOAD):
    return asyncio.run(
        zone_type_mapping(payload, types=TYPES, document_label="ПЗЗ", llm_factory=llm)
    )


def test_the_model_maps_an_edition_once():
    llm = FakeLlm({"zones": [{"code": "О-1", "functional_zone_type_ids": [7]}]})
    assert _map(llm).codes(7) == ["О-1"]
    assert _map(llm).method == METHOD_LLM
    assert llm.calls == 3 and llm.reasoning_effort == "medium"  # one mapping, voted
    # A re-parse under the same version label is a new edition.
    reparsed = {**PAYLOAD, "zones": ZONES[:-1]}
    assert edition_key(reparsed) != edition_key(PAYLOAD)
    _map(llm, reparsed)
    assert llm.calls == 6


def test_a_failing_model_falls_back_to_code_prefixes(caplog):
    llm = FakeLlm(error=ChatLlmError(503, "down"))
    with caplog.at_level("WARNING", logger="service.pzz_regulations"):
        mapping = _map(llm)
    assert mapping.method == METHOD_CODE_PREFIX
    assert mapping.codes(7) == ["О-1"]
    assert "mapping_failed" in caplog.text
    _map(llm)
    assert llm.calls == 3  # the fallback is cached for a while
    assert _map(None).method == METHOD_CODE_PREFIX


# --- resolving the ПЗЗ of a territory ----------------------------------------


class FakeUrban:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    async def get_project_territory(self, project_id, *, token=None):
        self.calls.append(("project", project_id))
        if self.fail:
            raise UrbanApiError(403, "forbidden")
        return {"geometry": {"type": "Point", "coordinates": [30.1, 59.5]}}

    async def get_common_territory(self, geometry, *, token=None):
        return {
            "territory_id": 73,
            "name": "Гатчинское городское поселение",
            "parent": {"id": 67, "name": "Гатчинский муниципальный район"},
        }

    async def get_territory(self, territory_id, *, token=None):
        self.calls.append(("territory", territory_id))
        parents = {
            73: {"id": 67, "name": "Гатчинский муниципальный район"},
            67: {"id": 1, "name": "Ленинградская область"},
            1: None,
        }
        names = {
            73: "Гатчинское городское поселение",
            67: "Гатчинский муниципальный район",
            1: "Ленинградская область",
        }
        return {
            "territory_id": territory_id,
            "name": names[territory_id],
            "parent": parents[territory_id],
        }


class FakeNormGraph:
    def __init__(self, documents, fail=False):
        self.documents = documents
        self.fail = fail
        self.asked = []

    async def regulation_documents(self, territory_ids):
        if self.fail:
            raise NormGraphError(503, "down")
        self.asked.append(("documents", territory_ids))
        return self.documents

    async def zones(self, doc_id):
        self.asked.append(("zones", doc_id))
        return [{**z, "document": {"doc_id": doc_id}} for z in ZONES]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None


def _resolve(urban, normgraph, **where):
    where = where or {"project_id": 938}
    return asyncio.run(
        resolve_territory_pzz(urban=urban, normgraph=normgraph, token="user", **where)
    )


def test_the_pzz_of_the_deepest_territory_is_taken():
    district = {**DOCUMENT, "doc_id": "pzz-district", "territory_id": 67, "zones": 40}
    settlement = {**DOCUMENT, "zones": 75}
    normgraph = FakeNormGraph([district, settlement])
    payload = _resolve(FakeUrban(), normgraph)

    assert normgraph.asked[0] == ("documents", [73, 67, 1])
    assert payload["document"]["doc_id"] == "pzz-1"
    assert payload["territory"] == {"id": 73, "name": "Гатчинское городское поселение"}
    assert len(payload["zones"]) == len(ZONES)
    assert "document" not in payload["zones"][0]


def test_a_territory_can_be_given_directly():
    urban = FakeUrban()
    payload = _resolve(
        urban, FakeNormGraph([{**DOCUMENT, "zones": 75}]), territory_id=73
    )
    assert payload["requested_territory"]["id"] == 73
    assert ("project", 938) not in urban.calls


def test_no_pzz_is_reported_and_failures_propagate():
    with pytest.raises(PzzNotFound, match="нет ПЗЗ"):
        _resolve(FakeUrban(), FakeNormGraph([]))
    pzz_regulations._DOCUMENTS.clear()
    with pytest.raises(NormGraphError):
        _resolve(FakeUrban(), FakeNormGraph([], fail=True))
    pzz_regulations._TERRITORIES.clear()
    with pytest.raises(UrbanApiError):
        _resolve(FakeUrban(fail=True), FakeNormGraph([DOCUMENT]))


def test_territories_documents_and_zones_are_cached():
    urban = FakeUrban()
    normgraph = FakeNormGraph([{**DOCUMENT, "zones": 75}])
    _resolve(urban, normgraph)
    _resolve(urban, normgraph)
    assert urban.calls.count(("project", 938)) == 1
    assert normgraph.asked == [("documents", [73, 67, 1]), ("zones", "pzz-1")]


# --- the endpoint ------------------------------------------------------------


class _Settings:
    urban_api_base_url = "http://urban.local"
    urban_api_timeout_seconds = 5
    physical_object_type_to_vri_path = "data/physical_object_type_to_vri.json"
    service_type_to_vri_path = "data/service_type_to_vri.json"
    default_fz_to_pzz_mapping_path = "data/functional_zones_to_pzz_mapping.json"


@pytest.fixture
def api(monkeypatch):
    from service.app import app
    from service.api.security import verify_token
    from service.dependencies import get_app_settings

    urban = FakeUrban()
    normgraph = FakeNormGraph([{**DOCUMENT, "zones": 75}])
    llm = FakeLlm(
        {
            "zones": [
                {"code": "Ж-1", "functional_zone_type_ids": [1, 11]},
                {"code": "Ж-4", "functional_zone_type_ids": [1]},
            ]
        }
    )

    class _Urban:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return urban

        async def __aexit__(self, *exc_info):
            return None

    state = {"normgraph": normgraph}
    monkeypatch.setattr(regulations_api, "UrbanApiClient", _Urban)
    monkeypatch.setattr(
        regulations_api, "build_normgraph_client", lambda s: state["normgraph"]
    )
    monkeypatch.setattr(regulations_api, "build_chat_llm_client", lambda s: llm)
    app.dependency_overrides[verify_token] = lambda: "user-token"
    app.dependency_overrides[get_app_settings] = lambda: _Settings()
    try:
        yield TestClient(app), state, llm
    finally:
        app.dependency_overrides.clear()


def test_a_zone_type_is_checked_against_its_pzz_zones(api):
    client, _, llm = api
    resp = client.post(
        "/pzz/regulations/check-vri",
        json={
            "project_id": 938,
            "physical_object_type_id": 4,
            "floors": 2,
            "functional_zone_type_id": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["document"]["label"].startswith("Правила землепользования")
    assert body["vri"]["code"] == "2.1.1"
    assert body["zone"]["mapping"] == {
        "method": METHOD_LLM,
        "zone_codes": ["Ж-1", "Ж-1.10", "Ж-4"],
    }
    # 2.1.1 falls under 2.1 of Ж-1, not under 2.6 of Ж-4.
    assert body["verdict"] == DEPENDS_ON_ZONE
    assert [z["verdict"] for z in body["zones"]] == [
        "allowed_main",
        "allowed_main",
        "not_allowed",
    ]
    assert llm.calls == 3


def test_an_exact_zone_checks_the_use_and_its_limits(api):
    client, _, _ = api
    resp = client.post(
        "/pzz/regulations/check-vri",
        json={
            "territory_id": 73,
            "vri_code": "2.1",
            "floors": 5,
            "pzz_zone_code": "ж-1",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["zone"] == {"pzz_zone_code": "Ж-1"}
    assert body["verdict"] == "allowed_main"
    assert body["vri"]["name"] == "вид Ж-1"
    assert body["zones"][0]["parameters"]["status"] == PARAMS_EXCEEDED


def test_bad_requests_and_missing_pzz(api):
    client, state, _ = api
    url = "/pzz/regulations/check-vri"
    both = {
        "project_id": 1,
        "territory_id": 2,
        "vri_code": "2.1",
        "pzz_zone_code": "Ж-1",
    }
    assert client.post(url, json=both).status_code == 422
    unknown_type = {"project_id": 938, "vri_code": "2.1", "functional_zone_type_id": 99}
    assert client.post(url, json=unknown_type).status_code == 422
    unknown_zone = {"project_id": 938, "vri_code": "2.1", "pzz_zone_code": "Х-9"}
    resp = client.post(url, json=unknown_zone)
    assert resp.status_code == 404 and "Ж-1" in resp.json()["detail"]

    state["normgraph"] = FakeNormGraph([])
    pzz_regulations._DOCUMENTS.clear()
    ok = {"project_id": 938, "vri_code": "2.1", "pzz_zone_code": "Ж-1"}
    resp = client.post(url, json=ok)
    assert resp.status_code == 404 and "нет ПЗЗ" in resp.json()["detail"]

    state["normgraph"] = FakeNormGraph([], fail=True)
    pzz_regulations._DOCUMENTS.clear()
    resp = client.post(url, json=ok)
    assert resp.status_code == 502 and "down" not in resp.text

    state["normgraph"] = None
    assert client.post(url, json=ok).status_code == 503


# --- the NormGraph client ----------------------------------------------------


class FakeTokenClient:
    async def get_authorization_headers(self, *, force_refresh=False):
        return {"Authorization": "Bearer service-tok"}


def test_normgraph_client_sends_the_service_token_and_territories():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            (
                request.url.path,
                request.url.query.decode(),
                request.headers["Authorization"],
            )
        )
        if request.url.path == "/regulations/documents":
            return httpx.Response(200, json=[DOCUMENT])
        return httpx.Response(200, json={"count": 1, "zones": ZONES[:1]})

    async def go():
        client = NormGraphClient("http://ng.local", FakeTokenClient())
        client._client = httpx.AsyncClient(
            base_url="http://ng.local", transport=httpx.MockTransport(handler)
        )
        async with client:
            documents = await client.regulation_documents([73, 67])
            zones = await client.zones("pzz-1")
        return documents, zones

    documents, zones = asyncio.run(go())
    assert documents == [DOCUMENT]
    assert zones == ZONES[:1]
    assert seen[0] == (
        "/regulations/documents",
        "territory_id=73&territory_id=67",
        "Bearer service-tok",
    )
    assert seen[1][1] == "doc_id=pzz-1&limit=2000"


def test_normgraph_client_raises_on_errors():
    async def go():
        client = NormGraphClient("http://ng.local", FakeTokenClient())
        client._client = httpx.AsyncClient(
            base_url="http://ng.local",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(401, json={"detail": "no"})
            ),
        )
        async with client:
            await client.regulation_documents([73])

    with pytest.raises(NormGraphError) as caught:
        asyncio.run(go())
    assert caught.value.status == 401


# --- the MCP tool ------------------------------------------------------------


def test_the_mcp_tool_sends_only_the_given_fields():
    from service.mcp_server.tools.regulations import check_vri_in_pzz

    sent = {}

    class _Api:
        async def check_vri(self, *, body, token=None):
            sent.update(body=body, token=token)
            return {"verdict": "allowed_main"}

    fn = getattr(check_vri_in_pzz, "fn", check_vri_in_pzz)
    result = asyncio.run(
        fn(project_id=938, vri_code="2.1", pzz_zone_code="Ж-1", token="t", api=_Api())
    )
    assert result == {"verdict": "allowed_main"}
    assert sent == {
        "body": {"project_id": 938, "vri_code": "2.1", "pzz_zone_code": "Ж-1"},
        "token": "t",
    }
