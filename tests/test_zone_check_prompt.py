"""build_zone_check_prompt keeps its content when the zone block is moved first."""

from types import SimpleNamespace

import pytest

from pipeline_modules.business import matching_layer

ZONE = {
    "zone_code": "Ж-1",
    "base_zone_code": "Ж",
    "zone_heading": "Ж-1 ЗОНА ЗАСТРОЙКИ ИНДИВИДУАЛЬНЫМИ ЖИЛЫМИ ДОМАМИ",
    "zone_name": "ЗОНА ЗАСТРОЙКИ ИНДИВИДУАЛЬНЫМИ ЖИЛЫМИ ДОМАМИ",
    "zone_summary": "Основные ВРИ: для индивидуального жилищного строительства.",
    "retrieval_text": "Наименование зоны: ЗОНА ЗАСТРОЙКИ. Основные виды: 2.1 Для ИЖС.",
}
VRI_LINE = "Кадастровый ВРИ: "
ZONE_LINE = "Код фактической зоны ПЗЗ: "


def _common_prefix(first: str, second: str) -> str:
    for index, (left, right) in enumerate(zip(first, second)):
        if left != right:
            return first[:index]
    return first[: min(len(first), len(second))]


def _prompt(vri_text="размещение гаражей для собственных нужд", exact_matches=None):
    return matching_layer.build_zone_check_prompt(
        vri_text=vri_text,
        zone_ref={"zone_code": ZONE["zone_code"], "zone_name": ZONE["zone_name"]},
        exact_matches=exact_matches or [],
        actual_zone_code=ZONE["zone_code"],
        actual_zone_name=ZONE["zone_name"],
        actual_share=None,
        intersect_codes=None,
        context=SimpleNamespace(raw_zone_lookup={ZONE["zone_code"]: ZONE}),
    )


@pytest.fixture
def zone_first(monkeypatch):
    monkeypatch.setattr(matching_layer, "ZONE_CHECK_PROMPT_ZONE_FIRST", True)


def test_parcel_fields_lead_by_default(monkeypatch):
    monkeypatch.setattr(matching_layer, "ZONE_CHECK_PROMPT_ZONE_FIRST", False)

    assert _prompt().startswith(VRI_LINE)


def test_zone_block_leads_when_enabled(zone_first):
    prompt = _prompt()

    assert prompt.startswith(ZONE_LINE)
    assert prompt.index(VRI_LINE) > prompt.index(ZONE["retrieval_text"])


def test_reordering_preserves_every_word(monkeypatch):
    monkeypatch.setattr(matching_layer, "ZONE_CHECK_PROMPT_ZONE_FIRST", False)
    matches = [{"section_name": "main", "matched_vri_code": "2.1", "matched_vri_name": "ИЖС"}]
    default = _prompt(exact_matches=matches)

    monkeypatch.setattr(matching_layer, "ZONE_CHECK_PROMPT_ZONE_FIRST", True)
    reordered = _prompt(exact_matches=matches)

    assert reordered != default
    assert sorted(reordered.split()) == sorted(default.split())


def test_parcels_of_one_zone_share_a_long_prefix(zone_first):
    first = _prompt(vri_text="размещение гаражей для собственных нужд")
    second = _prompt(vri_text="для индивидуального жилищного строительства")

    shared = len(_common_prefix(first, second))
    assert shared > 0.9 * min(len(first), len(second))
