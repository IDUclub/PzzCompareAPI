"""The heritage-case deep-reasoning trigger and its kill switch."""

import importlib

import pytest

from pipeline_modules.business import profiled_fast_match_layer, runtime_settings

HERITAGE_ZONE = {
    "zone_name": "Зона исторической застройки",
    "retrieval_text": "дворцово-парковый ансамбль, музей-заповедник",
}
PLAIN_ZONE = {"zone_name": "Зона застройки индивидуальными жилыми домами"}

HERITAGE_VRI = "объекты культурного наследия усадебного комплекса XVIII века"
PLAIN_VRI = "для индивидуального жилищного строительства"


def trigger(vri, zone):
    return profiled_fast_match_layer.should_use_deeper_llm_reasoning(vri, zone)


def test_heritage_wording_in_heritage_zone_goes_deep():
    assert trigger(HERITAGE_VRI, HERITAGE_ZONE) is True


def test_plain_wording_never_goes_deep():
    assert trigger(PLAIN_VRI, HERITAGE_ZONE) is False
    assert trigger(PLAIN_VRI, PLAIN_ZONE) is False


def test_heritage_wording_alone_is_not_enough():
    assert trigger("историческая застройка", PLAIN_ZONE) is False


def test_long_heritage_wording_goes_deep_outside_a_heritage_zone():
    long_vri = (
        "историческое здание с пристройками, хозяйственными постройками "
        "и прилегающей территорией"
    )
    assert len(long_vri) >= 70
    assert trigger(long_vri, PLAIN_ZONE) is True


def test_empty_wording_is_not_deep():
    assert trigger("", HERITAGE_ZONE) is False
    assert trigger(None, HERITAGE_ZONE) is False


def test_kill_switch_disables_every_deep_case(monkeypatch):
    monkeypatch.setattr(
        profiled_fast_match_layer, "LLM_DEEP_REASONING_ENABLED", False
    )
    assert trigger(HERITAGE_VRI, HERITAGE_ZONE) is False


def test_kill_switch_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_DEEP_REASONING_ENABLED", "false")
    try:
        assert importlib.reload(runtime_settings).LLM_DEEP_REASONING_ENABLED is False
    finally:
        monkeypatch.undo()
        importlib.reload(runtime_settings)


@pytest.mark.parametrize(
    "vri",
    [
        'земельный участок музея-заповедника "Гатчина"',
        "мемориальный комплекс воинского захоронения",
        "историко-художественный дворцово-парковый ансамбль",
    ],
)
def test_known_heritage_wordings_go_deep(vri):
    assert trigger(vri, HERITAGE_ZONE) is True
