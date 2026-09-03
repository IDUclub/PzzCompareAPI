import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline_modules.business.profiled_fast_match_layer import (
    canonicalize_vri_name,
    classify_object_profile,
)
from pipeline_modules.business.rerank_layer import (
    build_not_allowed_embed_query_text,
    classifier_requires_manual_review,
    enforce_classifier_candidate_policies,
)


@pytest.fixture(scope="module")
def policy_context() -> SimpleNamespace:
    classifier_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "rosreestr_vri_classifier_2024_12_24.json"
    )
    payload = json.loads(classifier_path.read_text(encoding="utf-8"))
    classifier_by_code = {
        entry["code"]: entry for entry in payload["entries"] if entry.get("code")
    }
    return SimpleNamespace(rosreestr_classifier_by_code=classifier_by_code)


def _candidates(*codes: str) -> list[dict]:
    return [
        {"code": code, "name": f"candidate {code}", "score": 1.0 - index / 10}
        for index, code in enumerate(codes)
    ]


@pytest.mark.parametrize(
    ("vri_text", "expected_code"),
    [
        ("Для размещения здания ТЭЦ", "6.7"),
        ("Пожарная часть, депо и пожарный гараж", "8.3"),
        ("Для размещения бассейна", "5.1.2"),
        ("Физкультурно-оздоровительный комплекс (ФОК)", "5.1.2"),
        ("Крытый каток с искусственным льдом", "5.1.2"),
        ("Паромные причалы и сооружения переправы", "7.3"),
        ("Многоярусная автостоянка", "4.9.2"),
        ("Парковка легковых автомобилей", "4.9.2"),
        ("рекреационная зона", "5.0"),
        ("для рекпеационных целей", "5.0"),
        (
            "Для обеспечения транспортной безопасности и реализации "
            "антитеррористических мероприятий",
            "7.0",
        ),
        ("Для размещения контейнера", "6.9"),
        ("Для размещения временных 20-ти футовых контейнеров", "6.9"),
        ("для сельхозиспользования (сенокос)", "1.19"),
        ('для размещения здания ЗРУ "Ливадных"', "3.1.1"),
        ("Под строительство подпорной стенки", "12.0.2"),
        ('для размещения здания кинодосугового центра "Россия"', "3.6.1"),
        ("для размещения здания - узел связи (РУС)", "6.8"),
        ("Для размещения здания лечебного корпуса", "3.4.2"),
        ("Для цветочного торгового павильона", "4.4"),
        ('Для размещения киоска "Пресса"', "4.4"),
        ("Для размещения здания Холмского городского суда", "3.8.1"),
        ("Для размещения мастерской по ремонту обуви", "3.3"),
        ("Для размещения памятника", "9.3"),
        ("Для размещения здания управления ЗЭС", "3.1.2"),
        ("для размещения хозяственной площадки", "12.0.2"),
        ("гостовой дом (общественно-деловых цели)", "4.7"),
        ("под производственную территорию", "6.0"),
        ("для размещения складских и производственных объектов", "6.0"),
        ("под существующее здание аптеки", "3.4"),
        ("под производственную базу", "6.0"),
        ("для размещения огородного земельного участка", "13.1"),
        ("Для размещения здания инфекционного отделения", "3.4.2"),
        ("Для размещения некапитальных объектов физической культуры и спора", "5.1"),
        ("для размещения Шашлычной", "4.6"),
        ("Под строительство стрелкового тира", "5.1"),
        (
            "Для размещения объектов некапитального строительства-база зимнего отдыха",
            "5.0",
        ),
        ("Для благоустройства территории, прилегающей к зданию магазина", "12.0.2"),
        ("Для строительства внутримикрорайонного проезда общего пользования", "12.0.1"),
        ("Под внутримикрорайонным проездом общего пользования", "12.0.1"),
        ("Капитальный ремонт пер. Мирного в г. Корсакове", "12.0.1"),
        (
            "под строительство 24 боксов гаражей для индивидуального автотранспорта",
            "2.7.2",
        ),
        ("Под общественную застройку", "3.0"),
        ("Под существующим зданием столовой", "4.6"),
        ("Пол нежилым зданием-кафетерием пристроенным к жилому дому", "4.6"),
        (
            "Для строительства крытой спортивной площадки "
            "(атлетический павильон) для гимнастических упражнений",
            "5.1.2",
        ),
    ],
)
def test_strong_marker_forces_classifier_top1(
    policy_context: SimpleNamespace,
    vri_text: str,
    expected_code: str,
) -> None:
    result = enforce_classifier_candidate_policies(
        vri_text,
        _candidates("3.1", "2.1", "5.4", "2.7.2", "7.5"),
        policy_context,
    )

    assert result[0]["code"] == expected_code
    assert result[0]["name"]


def test_negative_rules_remove_impossible_candidates(
    policy_context: SimpleNamespace,
) -> None:
    parking = enforce_classifier_candidate_policies(
        "Многоярусная автостоянка",
        _candidates("2.1", "2.7.2", "4.9.2"),
        policy_context,
    )
    fire = enforce_classifier_candidate_policies(
        "Пожарная часть с пожарным гаражом",
        _candidates("2.7.2", "4.9", "8.3"),
        policy_context,
    )
    admin = enforce_classifier_candidate_policies(
        "Административное здание районной администрации",
        _candidates("3.1.2", "3.8"),
        policy_context,
    )
    utility_admin = enforce_classifier_candidate_policies(
        "Административное здание управления коммунального водоканала",
        _candidates("3.1.2", "3.8"),
        policy_context,
    )

    assert "2.1" not in {item["code"] for item in parking}
    assert "2.7.2" not in {item["code"] for item in fire}
    assert "3.1.2" not in {item["code"] for item in admin}
    assert "3.1.2" in {item["code"] for item in utility_admin}


def test_fire_and_tec_shortlists_exclude_unrelated_codes(
    policy_context: SimpleNamespace,
) -> None:
    fire = enforce_classifier_candidate_policies(
        "Пожарная часть, пожарное депо и гараж",
        _candidates("2.7.2", "3.1", "3.1.1", "2.1", "2.5", "8.3", "8.0"),
        policy_context,
    )
    tec = enforce_classifier_candidate_policies(
        "Для размещения здания ТЭЦ",
        _candidates("3.1", "6.7.1", "3.2.3", "3.8", "6.7", "7.5", "6.0"),
        policy_context,
    )

    assert [item["code"] for item in fire] == ["8.3", "8.0"]
    assert [item["code"] for item in tec] == ["6.7", "7.5", "6.0"]


def test_transport_safety_is_not_sport(policy_context: SimpleNamespace) -> None:
    text = (
        "Для обеспечения транспортной безопасности и реализации "
        "антитеррористических мероприятий"
    )
    result = enforce_classifier_candidate_policies(
        text,
        _candidates("5.1.2", "5.1", "7.2.1", "7.0", "8.0"),
        policy_context,
    )
    profile = classify_object_profile(text)

    assert result[0]["code"] == "7.0"
    assert not any(item["code"].startswith("5.") for item in result)
    assert profile["family"] == "transport"


def test_waste_container_is_not_forced_to_warehouse(
    policy_context: SimpleNamespace,
) -> None:
    candidates = _candidates("3.1.1", "6.9")

    result = enforce_classifier_candidate_policies(
        "Контейнерная площадка для накопления ТКО",
        candidates,
        policy_context,
    )

    assert result == candidates


def test_garage_made_from_container_remains_a_garage(
    policy_context: SimpleNamespace,
) -> None:
    result = enforce_classifier_candidate_policies(
        "Для размещения временного гаража (металлического контейнера)",
        _candidates("6.9", "2.7.2"),
        policy_context,
    )

    assert result[0]["code"] == "2.7.2"


@pytest.mark.parametrize(
    "vri_text",
    [
        "Под строительство служебно-технического здания",
        "Под выстроенное административное здание",
        "под административное здание",
        "Под административное здание со встроенным магазином",
        "Для размещения нежилого здания",
        "для размещения объектов жилого и общественно-делового назначения",
        "Под зданием (кадастровый № 65:04:0000040:1661)",
        "Под зданием административного назначения",
        "Под пристроенным зданием казино-бара и офиса к жилому дому",
    ],
)
def test_underdetermined_vri_requires_manual_review(
    policy_context: SimpleNamespace,
    vri_text: str,
) -> None:
    assert classifier_requires_manual_review(vri_text)
    assert (
        enforce_classifier_candidate_policies(
            vri_text,
            _candidates("3.1.2", "4.5", "2.0"),
            policy_context,
        )
        == []
    )


def test_identified_public_admin_is_not_sent_to_manual_review() -> None:
    assert not classifier_requires_manual_review(
        "Административное здание районной администрации"
    )


@pytest.mark.parametrize(
    ("vri_text", "required_codes"),
    [
        ("Территория памятника истории", {"9.3", "12.0.2"}),
    ],
)
def test_policy_injects_missing_shortlist_candidates(
    policy_context: SimpleNamespace,
    vri_text: str,
    required_codes: set[str],
) -> None:
    result = enforce_classifier_candidate_policies(
        vri_text,
        _candidates("10.10", "3.6.1", "4.6", "7.5", "2.1"),
        policy_context,
    )[:5]

    assert required_codes <= {item["code"] for item in result}


def test_explicit_classifier_code_has_priority(policy_context: SimpleNamespace) -> None:
    explicit = {
        "code": "3.1",
        "name": "Коммунальное обслуживание",
        "score": 1.0,
        "explicit_code": True,
    }

    assert enforce_classifier_candidate_policies(
        "ТЭЦ, код 3.1",
        [explicit],
        policy_context,
    ) == [explicit]


def test_tec_expansion_and_profile_are_energy() -> None:
    canonical = canonicalize_vri_name("Для размещения ТЭЦ")
    query = build_not_allowed_embed_query_text("Для размещения ТЭЦ")
    profile = classify_object_profile("Для размещения ТЭЦ")

    assert "теплоэлектроцентрал" in canonical
    assert "энергетик" in canonical
    assert "энергетик" in query
    assert "коммунальное обслуживание" not in query
    assert profile["family"] == "energy"
