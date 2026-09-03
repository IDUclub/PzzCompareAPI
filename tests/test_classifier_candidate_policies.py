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
        entry["code"]: entry
        for entry in payload["entries"]
        if entry.get("code")
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


def test_negative_rules_remove_impossible_candidates(policy_context: SimpleNamespace) -> None:
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


@pytest.mark.parametrize(
    ("vri_text", "required_codes"),
    [
        ("Для рекпеационных целей", {"5.0"}),
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
