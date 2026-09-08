"""The zone-check prompt must resolve a VRI that enumerates several uses.

With ``reasoning_effort=low`` the model latched onto the first enumerated use
that fitted the zone and answered allowed_main: a parcel whose VRI read
"многоквартирные жилые дома, ..., бульвары, парки, скверы, ..." in a public
green zone came back as permitted because of the parks. Every listed use has to
be checked, and partial coverage is a manual review, not a permission.
"""

from __future__ import annotations

import re

import pytest

from pipeline_modules.business.matching_layer import ZONE_CHECK_SYSTEM_PROMPT

STEP = re.compile(r"^(\d+)\. ", re.MULTILINE)


def section(heading: str, next_heading: str) -> str:
    start = ZONE_CHECK_SYSTEM_PROMPT.index(heading)
    return ZONE_CHECK_SYSTEM_PROMPT[start : ZONE_CHECK_SYSTEM_PROMPT.index(next_heading, start)]


@pytest.fixture(scope="module")
def workflow() -> str:
    return section("Рабочий порядок:", "Строгие правила:")


def test_workflow_steps_are_numbered_without_gaps(workflow) -> None:
    numbers = [int(match) for match in STEP.findall(workflow)]

    assert numbers == list(range(len(numbers))), numbers


def test_every_step_reference_points_at_a_real_step(workflow) -> None:
    steps = {int(match) for match in STEP.findall(workflow)}
    referenced = {int(n) for n in re.findall(r"см\. пункт (\d+)", ZONE_CHECK_SYSTEM_PROMPT)}

    assert referenced, "expected the prompt to cross-reference its own steps"
    assert referenced <= steps, f"dangling references: {sorted(referenced - steps)}"


def test_enumerated_uses_must_all_be_checked(workflow) -> None:
    assert "разбирай КАЖДЫЙ отдельно" in workflow
    assert "не останавливайся на первом" in workflow


def test_descriptive_commas_are_not_an_enumeration(workflow) -> None:
    assert "Описательные уточнения одного объекта перечислением не считаются" in workflow


@pytest.mark.parametrize(
    "coverage, verdict",
    [
        ("основание есть у всех видов", "allowed_*"),
        ("основание есть у части видов", "unclear"),
        ("основания нет ни у одного вида", "not_allowed"),
    ],
)
def test_aggregation_covers_every_coverage_case(workflow, coverage, verdict) -> None:
    line = next(row for row in workflow.splitlines() if coverage in row)

    assert verdict in line


def test_partial_coverage_names_the_uncovered_uses() -> None:
    requirements = section("Требования к ответу:", "Разрешенные verdict:")

    assert "перечисли в reason виды" in requirements


def test_one_matching_use_does_not_permit_the_rest(workflow) -> None:
    assert "не разрешает остальные" in workflow
