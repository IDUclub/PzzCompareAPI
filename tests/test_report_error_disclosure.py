"""Report fields must not carry backend exception text.

A failed LLM call once put ``404 Client Error: Not Found for url:
http://<gpu-host>:8001/v1/chat/completions`` into the delivered report, and the
in-memory dedup fanned that one failure onto 2779 parcels. The verdict fallback
to manual review is correct; the wording is what leaked internal hosts, token
counts and URLs to the customer.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "pipeline_modules/business/pipeline_impl.py"
REPORT_FIELDS = {"reason", "PZZ_REASON", "Причина", "Вердикт_ПЗЗ", "Статус"}


def exception_names(tree: ast.AST) -> set[str]:
    return {
        handler.name
        for handler in ast.walk(tree)
        if isinstance(handler, ast.ExceptHandler) and handler.name
    }


def referenced_names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


@pytest.fixture(scope="module")
def source_tree() -> ast.AST:
    return ast.parse(MODULE.read_text(encoding="utf-8"))


def test_report_fields_never_interpolate_an_exception(source_tree) -> None:
    caught = exception_names(source_tree)
    assert caught, "expected the module to name caught exceptions"

    offenders = []
    for node in ast.walk(source_tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or key.value not in REPORT_FIELDS:
                continue
            if referenced_names(value) & caught:
                offenders.append(f"{key.value} at line {value.lineno}")

    assert not offenders, f"exception text reaches the report: {offenders}"


def test_failure_wording_sends_the_parcel_to_manual_review() -> None:
    from pipeline_modules.business import pipeline_impl

    for reason in (
        pipeline_impl.LLM_CHECK_FAILED_REASON,
        pipeline_impl.CLASSIFICATION_FAILED_REASON,
    ):
        assert "ручную проверку" in reason
        assert "http" not in reason
