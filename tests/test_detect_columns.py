"""Tests for LLM-assisted column detection (auto detect+run flow)."""
import asyncio
import json

import httpx
import pytest

from service.application.use_cases.detect_columns import (
    BUILDING_TARGETS,
    CADASTRAL_TARGETS,
    PZZ_ZONE_TARGETS,
    VRI_TARGET,
    _heuristic_match,
    detect_columns_for_file,
    profile_columns,
    render_detection_narrative,
    required_columns_resolved,
)
from service.infrastructure.ollama_chat_client import OllamaChatClient, OllamaChatError


def _fc(rows: list[dict]) -> dict:
    """Wrap property rows as a GeoJSON FeatureCollection (with dummy geometry)."""
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]},
             "properties": row}
            for row in rows
        ],
    }


class RecordingFakeOllama:
    """Fake client: returns a scripted complete_json result and records calls."""

    def __init__(self, result: dict | None = None, error: bool = False):
        self._result = result or {}
        self._error = error
        self.calls: list[dict] = []

    async def complete_json(self, messages, *, schema, model=None, temperature=0.0):
        self.calls.append({"messages": messages, "schema": schema, "model": model})
        if self._error:
            raise OllamaChatError(500, "boom")
        return self._result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


# --- profiling -------------------------------------------------------------

def test_profile_columns_excludes_geometry_and_dedups_samples() -> None:
    fc = _fc([
        {"vri_name": "Для ИЖС", "index": "Ж-1"},
        {"vri_name": "Для ИЖС", "index": "О-2"},
        {"vri_name": "Многоэтажная жилая застройка", "index": "Ж-1"},
    ])
    profiles = {p.name: p for p in profile_columns(fc)}
    assert set(profiles) == {"vri_name", "index"}  # no "geometry"
    assert profiles["vri_name"].dtype == "str"
    assert profiles["vri_name"].n_unique == 2
    # distinct samples, order preserved, no duplicate "Для ИЖС"
    assert profiles["vri_name"].samples == ["Для ИЖС", "Многоэтажная жилая застройка"]


def test_profile_columns_truncates_long_values() -> None:
    long = "x" * 200
    profiles = {p.name: p for p in profile_columns(_fc([{"c": long}]), max_value_chars=10)}
    assert profiles["c"].samples == ["x" * 10 + "…"]


# --- heuristics ------------------------------------------------------------

def test_heuristic_ignores_code_companion_decoy() -> None:
    # Regression: when the real VRI column is named non-standardly, the numeric
    # "Код_<known>" companion must NOT be picked by the heuristic — it should
    # return None and defer to the LLM (else it silently classifies on codes).
    fc = _fc([
        {"permitted_use": "Для ИЖС", "Код_Вид_разрешенного_исп": "13.2"},
        {"permitted_use": "Ведение садоводства", "Код_Вид_разрешенного_исп": "12.0"},
    ])
    assert _heuristic_match(VRI_TARGET, profile_columns(fc)) is None


def test_heuristic_resolves_known_defaults_without_llm() -> None:
    fc = _fc([{"Индекс_зоны": "Ж-1", "Код_объекта": "Зона Ж-1"}])
    fake = RecordingFakeOllama()
    suggestions = asyncio.run(
        detect_columns_for_file(fake, fc, PZZ_ZONE_TARGETS)
    )
    assert suggestions["pzz_zone_code_col"].value == "Индекс_зоны"
    assert suggestions["pzz_zone_code_col"].source == "heuristic"
    assert suggestions["pzz_zone_name_col"].value == "Код_объекта"
    assert fake.calls == []  # heuristic fully resolved -> no LLM call


def test_building_columns_resolve_by_heuristic() -> None:
    # Urban-API-shaped buildings: type/service/floors resolve without an LLM call.
    fc = _fc([{"physical_object_type_id": 4, "service_type_id": 22, "floors_count": 5}])
    fake = RecordingFakeOllama()
    suggestions = asyncio.run(detect_columns_for_file(fake, fc, BUILDING_TARGETS))
    assert suggestions["building_type_col"].value == "physical_object_type_id"
    assert suggestions["building_service_col"].value == "service_type_id"
    assert suggestions["building_floors_col"].value == "floors_count"
    assert fake.calls == []


def test_profile_columns_discovers_late_appearing_column() -> None:
    # Regression: a merged buildings+services layer lists all physical objects
    # first and services only later. The service column must still be discovered
    # (profiling scans every feature, not just the first window).
    feats = [{"physical_object_type_id": 45} for _ in range(60)]
    feats.append({"service_type_id": 87})  # first service well past the old 50 cap
    profiles = {p.name: p for p in profile_columns(_fc(feats))}
    assert "service_type_id" in profiles
    assert profiles["service_type_id"].samples == ["87"]


def test_building_service_column_resolved_when_only_in_late_features() -> None:
    # End-to-end of the fix: detection maps building_service_col even though
    # service_type_id appears only after 1500 physical-object features.
    feats = [{"physical_object_type_id": 4} for _ in range(1500)]
    feats += [{"service_type_id": 22} for _ in range(20)]
    fake = RecordingFakeOllama()
    suggestions = asyncio.run(detect_columns_for_file(fake, _fc(feats), BUILDING_TARGETS))
    # type + service resolve by heuristic (floors is absent -> defers to LLM)
    assert suggestions["building_type_col"].value == "physical_object_type_id"
    assert suggestions["building_service_col"].value == "service_type_id"


def test_building_text_name_columns_resolve_by_heuristic() -> None:
    fc = _fc([{
        "physical_object_type_name": "Жилой дом",
        "service_type_name": "Школа",
        "floors": 5,
    }])
    fake = RecordingFakeOllama()
    suggestions = asyncio.run(detect_columns_for_file(fake, fc, BUILDING_TARGETS))
    assert suggestions["building_type_col"].value == "physical_object_type_name"
    assert suggestions["building_service_col"].value == "service_type_name"
    assert suggestions["building_floors_col"].value == "floors"
    assert fake.calls == []


# --- LLM fallback + enum constraint ---------------------------------------

def test_llm_resolves_unknown_column_names() -> None:
    fc = _fc([{"permitted_use": "Для ИЖС", "other": "x"}])
    fake = RecordingFakeOllama(
        {"cadastral_vri_col": {"column": "permitted_use", "reason": "текст ВРИ"}}
    )
    suggestions = asyncio.run(detect_columns_for_file(fake, fc, CADASTRAL_TARGETS))
    assert suggestions["cadastral_vri_col"].value == "permitted_use"
    assert suggestions["cadastral_vri_col"].source == "llm"
    assert len(fake.calls) == 1
    # schema enum is constrained to the real column names (+ null)
    schema = fake.calls[0]["schema"]
    enum = schema["properties"]["cadastral_vri_col"]["properties"]["column"]["enum"]
    assert set(enum) == {"permitted_use", "other", None}


def test_llm_hallucinated_column_is_rejected() -> None:
    fc = _fc([{"permitted_use": "Для ИЖС"}])
    fake = RecordingFakeOllama(
        {"cadastral_vri_col": {"column": "does_not_exist", "reason": "…"}}
    )
    suggestions = asyncio.run(detect_columns_for_file(fake, fc, CADASTRAL_TARGETS))
    assert suggestions["cadastral_vri_col"].value is None
    assert suggestions["cadastral_vri_col"].source == "none"


def test_llm_error_degrades_to_none() -> None:
    fc = _fc([{"permitted_use": "Для ИЖС"}])
    fake = RecordingFakeOllama(error=True)
    suggestions = asyncio.run(detect_columns_for_file(fake, fc, CADASTRAL_TARGETS))
    assert suggestions["cadastral_vri_col"].source == "none"


# --- narrative + required check -------------------------------------------

def test_narrative_lists_recognised_and_missing() -> None:
    fc = _fc([{"permitted_use": "Для ИЖС"}])
    fake = RecordingFakeOllama(
        {"cadastral_vri_col": {"column": "permitted_use", "reason": "…"}}
    )
    suggestions = asyncio.run(detect_columns_for_file(fake, fc, CADASTRAL_TARGETS))
    text = render_detection_narrative(suggestions, CADASTRAL_TARGETS)
    assert "permitted_use" in text
    assert "вид разрешённого использования (ВРИ) участка" in text
    assert "определено как" in text
    assert required_columns_resolved(suggestions, CADASTRAL_TARGETS) is True


def test_required_not_resolved_when_missing() -> None:
    fc = _fc([{"unrelated": "x"}])
    fake = RecordingFakeOllama({"cadastral_vri_col": {"column": None, "reason": "нет"}})
    suggestions = asyncio.run(detect_columns_for_file(fake, fc, CADASTRAL_TARGETS))
    assert required_columns_resolved(suggestions, CADASTRAL_TARGETS) is False
    text = render_detection_narrative(suggestions, [VRI_TARGET])
    assert "Не удалось определить" in text


# --- complete_json client (structured output) -----------------------------

def _real_client(handler) -> OllamaChatClient:
    client = OllamaChatClient(base_url="http://llm.local", default_model="m", timeout_seconds=5)
    client._client = httpx.AsyncClient(
        base_url="http://llm.local", transport=httpx.MockTransport(handler)
    )
    return client


def test_complete_json_sends_format_and_parses_content() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"message": {"content": json.dumps({"cadastral_vri_col": {"column": "a"}})}}
        )

    async def run():
        async with _real_client(handler) as oc:
            return await oc.complete_json(
                [{"role": "user", "content": "x"}], schema={"type": "object"}
            )

    result = asyncio.run(run())
    assert result == {"cadastral_vri_col": {"column": "a"}}
    assert seen["body"]["stream"] is False
    assert seen["body"]["format"] == {"type": "object"}


def test_complete_json_raises_on_bad_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "not json"}})

    async def run():
        async with _real_client(handler) as oc:
            await oc.complete_json([{"role": "user", "content": "x"}], schema={})

    with pytest.raises(OllamaChatError):
        asyncio.run(run())


def test_complete_json_recovers_json_from_reasoning_wrapper() -> None:
    # gpt-oss-style: chain-of-thought + fenced JSON around the actual object.
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        content = 'Let me think...\nThe answer is:\n```json\n{"n0": 26, "n1": null}\n```\n'
        return httpx.Response(200, json={"message": {"content": content}})

    async def run():
        async with _real_client(handler) as oc:
            return await oc.complete_json([{"role": "user", "content": "x"}], schema={})

    result = asyncio.run(run())
    assert result == {"n0": 26, "n1": None}
    assert seen["body"]["think"] is False  # reasoning suppressed at the source
