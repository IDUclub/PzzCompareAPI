"""Scenario placement: access, cache isolation, object enrichment and worker routing."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from service.api import scenarios as mod
from service.api.tasks import _build_chat_message_placement
from service.application.use_cases.start_task import start_task
from service.domain import TaskStatus
from service.infrastructure.urban_api_client import UrbanApiError


def feature(**props):
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "Point", "coordinates": [30.12, 59.57]},
    }


@pytest.fixture
def flow(monkeypatch):
    state = SimpleNamespace(
        cached=None,
        access_error=None,
        sources=[{"year": 2025, "source": "User"}],
        physical_filters=[],
        calls=[],
        captured=[],
    )
    settings = SimpleNamespace(
        urban_api_base_url="http://urban.test",
        urban_api_timeout_seconds=1,
        physical_object_type_to_vri_path="data/physical_object_type_to_vri.json",
        service_type_to_vri_path="data/service_type_to_vri.json",
        default_vri_classifier_path="data/rosreestr_vri_classifier_2024_12_24.json",
        vectorizer_url="",
        building_semantic_fallback=False,
    )

    class Urban:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get_scenario_info(self, sid, *, token):
            state.calls.append(("access", sid, token))
            if state.access_error:
                raise state.access_error
            return {"updated_at": "2026-10-01", "project_id": 7}

        async def list_functional_zone_sources(self, sid, *, token):
            return state.sources

        async def get_functional_zones(self, sid, **kwargs):
            return {
                "type": "FeatureCollection",
                "features": [feature(functional_zone_type={"id": 1, "name": "Жилая"})],
            }

        async def get_physical_objects_with_geometry(self, sid, **kwargs):
            state.physical_filters.append(kwargs)
            return {
                "features": [
                    feature(
                        physical_object_type={
                            "physical_object_type_id": 4,
                            "name": "Жилой дом",
                        }
                    )
                ]
            }

        async def get_services_with_geometry(self, sid, **kwargs):
            return {
                "features": [
                    feature(service_type={"service_type_id": 22, "name": "Школа"})
                ]
            }

    async def context(*args, **kwargs):
        state.context_args = kwargs
        return {"version": 1, "regulations": None, "rules": [], "notes": []}

    def persist(**kwargs):
        state.captured.append(copy.deepcopy(kwargs))
        return SimpleNamespace(external_id="new-placement")

    monkeypatch.setattr(mod, "UrbanApiClient", Urban)
    monkeypatch.setattr(mod, "_placement_context", context)
    monkeypatch.setattr(mod, "_persist_and_create_scenario_task", persist)
    repo = SimpleNamespace(get_by_idempotency_key=lambda key: state.cached)

    def run(**overrides):
        args = dict(
            scenario_id=42,
            year=2025,
            source="User",
            service_name="школа",
            model=None,
            priority=1,
            force_recompute=False,
            idempotency_key_form=None,
            idempotency_key_header=None,
            token="user-token",
            app_settings=settings,
            task_repo=repo,
            event_repo=None,
            session=None,
        )
        args.update(overrides)
        return asyncio.run(mod._build_scenario_placement_task(**args))

    state.run = run
    state.settings = settings
    return state


def test_scenario_fetches_all_objects_and_persists_placement_context(flow):
    assert flow.run().external_id == "new-placement"
    args = flow.captured[0]
    assert flow.physical_filters == [{"token": "user-token"}]
    assert len(args["objects_fc"]["features"]) == 2
    assert args["placement_vri_code"] == "3.5.1"
    assert args["placement_context"]["input_kind"] == "scenario_objects"
    assert "а не кадастровые" in " ".join(args["placement_context"]["notes"])
    assert flow.context_args["norms_only"] is True
    assert flow.context_args["territory_id"] is None
    assert "ВРИ 2." in args["objects_fc"]["features"][0]["properties"]["vri_text"]
    assert "ВРИ 3.5.1" in args["objects_fc"]["features"][1]["properties"]["vri_text"]


def test_classification_regulations_are_persisted_and_isolate_cache(flow):
    descriptions = {
        "functional_zone_mappings": [
            {"functional_zone_type_id": 1, "averaged_pzz_profile": {"main_vri": []}}
        ]
    }
    classifier = {"by_code": {"2.1": {"name": "Свое название"}}}
    cache = {}
    args = dict(
        scenario_id=42,
        year=2025,
        source="User",
        physical_object_type_id=4,
        priority=1,
        force_recompute=False,
        idempotency_key_form=None,
        idempotency_key_header="same-key",
        token="user-token",
        app_settings=flow.settings,
        task_repo=SimpleNamespace(get_by_idempotency_key=cache.get),
        event_repo=None,
        session=None,
    )
    for extra in [
        {},
        {"zone_descriptions": descriptions},
        {"vri_classifier": classifier},
        {"zone_descriptions": descriptions, "vri_classifier": classifier},
        {"cadastral_fc": {"type": "FeatureCollection", "features": []}},
        {
            "cadastral_fc": {
                "type": "FeatureCollection",
                "features": [feature(vri_text="2.1")],
            }
        },
    ]:
        asyncio.run(mod._build_scenario_classification_task(**args, **extra))
        captured = flow.captured[-1]
        assert captured["zone_descriptions"] == extra.get("zone_descriptions")
        assert captured["vri_classifier"] == extra.get("vri_classifier")
        assert captured["cadastral_fc"] == extra.get("cadastral_fc")
        key = captured["namespaced_key"]
        assert key not in cache
        cache[key] = SimpleNamespace(id=1, external_id=key)
        assert (
            asyncio.run(mod._build_scenario_classification_task(**args, **extra))
            is cache[key]
        )
    assert len(flow.captured) == 6
    assert len(flow.calls) == 12  # Access is checked even for cached results.


def test_scenario_placement_cadastre_is_separate_and_changes_cache(flow):
    for cadastre in [
        None,
        {"features": [feature(vri_text="2.1")]},
        {"features": [feature(vri_text="6.9")]},
    ]:
        flow.run(cadastral_fc=cadastre)
    assert len({args["namespaced_key"] for args in flow.captured}) == 3
    for args in flow.captured[1:]:
        assert args["placement_context"]["input_kind"] == "scenario_with_cadastre"
        assert args["cadastral_fc"]["features"][0]["properties"]["vri_text"] in {
            "2.1",
            "6.9",
        }
        assert all(
            f["properties"]["placement_object_type"]
            for f in args["objects_fc"]["features"]
        )


def test_scenario_cache_never_bypasses_access_check(flow):
    flow.cached = SimpleNamespace(external_id="cached")
    assert flow.run() is flow.cached
    assert flow.calls == [("access", 42, "user-token")]
    assert not flow.captured
    flow.access_error = UrbanApiError(403, "denied")
    with pytest.raises(HTTPException) as exc:
        flow.run()
    assert exc.value.status_code == 403


def test_scenario_cache_separates_sibling_services_and_sources(flow):
    for params in [{}, {"service_name": "детский сад"}, {"source": "PZZ"}]:
        flow.sources.append({"year": 2025, "source": "PZZ"})
        flow.run(idempotency_key_header="same-key", **params)
    keys = [args["namespaced_key"] for args in flow.captured]
    assert len(set(keys)) == 3
    assert all(key.startswith("sc:42:2025:") and ":placement:" in key for key in keys)


def test_scenario_rejects_unavailable_zone_source_before_fetching_objects(flow):
    with pytest.raises(HTTPException) as exc:
        flow.run(source="PZZ")
    assert exc.value.status_code == 422
    assert exc.value.detail["available"] == [{"year": 2025, "source": "User"}]
    assert not flow.physical_filters


def test_scenario_placement_is_routed_by_target_vri_not_key_prefix():
    task = SimpleNamespace(
        id=1,
        external_id="x",
        priority=1,
        status=TaskStatus.queued,
        cadastral_data_path="objects",
        pzz_zones_data_path="zones",
        pzz_zone_vri_labels_path="",
        vri_classifier_path="vri",
        include_pzz_check=True,
        cadastral_vri_col="vri_text",
        pzz_zone_code_col="zone_code",
        pzz_zone_name_col="zone_name",
        building_type_col=None,
        building_service_col=None,
        building_floors_col=None,
        mo_boundaries_data_path=None,
        placement_vri_code="3.5.1",
        placement_service_name="Школа",
        placement_context_path="context",
    )
    repo = Mock()
    repo.get_by_id.return_value = task
    repo.get_idempotency_key_by_external_id.return_value = (
        "sc:42:2025:User:placement:3.5.1"
    )
    config = Mock()
    config.get_int.side_effect = lambda key, default, **kw: default
    result = start_task(
        task_id=1,
        task_repo=repo,
        config_repo=config,
        event_repo=Mock(),
        settings=SimpleNamespace(priority_max_sum_default=20, outputs_dir="outputs"),
    )
    assert result.request.is_scenario and result.request.is_placement
    assert result.request.placement_vri_code == "3.5.1"


def test_scenario_summary_does_not_call_objects_cadastral_parcels():
    text = _build_chat_message_placement(
        {
            "scenario_objects_used": True,
            "cadastre_used": False,
            "parcels": {"housing": 2, "target": 1},
            "classes": {},
            "norms": [],
        }
    )
    assert "Объекты сценария:" in text
    assert "Кадастровый слой не передан" not in text


def test_placement_summary_checks_scenario_membership(monkeypatch):
    async def verify(**kwargs):
        pass

    monkeypatch.setattr(mod, "_verify_scenario_access", verify)
    repo = Mock()
    repo.get_by_external_id.return_value = SimpleNamespace(external_id="x")
    repo.get_idempotency_key_by_external_id.return_value = (
        "sc:99:2025:User:placement:3.5.1"
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            mod.get_scenario_placement_summary_endpoint(
                scenario_id=42,
                external_id="x",
                token="t",
                app_settings=SimpleNamespace(),
                task_repo=repo,
            )
        )
    assert exc.value.status_code == 404


def test_chat_placement_preserves_project_and_uses_placement_report(monkeypatch):
    from service.api.security import AuthUser
    from service.schemas import TaskOut
    from datetime import datetime, timezone

    task = TaskOut(
        id=1,
        external_id="x",
        status="queued",
        priority=1,
        cadastral_data_path="objects",
        pzz_zones_data_path="zones",
        created_at=datetime.now(timezone.utc),
        include_pzz_check=True,
        cadastral_vri_col="vri_text",
        pzz_zone_code_col="zone_code",
        pzz_zone_name_col="zone_name",
        result_path=None,
        error_text=None,
        celery_task_id=None,
        started_at=None,
        finished_at=None,
    )

    async def build(**kwargs):
        return task

    async def project(*args):
        return 7

    captured = {}

    async def stream(*args, **kwargs):
        captured.update(kwargs)
        yield {"event": "done", "data": "{}"}

    monkeypatch.setattr(mod, "_build_scenario_placement_task", build)
    monkeypatch.setattr(mod, "_fetch_project_id", project)
    monkeypatch.setattr(mod, "task_stream_with_chat_generator", stream)
    response = asyncio.run(
        mod.scenario_chat_stream_endpoint(
            request=None,
            scenario_id=42,
            year=2025,
            source="User",
            user_query=None,
            mode="placement",
            service_name="школа",
            chat_id=None,
            physical_object_type_id=4,
            priority=1,
            force_recompute=False,
            group_by="zone",
            model=None,
            temperature=None,
            poll_interval=2,
            idempotency_key_form=None,
            idempotency_key_header=None,
            pzz_descriptions_file=None,
            pzz_descriptions_upload_id=None,
            vri_classifier_file=None,
            vri_classifier_upload_id=None,
            cadastral_feature_collection_file=None,
            cadastral_feature_collection_upload_id=None,
            cadastral_vri_col=None,
            user=AuthUser(token="t", user_id="u"),
            app_settings=SimpleNamespace(
                chat_system_prompt_placement_path="placement-prompt"
            ),
            task_repo=None,
            event_repo=None,
            session=Mock(),
        )
    )

    async def consume():
        return [chunk async for chunk in response.body_iterator]

    asyncio.run(consume())
    assert captured["report_kind"] == "placement"
    assert captured["scenario_id"] == 42 and captured["project_id"] == 7
    assert captured["system_prompt_path"] == "placement-prompt"
