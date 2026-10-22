"""Non-residential buildings (type 5) without services get «Не определено», not a 422."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from service.api import scenarios
from service.api.scenarios import _flatten_functional_zone_features
from service.domain.contracts import PipelineRequest
from service.infrastructure.runners.deterministic_scenario_runner import (
    DeterministicScenarioRunner,
)

_RESIDENTIAL = {"id": 1, "name": "residential", "nickname": "Жилая зона"}


def _zone() -> dict:
    ring = [[30.0, 59.0], [30.01, 59.0], [30.01, 59.01], [30.0, 59.01], [30.0, 59.0]]
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {"name": "Квартал", "functional_zone_type": _RESIDENTIAL},
    }


def _non_residential(x: float) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [x, 59.005]},
        "properties": {
            "physical_object_type": {
                "physical_object_type_id": 5,
                "name": "Нежилое здание",
            },
            "vri_text": "Нежилое здание",
        },
    }


def _run(tmp_path, objects: list[dict]) -> list[dict]:
    zones = _flatten_functional_zone_features(
        {"type": "FeatureCollection", "features": [_zone()]},
        code_col="zone_code",
        name_col="zone_name",
    )
    (tmp_path / "o.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": objects}),
        encoding="utf-8",
    )
    (tmp_path / "z.geojson").write_text(json.dumps(zones), encoding="utf-8")
    runner = DeterministicScenarioRunner(
        SimpleNamespace(
            physical_object_type_to_vri_path="data/physical_object_type_to_vri.json",
            default_fz_to_pzz_mapping_path="data/functional_zones_to_pzz_mapping.json",
            service_type_to_vri_path="data/service_type_to_vri.json",
        )
    )
    request = PipelineRequest(
        task_external_id="ext-nr",
        cadastral_data_path=str(tmp_path / "o.geojson"),
        pzz_zones_data_path=str(tmp_path / "z.geojson"),
        pzz_zone_vri_labels_path="",
        vri_classifier_path="",
        include_pzz_check=True,
        cadastral_vri_col="vri_text",
        pzz_zone_code_col="zone_code",
        pzz_zone_name_col="zone_name",
        outputs_dir=str(tmp_path / "out"),
        is_scenario=True,
    )
    with open(runner.run(request), encoding="utf-8") as fh:
        return [f["properties"] for f in json.load(fh)["features"]]


def test_non_residential_building_is_undetermined(tmp_path):
    inside, outside = _run(tmp_path, [_non_residential(30.005), _non_residential(31.0)])

    assert "Не определено" in inside.values()
    assert "Нежилое здание: вид разрешённого использования не определён" in (
        inside["Причина"]
    )
    # Outside every zone the "no zone" verdict still wins.
    assert "Не определено" not in outside.values()
    assert "Нежилое здание" not in outside["Причина"]


class _FakeUrban:
    def __init__(self, by_type: dict[int, list[dict]]):
        self.by_type = by_type
        self.requested_types: list[int] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_scenario_info(self, scenario_id, token):
        return {"updated_at": "t"}

    async def list_functional_zone_sources(self, scenario_id, token):
        return [{"year": 2024, "source": "OSM"}]

    async def get_functional_zones(self, scenario_id, year, source, token):
        return {"type": "FeatureCollection", "features": [_zone()]}

    async def get_physical_objects_with_geometry(
        self, scenario_id, physical_object_type_id, token
    ):
        self.requested_types.append(physical_object_type_id)
        return {
            "type": "FeatureCollection",
            "features": self.by_type.get(physical_object_type_id, []),
        }

    async def get_services_with_geometry(self, scenario_id, token):
        return {"type": "FeatureCollection", "features": []}


def _build(monkeypatch, fake: _FakeUrban) -> dict:
    captured: dict = {}
    monkeypatch.setattr(scenarios, "UrbanApiClient", lambda **kw: fake)
    monkeypatch.setattr(
        scenarios,
        "_persist_and_create_scenario_task",
        lambda **kw: captured.update(kw) or "task",
    )
    task_repo = SimpleNamespace(get_by_idempotency_key=lambda key: None)
    result = asyncio.run(
        scenarios._build_scenario_classification_task(
            scenario_id=14656,
            year=2024,
            source="OSM",
            physical_object_type_id=4,
            priority=0,
            force_recompute=False,
            idempotency_key_form=None,
            idempotency_key_header=None,
            token="t",
            app_settings=SimpleNamespace(
                urban_api_base_url="http://urban", urban_api_timeout_seconds=5
            ),
            task_repo=task_repo,
            event_repo=None,
            session=None,
        )
    )
    assert result == "task"
    return captured


def test_falls_back_to_non_residential_when_no_houses(monkeypatch):
    fake = _FakeUrban({5: [_non_residential(30.005)]})

    captured = _build(monkeypatch, fake)

    assert fake.requested_types == [4, 5]
    assert len(captured["objects_fc"]["features"]) == 1


def test_still_422_when_no_buildings_at_all(monkeypatch):
    fake = _FakeUrban({})

    with pytest.raises(HTTPException) as exc_info:
        _build(monkeypatch, fake)

    assert exc_info.value.status_code == 422
    assert fake.requested_types == [4, 5]
