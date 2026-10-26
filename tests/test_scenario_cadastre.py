"""Uploaded parcels constrain scenario objects, without becoming result objects."""

import asyncio
import copy
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient
from shapely.geometry import Point, box, mapping
from sse_starlette.sse import AppStatus

from service.api import scenarios as mod
from service.api import classifier as classifier_api
from service.api.security import (
    AuthUser,
    get_current_user,
    get_optional_user,
    verify_token,
)
from service.api.tasks import (
    build_object_zone_fit_response,
    build_scenario_zone_geo_layers,
)
from service.dependencies import get_app_settings, get_db, get_event_repo, get_task_repo
from service.domain import PipelineRequest
from service.infrastructure.runners.deterministic_scenario_runner import (
    DeterministicScenarioRunner,
)
from service.infrastructure.scenario_cadastre import CadastreChecker, PARCEL_VERDICT
from service.schemas import TaskOut


def fc(*features):
    return {"type": "FeatureCollection", "features": list(features)}


def feature(geometry, **properties):
    return {
        "type": "Feature",
        "geometry": json.loads(json.dumps(mapping(geometry))),
        "properties": properties,
    }


def parcel(vri="2.1", x=30, **props):
    return feature(
        box(x, 59, x + 0.01, 59.01),
        vri_text=vri,
        cadastral_number=f"65:01:{x}",
        **props,
    )


@pytest.fixture
def settings(tmp_path):
    return SimpleNamespace(
        uploads_dir=str(tmp_path / "uploads"),
        max_upload_bytes=1_000_000,
        task_inputs_dir=str(tmp_path / "inputs"),
        outputs_dir=str(tmp_path / "outputs"),
        default_vri_classifier_path="data/rosreestr_vri_classifier_2024_12_24.json",
        physical_object_type_to_vri_path="data/physical_object_type_to_vri.json",
        default_fz_to_pzz_mapping_path="data/functional_zones_to_pzz_mapping.json",
        service_type_to_vri_path="data/service_type_to_vri.json",
        chat_system_prompt_scenario_path="scenario-prompt",
        chat_system_prompt_placement_path="placement-prompt",
        app_name="pzz",
    )


def read(doc, settings, column=None, filename="parcels.geojson"):
    raw = json.dumps(doc).encode() if not isinstance(doc, bytes) else doc
    upload = UploadFile(filename=filename, file=io.BytesIO(raw))
    return mod._read_scenario_cadastre(
        upload, None, column, owner_id="u", app_settings=settings
    )


def test_cadastre_geometry_and_selected_vri_column_are_kept(settings):
    p = parcel()
    p["properties"] = {"my_use": "2.1", "cadastral_number": "65:01:1"}
    result = read(fc(p), settings, "my_use")
    assert result["features"][0]["geometry"] == p["geometry"]
    assert result["features"][0]["properties"]["vri_text"] == "2.1"
    assert result["features"][0]["properties"]["my_use"] == "2.1"
    assert not list(Path(settings.uploads_dir).glob("resolve-*"))


@pytest.mark.parametrize(
    "doc,column,filename,status",
    [
        (b"not json", None, "p.json", 422),
        ([], None, "p.json", 422),
        (fc(), None, "p.geojson", 422),
        (fc(feature(Point(30, 59), vri="2.1")), None, "p.geojson", 422),
        (fc(parcel()), "no_such_column", "p.geojson", 422),
        (fc(parcel()), None, "p.csv", 415),
        (
            fc(feature(box(1000000, 1000000, 1000010, 1000010), vri="2.1")),
            None,
            "p.geojson",
            422,
        ),
    ],
)
def test_invalid_cadastre_is_rejected(settings, doc, column, filename, status):
    with pytest.raises(HTTPException) as exc:
        read(doc, settings, column, filename)
    assert exc.value.status_code == status


def test_cadastre_size_limit(settings):
    settings.max_upload_bytes = 10
    with pytest.raises(HTTPException) as exc:
        read(fc(parcel()), settings)
    assert exc.value.status_code == 413


def test_cadastre_vector_conversion_keeps_geometry(settings, tmp_path):
    import geopandas as gpd

    path = tmp_path / "parcels.gpkg"
    gpd.GeoDataFrame(
        {"vri": ["2.1"]}, geometry=[box(30, 59, 30.01, 59.01)], crs=4326
    ).to_file(path)
    result = read(path.read_bytes(), settings, filename="parcels.gpkg")
    assert result["features"][0]["geometry"]["type"] == "Polygon"
    assert result["features"][0]["properties"]["vri_text"] == "2.1"


@pytest.mark.parametrize(
    "value,object_code,expected",
    [
        ("2.1", "2.1.1", True),
        ("2.1.1", "2.1.1", True),
        ("6.9", "2.1.1", False),
        ("2.1.1", "2.1", None),
        ("неизвестное назначение", "2.1.1", None),
        (None, "2.1.1", None),
        ("2.1", None, None),
        ("Для складов", "6.9", True),
        (["6.9", "2.1"], "2.1.1", True),
    ],
)
def test_cadastre_codes_and_names(value, object_code, expected):
    checker = CadastreChecker(
        fc(parcel(value)),
        "vri_text",
        {"2.1": "Жилье", "2.1.1": "Малоэтажное жилье", "6.9": "Складов"},
    )
    status, props = checker.check(feature(Point(30.005, 59.005)), object_code)
    assert status is expected
    assert props[PARCEL_VERDICT] == (
        "Соответствует"
        if expected is True
        else "Не соответствует" if expected is False else "Требуется ручная проверка"
    )


def test_ambiguous_classifier_name_is_not_allowed():
    checker = CadastreChecker(
        fc(parcel("Жилье")), "vri_text", {"2.1": "Жилье", "2.5": "Жилье"}
    )
    assert checker.check(feature(Point(30.005, 59.005)), "2.1")[0] is None


def test_cadastre_checks_all_intersected_parcels_and_partial_coverage():
    checker = CadastreChecker(
        fc(parcel("2.1"), parcel("6.9", x=30.01)),
        "vri_text",
        {"2.1": "Жилье", "6.9": "Склады"},
    )
    assert (
        checker.check(feature(box(30.005, 59.002, 30.015, 59.008)), "2.1")[0] is False
    )
    assert checker.check(feature(box(29.999, 59.002, 30.005, 59.008)), "2.1")[0] is None
    assert checker.check(feature(Point(31, 59)), "2.1")[0] is None
    # An adjacent parcel touched by an object's boundary cannot prohibit it.
    assert checker.check(feature(box(30.002, 59.002, 30.01, 59.008)), "2.1")[0] is True


def test_worker_and_report_keep_scenario_objects_and_cadastral_evidence(
    settings, tmp_path, monkeypatch
):
    objects = fc(
        *[
            feature(
                Point(x, 59.005),
                physical_object_type={
                    "physical_object_type_id": 4,
                    "name": "Жилой дом",
                },
                properties={"Количество этажей": 2},
                vri_text="Жилой дом",
            )
            for x in (30.005, 30.015, 30.03)
        ]
    )
    zones = fc(
        feature(box(29.99, 58.99, 30.1, 59.1), zone_code="1", zone_name="Квартал")
    )
    rules = {
        "functional_zone_mappings": [
            {
                "functional_zone_type_id": 1,
                "averaged_pzz_profile": {"main_vri": [{"vri_code": "2.1"}]},
            }
        ]
    }
    docs = {
        "objects.json": objects,
        "parcels.json": fc(parcel(), parcel("6.9", x=30.01)),
        "zones.json": zones,
        "rules.json": rules,
    }
    for name, doc in docs.items():
        (tmp_path / name).write_text(json.dumps(doc), encoding="utf-8")
    request = PipelineRequest(
        task_external_id="cadastre",
        cadastral_data_path=str(tmp_path / "parcels.json"),
        existing_objects_data_path=str(tmp_path / "objects.json"),
        pzz_zones_data_path=str(tmp_path / "zones.json"),
        pzz_zone_vri_labels_path=str(tmp_path / "rules.json"),
        vri_classifier_path=settings.default_vri_classifier_path,
        include_pzz_check=True,
        cadastral_vri_col="vri_text",
        pzz_zone_code_col="zone_code",
        pzz_zone_name_col="zone_name",
        outputs_dir=settings.outputs_dir,
        is_scenario=True,
    )
    from service.infrastructure.runners import pipeline_runner as runner_module

    monkeypatch.setattr(runner_module, "get_object_storage", lambda: None)
    settings.scenario_deterministic = False
    routed = runner_module.PipelineRunnerFactory.create(settings, request)
    assert isinstance(routed._inner, DeterministicScenarioRunner)
    task = SimpleNamespace(
        status="finished",
        result_path=DeterministicScenarioRunner(settings).run(request),
        pzz_zone_vri_labels_path=request.pzz_zone_vri_labels_path,
        cadastral_data_path=request.cadastral_data_path,
        pzz_zones_data_path=request.pzz_zones_data_path,
        existing_objects_data_path=request.existing_objects_data_path,
    )
    report = build_object_zone_fit_response(
        task, "cadastre", "object", settings, scenario=True
    )
    assert report["summary"]["total"] == 3
    assert report["summary"]["cadastre_check"] == {
        "correct": 1,
        "wrong": 1,
        "unclear": 1,
    }
    assert [o["fit"] for o in report["objects"]] == ["correct", "wrong", "unclear"]
    assert all(o["category"] == "Здание" and o["cadastre"] for o in report["objects"])
    assert "ВРИ" in report["chat_message"]
    layers = build_scenario_zone_geo_layers(task, "cadastre", settings)
    assert [layer["name"] for layer in layers] == [
        "functional_zones",
        "input_cadastral",
    ]
    assert layers[1]["url"] == "/files/cadastral/cadastre"


def test_placement_uses_parcels_and_scenario_target_objects_separately(
    settings, tmp_path
):
    from service.infrastructure.runners.placement_pzz_runner import (
        PlacementPzzRunner,
        PLACEMENT_SUMMARY_KEY,
    )

    docs = {
        "parcels.json": fc(parcel("3.5.1"), parcel("3.5.1", x=30.01)),
        "objects.json": fc(
            feature(Point(30.005, 59.005), placement_object_type="Школа (ВРИ 3.5.1)")
        ),
        "zones.json": fc(
            feature(box(29.99, 58.99, 30.1, 59.1), zone_code="1", zone_name="Квартал")
        ),
        "rules.json": {
            "functional_zone_mappings": [
                {
                    "functional_zone_type_id": 1,
                    "averaged_pzz_profile": {"main_vri": [{"vri_code": "3.5.1"}]},
                }
            ]
        },
        "context.json": {
            "input_kind": "scenario_with_cadastre",
            "rules": [],
            "notes": [],
        },
    }
    for name, doc in docs.items():
        (tmp_path / name).write_text(json.dumps(doc), encoding="utf-8")
    request = PipelineRequest(
        task_external_id="placement-cadastre",
        cadastral_data_path=str(tmp_path / "parcels.json"),
        existing_objects_data_path=str(tmp_path / "objects.json"),
        pzz_zones_data_path=str(tmp_path / "zones.json"),
        pzz_zone_vri_labels_path=str(tmp_path / "rules.json"),
        vri_classifier_path=settings.default_vri_classifier_path,
        include_pzz_check=True,
        cadastral_vri_col="vri_text",
        pzz_zone_code_col="zone_code",
        pzz_zone_name_col="zone_name",
        outputs_dir=settings.outputs_dir,
        is_scenario=True,
        is_placement=True,
        placement_vri_code="3.5.1",
        placement_service_name="Школа",
        placement_context_path=str(tmp_path / "context.json"),
    )
    result = json.loads(
        Path(PlacementPzzRunner(settings).run(request)).read_text(encoding="utf-8")
    )
    summary = result[PLACEMENT_SUMMARY_KEY]
    assert summary["cadastre_used"] and summary["scenario_objects_used"]
    assert summary["existing_objects_source"] == "scenario"
    assert summary["existing_objects_count"] == 1
    assert summary["parcels"]["target"] == 2


@pytest.mark.parametrize("placement", [False, True])
def test_persistence_separates_parcels_and_objects(monkeypatch, settings, placement):
    persisted, captured = {}, {}

    def persist(doc, task_dir, name, external_id, storage):
        persisted[name] = copy.deepcopy(doc)
        return name

    monkeypatch.setattr(mod, "persist_geojson_dict", persist)
    monkeypatch.setattr(mod, "get_object_storage", lambda: None)
    monkeypatch.setattr(
        mod,
        "create_task",
        lambda **kw: captured.update(kw) or SimpleNamespace(id=1, external_id="x"),
    )
    parcels = fc(parcel())
    objects = fc(
        feature(
            Point(30.005, 59.005),
            physical_object_type={"physical_object_type_id": 4, "name": "Дом"},
        )
    )
    mod._persist_and_create_scenario_task(
        objects_fc=objects,
        zones_fc=fc(),
        cadastral_fc=parcels,
        scenario_id=1,
        year=2026,
        source="User",
        physical_object_type_id=4,
        priority=1,
        namespaced_key="key",
        force_recompute=False,
        app_settings=settings,
        task_repo=None,
        event_repo=None,
        session=SimpleNamespace(flush=lambda: None, refresh=lambda t: None),
        placement_vri_code="3.5.1" if placement else None,
    )
    assert persisted["cadastral_feature_collection.geojson"] == parcels
    assert len(persisted["scenario_objects.geojson"]["features"]) == 1
    assert (
        captured["input_paths"]["existing_objects_data_path"]
        == "scenario_objects.geojson"
    )


@pytest.mark.parametrize(
    "route,mode",
    [
        ("chat/stream", "pzz_check"),
        ("chat/stream", "placement"),
        ("classify", None),
        ("classify/stream", None),
        ("placement", None),
    ],
)
@pytest.mark.parametrize("via_id", [False, True])
def test_all_scenario_submission_routes_accept_parcels(
    monkeypatch, settings, tmp_path, route, mode, via_id
):
    captured = {}
    task = TaskOut(
        id=1,
        external_id="x",
        status="queued",
        priority=1,
        cadastral_data_path="parcels",
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
        captured.update(kwargs)
        return task

    async def stream(*args, **kwargs):
        yield {"event": "done", "data": "{}"}

    async def project(*args):
        return 7

    monkeypatch.setattr(mod, "_build_scenario_classification_task", build)
    monkeypatch.setattr(mod, "_build_scenario_placement_task", build)
    monkeypatch.setattr(mod, "_fetch_project_id", project)
    monkeypatch.setattr(mod, "task_stream_with_chat_generator", stream)
    monkeypatch.setattr(mod, "task_stream_with_report_generator", stream)
    monkeypatch.setattr(AppStatus, "should_exit_event", None)
    app = FastAPI()
    app.include_router(mod.router)
    app.dependency_overrides.update(
        {
            get_app_settings: lambda: settings,
            get_db: lambda: SimpleNamespace(commit=lambda: None),
            get_current_user: lambda: AuthUser(token="t", user_id="u"),
            get_optional_user: lambda: AuthUser(token="t", user_id="u"),
            verify_token: lambda: "t",
            get_task_repo: lambda: None,
            get_event_repo: lambda: None,
        }
    )
    doc = fc(parcel())
    data = {
        "year": 2026,
        "source": "User",
        "user_query": "проверить",
        "service_name": "школа",
    }
    if mode:
        data["mode"] = mode
    files = None
    if via_id:
        path = tmp_path / "cadastre.geojson"
        path.write_text(json.dumps(doc), encoding="utf-8")

        def resolve(upload_id, *, owner_id, **kwargs):
            assert upload_id == "parcel-id" and owner_id == "u"
            return path

        monkeypatch.setattr(classifier_api, "resolve_upload", resolve)
        data["cadastral_feature_collection_upload_id"] = "parcel-id"
    else:
        files = {
            "cadastral_feature_collection_file": ("parcels.geojson", json.dumps(doc))
        }
    response = TestClient(app).post(f"/scenarios/42/{route}", data=data, files=files)
    assert response.status_code == 200, response.text
    assert captured["cadastral_fc"] == doc
