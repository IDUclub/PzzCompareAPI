"""Tests for geo-layer download links: descriptor + /files redirect (phase 8)."""
from types import SimpleNamespace

import service.api.tasks as tasks_mod
from service.api.tasks import (
    build_input_geo_layers,
    build_result_geo_layer,
    build_result_geo_layers,
    geo_layer_to_file_part,
)
from service.settings import get_settings


def _task(
    status="finished",
    result_path="outputs/abc/result.geojson",
    cadastral_data_path="minio://inputs/abc/cadastral_feature_collection.geojson",
    pzz_zones_data_path="minio://inputs/abc/pzz_zones_feature_collection.geojson",
    include_pzz_check=True,
    building_type_col=None,
    building_service_col=None,
):
    return SimpleNamespace(
        status=status,
        result_path=result_path,
        cadastral_data_path=cadastral_data_path,
        pzz_zones_data_path=pzz_zones_data_path,
        include_pzz_check=include_pzz_check,
        building_type_col=building_type_col,
        building_service_col=building_service_col,
    )


def test_layer_descriptor_local_result_relative_url() -> None:
    settings = get_settings()  # public_base_url empty in tests
    layer = build_result_geo_layer(_task(), "abc123", settings)
    assert layer is not None
    assert layer["url"] == "/files/result/abc123"
    assert layer["download_url"] is None  # local storage can't presign
    # Human-readable RU label + ASCII English filename (not the opaque task hash).
    assert layer["title"] == "Результат проверки ПЗЗ"
    assert layer["filename"] == "pzz_check_result.geojson"
    assert layer["mime_type"] == "application/geo+json"


def test_result_layer_labels_classify_run() -> None:
    # Same ``result`` slot, but a classify-only run is labelled distinctly.
    layer = build_result_geo_layer(_task(include_pzz_check=False), "abc123", get_settings())
    assert layer is not None
    assert layer["title"] == "Результат классификации ВРИ"
    assert layer["filename"] == "classification_result.geojson"


def test_result_layers_single_for_non_building() -> None:
    # Non-building tasks keep the single combined result layer.
    layers = build_result_geo_layers(_task(), "abc123", get_settings())
    assert [layer["name"] for layer in layers] == ["classified_result"]
    assert layers[0]["url"] == "/files/result/abc123"


def test_result_layers_two_for_building() -> None:
    # building_pzz_check splits the result into здания + сервисы layers.
    task = _task(building_type_col="po_type", building_service_col="svc_type")
    layers = build_result_geo_layers(task, "abc123", get_settings())
    by_name = {layer["name"]: layer for layer in layers}
    assert set(by_name) == {"buildings_result", "services_result"}
    assert by_name["buildings_result"]["url"] == "/files/result_buildings/abc123"
    assert by_name["buildings_result"]["title"] == "Результат — здания"
    assert by_name["buildings_result"]["filename"] == "buildings_result.geojson"
    assert by_name["services_result"]["url"] == "/files/result_services/abc123"
    assert by_name["services_result"]["title"] == "Результат — сервисы"
    # split served on the fly -> durable url only, no presigned download
    assert all(layer["download_url"] is None for layer in layers)
    assert all(layer["role"] == "result" for layer in layers)


def test_files_result_split_filters_by_category(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from service import app as app_module
    from service.dependencies import get_app_settings, get_task_repo

    result = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": None, "properties": {"Категория_объекта": "Здание", "id": 1}},
            {"type": "Feature", "geometry": None, "properties": {"Категория_объекта": "Сервис", "id": 2}},
            {"type": "Feature", "geometry": None, "properties": {"Категория_объекта": "Здание", "id": 3}},
        ],
    }
    result_file = tmp_path / "result.geojson"
    result_file.write_text(__import__("json").dumps(result, ensure_ascii=False), encoding="utf-8")
    task = _task(result_path=str(result_file), building_type_col="po", building_service_col="svc")

    class StubRepo:
        def get_by_external_id(self, external_id):
            return task

    app_module.app.dependency_overrides[get_task_repo] = lambda: StubRepo()
    app_module.app.dependency_overrides[get_app_settings] = get_settings
    try:
        client = TestClient(app_module.app)
        r_bld = client.get("/files/result_buildings/abc")
        r_svc = client.get("/files/result_services/abc")
        assert r_bld.status_code == 200 and r_svc.status_code == 200
        bld_ids = [f["properties"]["id"] for f in r_bld.json()["features"]]
        svc_ids = [f["properties"]["id"] for f in r_svc.json()["features"]]
        assert bld_ids == [1, 3]
        assert svc_ids == [2]
        assert "attachment" in r_bld.headers.get("content-disposition", "")
    finally:
        app_module.app.dependency_overrides.clear()


def test_layer_descriptor_none_when_no_result() -> None:
    settings = get_settings()
    assert build_result_geo_layer(_task(status="running"), "x", settings) is None
    assert build_result_geo_layer(_task(result_path=None), "x", settings) is None


def test_file_part_keeps_only_durable_url() -> None:
    layer = build_result_geo_layer(_task(), "abc123", get_settings())
    part = geo_layer_to_file_part(layer)
    assert part["url"] == "/files/result/abc123"
    # The ephemeral presigned download_url must never be persisted to history.
    assert "download_url" not in part
    # Human-readable label rides along so the frontend can show it in history.
    assert part["title"] == "Результат проверки ПЗЗ"
    assert part["filename"] == "pzz_check_result.geojson"
    assert part["mime_type"] == "application/geo+json"


def test_input_layers_for_uploaded_files() -> None:
    settings = get_settings()
    layers = build_input_geo_layers(_task(), "abc123", settings)
    by_name = {layer["name"]: layer for layer in layers}
    assert set(by_name) == {"input_cadastral", "input_zones"}
    assert by_name["input_cadastral"]["url"] == "/files/cadastral/abc123"
    assert by_name["input_cadastral"]["title"] == "Исходные участки"
    assert by_name["input_cadastral"]["filename"] == "input_parcels.geojson"
    assert by_name["input_zones"]["url"] == "/files/zones/abc123"
    assert by_name["input_zones"]["title"] == "Зоны ПЗЗ"
    assert by_name["input_zones"]["filename"] == "pzz_zones.geojson"
    assert all(layer["role"] == "input" for layer in layers)


def test_input_layers_skip_missing_zones() -> None:
    # classify-only uploads have no zones layer.
    layers = build_input_geo_layers(
        _task(pzz_zones_data_path=""), "abc123", get_settings()
    )
    assert [layer["name"] for layer in layers] == ["input_cadastral"]


def test_files_cadastral_slot_redirects(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from service import app as app_module
    from service.dependencies import get_app_settings, get_task_repo

    task = _task(status="running")  # inputs available even before finish

    class StubRepo:
        def get_by_external_id(self, external_id):
            return task

    class FakeStorage:
        def presigned_url(self, stored_path, expires_seconds=3600):
            return "https://minio.example/cadastral?sig=1"

    monkeypatch.setattr(tasks_mod, "get_object_storage", lambda: FakeStorage())
    app_module.app.dependency_overrides[get_task_repo] = lambda: StubRepo()
    app_module.app.dependency_overrides[get_app_settings] = get_settings
    try:
        client = TestClient(app_module.app)
        resp = client.get("/files/cadastral/abc", follow_redirects=False)
        assert resp.status_code == 307
        assert resp.headers["location"] == "https://minio.example/cadastral?sig=1"
    finally:
        app_module.app.dependency_overrides.clear()


def test_files_unknown_slot_404() -> None:
    from fastapi.testclient import TestClient

    from service import app as app_module
    from service.dependencies import get_app_settings, get_task_repo

    app_module.app.dependency_overrides[get_task_repo] = lambda: object()
    app_module.app.dependency_overrides[get_app_settings] = get_settings
    try:
        client = TestClient(app_module.app)
        resp = client.get("/files/bogus/abc", follow_redirects=False)
        assert resp.status_code == 404
    finally:
        app_module.app.dependency_overrides.clear()


def test_files_result_redirects_to_presigned(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from service import app as app_module
    from service.dependencies import get_app_settings, get_task_repo

    task = _task(result_path="minio://outputs/abc/result.geojson")

    class StubRepo:
        def get_by_external_id(self, external_id):
            return task

    class FakeStorage:
        def presigned_url(self, stored_path, expires_seconds=3600):
            return "https://minio.example/presigned?sig=1"

    monkeypatch.setattr(tasks_mod, "get_object_storage", lambda: FakeStorage())
    app_module.app.dependency_overrides[get_task_repo] = lambda: StubRepo()
    app_module.app.dependency_overrides[get_app_settings] = get_settings
    try:
        client = TestClient(app_module.app)
        resp = client.get("/files/result/abc", follow_redirects=False)
        assert resp.status_code == 307
        assert resp.headers["location"] == "https://minio.example/presigned?sig=1"
    finally:
        app_module.app.dependency_overrides.clear()


def test_files_result_404_for_unknown_task() -> None:
    from fastapi.testclient import TestClient

    from service import app as app_module
    from service.dependencies import get_app_settings, get_task_repo

    class EmptyRepo:
        def get_by_external_id(self, external_id):
            return None

    app_module.app.dependency_overrides[get_task_repo] = lambda: EmptyRepo()
    app_module.app.dependency_overrides[get_app_settings] = get_settings
    try:
        client = TestClient(app_module.app)
        resp = client.get("/files/result/missing", follow_redirects=False)
        assert resp.status_code == 404
    finally:
        app_module.app.dependency_overrides.clear()
