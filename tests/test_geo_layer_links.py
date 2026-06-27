"""Tests for geo-layer download links: descriptor + /files redirect (phase 8)."""
from types import SimpleNamespace

import service.api.tasks as tasks_mod
from service.api.tasks import (
    build_input_geo_layers,
    build_result_geo_layer,
    geo_layer_to_file_part,
)
from service.settings import get_settings


def _task(
    status="finished",
    result_path="outputs/abc/result.geojson",
    cadastral_data_path="minio://inputs/abc/cadastral_feature_collection.geojson",
    pzz_zones_data_path="minio://inputs/abc/pzz_zones_feature_collection.geojson",
):
    return SimpleNamespace(
        status=status,
        result_path=result_path,
        cadastral_data_path=cadastral_data_path,
        pzz_zones_data_path=pzz_zones_data_path,
    )


def test_layer_descriptor_local_result_relative_url() -> None:
    settings = get_settings()  # public_base_url empty in tests
    layer = build_result_geo_layer(_task(), "abc123", settings)
    assert layer is not None
    assert layer["url"] == "/files/result/abc123"
    assert layer["download_url"] is None  # local storage can't presign
    assert layer["filename"] == "abc123.geojson"
    assert layer["mime_type"] == "application/geo+json"


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
    assert part["mime_type"] == "application/geo+json"


def test_input_layers_for_uploaded_files() -> None:
    settings = get_settings()
    layers = build_input_geo_layers(_task(), "abc123", settings)
    by_name = {layer["name"]: layer for layer in layers}
    assert set(by_name) == {"input_cadastral", "input_zones"}
    assert by_name["input_cadastral"]["url"] == "/files/cadastral/abc123"
    assert by_name["input_zones"]["url"] == "/files/zones/abc123"
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
