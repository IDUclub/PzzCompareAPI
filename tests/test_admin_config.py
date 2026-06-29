"""Tests for the runtime config override admin API and the env-sync core."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


# ── env-sync core (no DB) ────────────────────────────────────────────────────

def test_is_overridable_rules(monkeypatch):
    from service.infrastructure import config_runtime as cr

    monkeypatch.setenv("SOME_APP_PARAM", "1")
    assert cr.is_overridable("SOME_APP_PARAM") is True          # known + not denied
    assert cr.is_overridable("DATABASE_URL") is False           # credential (denied)
    assert cr.is_overridable("VLLM_API_KEY") is False           # credential (denied)
    assert cr.is_overridable("DEFINITELY_NOT_SET_XYZ") is False  # unknown key


def test_sync_env_apply_then_restore(monkeypatch):
    from service.infrastructure import config_runtime as cr

    monkeypatch.setenv("TUNE_ME", "orig")
    cr._applied.clear()
    cr._baseline.clear()

    assert cr._sync_env({"TUNE_ME": "new"}) is True
    assert os.environ["TUNE_ME"] == "new"

    # idempotent: same override -> no change
    assert cr._sync_env({"TUNE_ME": "new"}) is False

    # removed from store -> reverts to the original deployed value
    assert cr._sync_env({}) is True
    assert os.environ["TUNE_ME"] == "orig"


def test_sync_env_ignores_denied_keys(monkeypatch):
    from service.infrastructure import config_runtime as cr

    monkeypatch.setenv("DATABASE_URL", "postgres://real")
    cr._applied.clear()
    cr._baseline.clear()
    assert cr._sync_env({"DATABASE_URL": "postgres://evil"}) is False
    assert os.environ["DATABASE_URL"] == "postgres://real"


# ── admin auth gate ──────────────────────────────────────────────────────────

def _client():
    from fastapi.testclient import TestClient
    from service import app as app_module

    return TestClient(app_module.app), app_module.app


def test_admin_requires_token():
    from service.dependencies import get_app_settings

    client, app = _client()
    app.dependency_overrides[get_app_settings] = lambda: SimpleNamespace(admin_api_token="secret")
    try:
        # Token configured but no X-Admin-Token header -> rejected.
        resp = client.get("/admin/config/settings")
        assert resp.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_admin_disabled_when_token_unset():
    from service.dependencies import get_app_settings

    client, app = _client()
    app.dependency_overrides[get_app_settings] = lambda: SimpleNamespace(admin_api_token="")
    try:
        resp = client.get("/admin/config/settings", headers={"X-Admin-Token": "whatever"})
        assert resp.status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_admin_wrong_token_rejected():
    from service.dependencies import get_app_settings

    client, app = _client()
    app.dependency_overrides[get_app_settings] = lambda: SimpleNamespace(admin_api_token="secret")
    try:
        resp = client.get("/admin/config/settings", headers={"X-Admin-Token": "nope"})
        assert resp.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_admin_settings_masks_secrets():
    from service.dependencies import get_app_settings

    client, app = _client()
    app.dependency_overrides[get_app_settings] = lambda: SimpleNamespace(admin_api_token="secret")
    try:
        resp = client.get("/admin/config/settings", headers={"X-Admin-Token": "secret"})
        assert resp.status_code == 200
        data = resp.json()
        # database_url holds a password -> must be masked, never returned raw.
        if data.get("database_url"):
            assert data["database_url"] == "***"
        if data.get("vllm_api_key"):
            assert data["vllm_api_key"] == "***"
    finally:
        app.dependency_overrides.clear()
