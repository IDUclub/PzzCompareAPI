"""Admin API for runtime config overrides — ``/admin/config/*``.

View / set / delete app params and service URLs live (no redeploy). Overrides
are persisted to a shared store and synced into every process's env within a few
seconds (see ``infrastructure.config_runtime``).

Guarded by the ``ADMIN_API_TOKEN`` shared secret, passed as the ``X-Admin-Token``
header. Credentials and boot-only keys are never editable or shown unmasked.
"""
from __future__ import annotations

import hmac
import os
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from ..dependencies import get_app_settings
from ..infrastructure import config_runtime as cr
from ..infrastructure.config_runtime import (
    apply_overrides,
    delete_override,
    is_overridable,
    list_overrides,
    set_override,
)
from ..settings import Settings, _build_settings_cached, get_settings
from .utils import api_log


def _validation_error(key: str, value: str) -> str | None:
    """Try the candidate value without persisting it; return an error or None.

    Applies the value to this process's env transiently, rebuilds the typed
    Settings (which raises on a bad value, e.g. ``TOP_K=abc``), then restores —
    so a rejected update never touches the store or clobbers an existing
    override. Serialised against the override sync via the runtime lock.
    """
    sentinel = object()
    with cr._lock:
        old = os.environ.get(key, sentinel)
        os.environ[key] = value
        try:
            _build_settings_cached.cache_clear()
            _build_settings_cached()
            return None
        except Exception as exc:  # noqa: BLE001
            return str(exc)
        finally:
            if old is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
            _build_settings_cached.cache_clear()

router = APIRouter(prefix="/admin/config", tags=["admin"])

# Settings attributes that hold secrets — masked in the resolved-settings view.
_SECRET_SETTING_FIELDS = frozenset(
    {"vllm_api_key", "fileserver_access_key", "fileserver_secret_key",
     "admin_api_token", "database_url"}
)


def verify_admin(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    settings: Settings = Depends(get_app_settings),
) -> bool:
    """Gate every admin-config route on the shared ADMIN_API_TOKEN secret."""
    expected = settings.admin_api_token
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Admin config API is disabled (ADMIN_API_TOKEN not set)",
        )
    if not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Token")
    return True


class ConfigValueIn(BaseModel):
    value: str = Field(..., description="New string value for the env key")
    updated_by: str | None = Field(default=None, description="Optional audit label")


def _key_view(key: str) -> dict[str, Any]:
    overrides = {o["key"]: o for o in list_overrides()}
    ov = overrides.get(key)
    return {
        "key": key,
        "effective": os.environ.get(key),
        "overridden": ov is not None,
        "override_value": ov["value"] if ov else None,
        "overridable": is_overridable(key),
        "updated_at": ov["updated_at"] if ov else None,
        "updated_by": ov["updated_by"] if ov else None,
    }


@router.get("/settings", dependencies=[Depends(verify_admin)])
def get_resolved_settings() -> dict[str, Any]:
    """Resolved, typed application settings the app is actually using (secrets masked)."""
    data = get_settings().model_dump()
    for field in _SECRET_SETTING_FIELDS:
        if data.get(field):
            data[field] = "***"
    return data


@router.get("/overrides", dependencies=[Depends(verify_admin)])
def get_active_overrides() -> dict[str, Any]:
    """Only the keys currently overridden (with audit metadata)."""
    items = list_overrides()
    return {"count": len(items), "overrides": items}


@router.post("/reload", dependencies=[Depends(verify_admin)])
def reload_overrides() -> dict[str, Any]:
    """Force this process to re-sync overrides from the store immediately."""
    changed = apply_overrides(force=True)
    api_log("admin_config", "reload", changed=changed)
    return {"reloaded": True, "changed": changed}


@router.get("/{key}", dependencies=[Depends(verify_admin)])
def get_config_key(key: str) -> dict[str, Any]:
    """Effective value + override status for a single env key."""
    return _key_view(key)


@router.put("/{key}", dependencies=[Depends(verify_admin)])
def put_config_key(key: str, body: ConfigValueIn) -> dict[str, Any]:
    """Set (or replace) a runtime override for one key.

    Rejects credentials / boot-only keys and unknown keys. The new value is
    validated by rebuilding Settings; on a parse error the override is reverted
    and 400 is returned, so a bad value can't wedge the service.
    """
    if not is_overridable(key):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Key '{key}' is not overridable "
                "(unknown env key, a credential, or boot-only)."
            ),
        )
    error = _validation_error(key, body.value)  # pre-check, never persists a bad value
    if error is not None:
        raise HTTPException(status_code=400, detail=f"Value rejected for '{key}': {error}")
    set_override(key, body.value, updated_by=body.updated_by)
    api_log("admin_config", "set", key=key, updated_by=body.updated_by or "")
    return _key_view(key)


@router.delete("/{key}", dependencies=[Depends(verify_admin)])
def delete_config_key(key: str) -> dict[str, Any]:
    """Remove a runtime override, reverting the key to its deployed value."""
    existed = delete_override(key)
    if not existed:
        raise HTTPException(status_code=404, detail=f"No override set for '{key}'")
    api_log("admin_config", "delete", key=key)
    return _key_view(key)
