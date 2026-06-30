"""Runtime config overrides — a shared layer applied on top of the deployed env.

Overrides live in the ``config_override`` table (durable, shared by every
container). Each process periodically syncs them into its **own** ``os.environ``
and busts the Settings cache, so ``api`` / ``worker`` / ``worker_llm`` and the
pipeline subprocess (which copies ``os.environ``) all pick up changes within
``SYNC_TTL`` seconds — no redeploy.

Safety:
- credentials and boot-only keys are never overridable (``_DENY``);
- only keys that already exist in the process env can be overridden (you tune
  known config, you don't inject arbitrary variables);
- any DB hiccup is swallowed — a failing override layer must never break the app
  or the pipeline; it simply falls back to the deployed env.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

from sqlalchemy import delete, select

from ..models import ConfigOverride

# How long a process trusts its last sync before re-reading the DB.
SYNC_TTL = 5.0

# Never overridable at runtime: credentials + keys only read at process boot.
_DENY: frozenset[str] = frozenset(
    {
        # credentials
        "DATABASE_URL",
        "VLLM_API_KEY",
        "URBAN_API_TOKEN",
        "FILESERVER_ACCESS_KEY",
        "FILESERVER_SECRET_KEY",
        "ADMIN_API_TOKEN",
        # boot-only infra (a live change here can't take effect without a restart)
        "REDIS_URL",
        "APP_ENV",
        "DB_WAIT_TIMEOUT",
        "DB_WAIT_INTERVAL",
        "RUN_MIGRATIONS_ON_STARTUP",
    }
)

_lock = threading.RLock()
_last_sync = 0.0
# key -> value currently written into os.environ by us
_applied: dict[str, str] = {}
# key -> the env value before we first overrode it (None == key was absent)
_baseline: dict[str, str | None] = {}


def is_overridable(key: str) -> bool:
    """A key is tunable at runtime only if it's known config and not sensitive."""
    return key not in _DENY and key in os.environ


def _load_from_db() -> dict[str, str]:
    """Read all overrides from the DB. Imports ``session_scope`` lazily to avoid
    an import cycle (``db`` builds its engine via ``get_settings`` at import)."""
    from ..db import session_scope  # local: db imports settings at module load

    with session_scope() as session:
        rows = session.execute(select(ConfigOverride)).scalars().all()
        return {r.key: r.value for r in rows}


def _sync_env(overrides: dict[str, str]) -> bool:
    """Reconcile ``os.environ`` with ``overrides``. Returns True if anything moved."""
    changed = False
    for key, value in overrides.items():
        if key in _DENY:
            continue  # ignore rows that somehow target a protected key
        if _applied.get(key) != value:
            if key not in _baseline:
                _baseline[key] = os.environ.get(key)
            os.environ[key] = value
            _applied[key] = value
            changed = True
    # Overrides that disappeared from the DB -> restore the deployed value.
    for key in list(_applied):
        if key not in overrides:
            base = _baseline.pop(key, None)
            if base is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = base
            _applied.pop(key, None)
            changed = True
    return changed


def apply_overrides(force: bool = False) -> bool:
    """Sync this process's env from the override store (TTL-gated). Returns True
    if the env changed (callers may use this to invalidate derived caches)."""
    global _last_sync
    now = time.monotonic()
    if not force and (now - _last_sync) < SYNC_TTL:
        return False
    with _lock:
        now = time.monotonic()
        if not force and (now - _last_sync) < SYNC_TTL:
            return False
        try:
            overrides = _load_from_db()
        except Exception:  # noqa: BLE001 — never let config break the app
            _last_sync = now
            return False
        changed = _sync_env(overrides)
        _last_sync = now
    if changed:
        # Lazy import to avoid a cycle; settings is fully loaded by now.
        from ..settings import _build_settings_cached

        _build_settings_cached.cache_clear()
    return changed


# ── admin-API helpers ────────────────────────────────────────────────────────

def list_overrides() -> list[dict[str, Any]]:
    """All active overrides as plain dicts (for the admin view)."""
    from ..db import session_scope

    with session_scope() as session:
        rows = session.execute(
            select(ConfigOverride).order_by(ConfigOverride.key)
        ).scalars().all()
        return [
            {
                "key": r.key,
                "value": r.value,
                "updated_at": r.updated_at,
                "updated_by": r.updated_by,
            }
            for r in rows
        ]


def set_override(key: str, value: str, updated_by: str | None = None) -> None:
    """Upsert one override and immediately re-sync this process."""
    from ..db import session_scope
    from ..time_utils import utc_now

    with session_scope() as session:
        row = session.get(ConfigOverride, key)
        if row is None:
            session.add(
                ConfigOverride(key=key, value=value, updated_by=updated_by)
            )
        else:
            row.value = value
            row.updated_by = updated_by
            row.updated_at = utc_now()
    apply_overrides(force=True)


def delete_override(key: str) -> bool:
    """Remove one override (reverting to the deployed value). Returns True if a
    row existed."""
    from ..db import session_scope

    with session_scope() as session:
        result = session.execute(
            delete(ConfigOverride).where(ConfigOverride.key == key)
        )
        existed = bool(result.rowcount)
    apply_overrides(force=True)
    return existed
