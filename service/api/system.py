"""System / operational endpoints (health, readiness, metrics, logs, root)."""
from __future__ import annotations

import json
from typing import Any

import redis
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import RedirectResponse, Response

from ..db import session_scope
from ..dependencies import get_app_settings
from ..log_sink import LOG_STREAM_KEY
from ..settings import Settings

router = APIRouter(tags=["system"])


@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readiness")
def readiness(app_settings: Settings = Depends(get_app_settings)) -> dict[str, str]:
    """Verify DB + broker are reachable. Returns 503 on any failure."""
    with session_scope() as session:
        session.execute(text("SELECT 1"))
    try:
        broker = redis.Redis.from_url(app_settings.redis_url, socket_connect_timeout=2)
        broker.ping()
        broker.close()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"broker unavailable: {exc}") from exc
    return {"status": "ready"}


@router.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/logs", response_model=list[dict[str, Any]])
def get_logs(
    limit: int = Query(200, ge=1, le=2000, description="Max entries to return"),
    level: str | None = Query(None, description="Filter by level: DEBUG, INFO, WARNING, ERROR, CRITICAL"),
    service: str | None = Query(None, description="Filter by service: api, worker, beat, mcp"),
    app_settings: Settings = Depends(get_app_settings),
) -> list[dict[str, Any]]:
    """Return recent aggregated logs from all services, newest first.

    Logs are stored in Redis by every running container.  Use *level* and
    *service* to narrow down what you see.  The list is capped at 10 000
    entries across all services combined.
    """
    r = redis.Redis.from_url(app_settings.redis_url, socket_connect_timeout=2, socket_timeout=2)
    # When filtering, fetch more records so we can satisfy `limit` after filtering.
    fetch_n = min(limit * 10, 10_000) if (level or service) else limit
    try:
        raw_entries = r.lrange(LOG_STREAM_KEY, 0, fetch_n - 1)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Redis unavailable: {exc}") from exc

    result: list[dict[str, Any]] = []
    level_upper = level.upper() if level else None
    for raw in raw_entries:
        try:
            entry: dict[str, Any] = json.loads(raw)
        except Exception:  # noqa: BLE001
            continue
        if level_upper and entry.get("level", "").upper() != level_upper:
            continue
        if service and entry.get("service") != service:
            continue
        result.append(entry)
        if len(result) >= limit:
            break
    return result
