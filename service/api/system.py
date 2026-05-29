"""System / operational endpoints (health, readiness, metrics, root)."""
from __future__ import annotations

import redis
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import RedirectResponse, Response

from ..db import session_scope
from ..dependencies import get_app_settings
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
