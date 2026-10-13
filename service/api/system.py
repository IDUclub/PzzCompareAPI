"""System / operational endpoints (health, readiness, metrics, logs, root)."""

from __future__ import annotations

import json
import logging
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
logger = logging.getLogger("service.api.system")

# Connection errors carry the broker/Redis URL, credentials included, and
# /readiness is unauthenticated.
BROKER_UNAVAILABLE_MESSAGE = "broker unavailable"
REDIS_UNAVAILABLE_MESSAGE = "Redis unavailable"


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
        logger.exception("Broker ping failed")
        raise HTTPException(
            status_code=503, detail=BROKER_UNAVAILABLE_MESSAGE
        ) from exc
    return {"status": "ready"}


@router.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


_LEVEL_RANK = {
    "TRACE": 5,
    "DEBUG": 10,
    "INFO": 20,
    "SUCCESS": 25,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}
# The stream is trimmed to this many entries (see log_sink).
_LOG_SCAN_MAX = 10_000
_LOG_SCAN_CHUNK = 1_000


def _level_rank(level: str | None) -> int:
    return _LEVEL_RANK.get((level or "").upper(), 0)


@router.get("/logs", response_model=list[dict[str, Any]])
def get_logs(
    limit: int = Query(200, ge=1, le=2000, description="Max entries to return"),
    level: str | None = Query(
        None,
        description="Exact level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    ),
    min_level: str | None = Query(
        None, description="This level and above, e.g. WARNING → warnings + errors"
    ),
    errors_only: bool = Query(
        False, description="Only ERROR/CRITICAL entries and entries with a traceback"
    ),
    service: str | None = Query(
        None, description="Service: api, worker, worker_llm, beat, mcp"
    ),
    logger_name: str | None = Query(
        None,
        alias="logger",
        description="Logger name prefix, e.g. celery, pipeline_modules, service.tasks",
    ),
    task_id: str | None = Query(None, description="Our task id"),
    external_id: str | None = Query(None, description="Task external id"),
    celery_task_id: str | None = Query(None, description="Celery task id"),
    q: str | None = Query(
        None, description="Case-insensitive substring of message or traceback"
    ),
    app_settings: Settings = Depends(get_app_settings),
) -> list[dict[str, Any]]:
    """Return recent aggregated logs from all services, newest first.

    Every container appends to one Redis list (capped at 10 000 entries), so
    each entry names its *service*, *logger* and code location (*where*);
    entries logged while a task runs also carry *task_id*, *external_id*,
    *celery_task_id* and *task_name*, and failures carry the full traceback
    in *exception*. Combine the filters to narrow down what you see.
    """
    level_upper = level.upper() if level else None
    min_rank = _level_rank(min_level)
    q_lower = q.lower() if q else None
    context_filters = {
        key: value
        for key, value in (
            ("task_id", task_id),
            ("external_id", external_id),
            ("celery_task_id", celery_task_id),
        )
        if value
    }
    filtering = bool(
        level_upper
        or min_rank
        or errors_only
        or service
        or logger_name
        or context_filters
        or q_lower
    )

    def _matches(entry: dict[str, Any]) -> bool:
        entry_level = str(entry.get("level", "")).upper()
        if level_upper and entry_level != level_upper:
            return False
        if min_rank and _level_rank(entry_level) < min_rank:
            return False
        if errors_only and not (
            _level_rank(entry_level) >= _LEVEL_RANK["ERROR"] or entry.get("exception")
        ):
            return False
        if service and entry.get("service") != service:
            return False
        if logger_name and not str(entry.get("logger", "")).startswith(logger_name):
            return False
        for key, value in context_filters.items():
            if str(entry.get(key, "")) != value:
                return False
        if q_lower:
            haystack = f"{entry.get('message', '')}\n{entry.get('exception', '')}"
            if q_lower not in haystack.lower():
                return False
        return True

    r = redis.Redis.from_url(
        app_settings.redis_url, socket_connect_timeout=2, socket_timeout=2
    )
    # Unfiltered: just the newest `limit`. Filtered: scan the whole stream in
    # chunks, stopping as soon as `limit` matches are found.
    scan_max = _LOG_SCAN_MAX if filtering else limit
    chunk = _LOG_SCAN_CHUNK if filtering else limit
    result: list[dict[str, Any]] = []
    start = 0
    while start < scan_max and len(result) < limit:
        stop = min(start + chunk, scan_max) - 1
        try:
            raw_entries = r.lrange(LOG_STREAM_KEY, start, stop)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Reading the aggregated log stream from Redis failed")
            raise HTTPException(
                status_code=503, detail=REDIS_UNAVAILABLE_MESSAGE
            ) from exc
        for raw in raw_entries:
            try:
                entry: dict[str, Any] = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            if _matches(entry):
                result.append(entry)
                if len(result) >= limit:
                    break
        if len(raw_entries) < stop - start + 1:
            break
        start = stop + 1
    return result
