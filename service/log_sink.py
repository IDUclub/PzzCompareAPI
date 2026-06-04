"""Redis log sink for loguru.

All containers share one Redis instance; each appends JSON log entries to the
same list key.  The API's GET /logs reads from that list.

Call ``setup_redis_sink(redis_url, service_name)`` once at startup — after
``setup_logging()`` — to wire up the sink.  The sink runs in a background
thread (loguru enqueue=True) and never blocks the caller.
"""
from __future__ import annotations

import json
import os
import traceback
from typing import Any

from loguru import logger


LOG_STREAM_KEY = "logs:stream"
_DEFAULT_MAX_ENTRIES = 10_000

_sink_installed = False


def _make_sink(redis_url: str, service_name: str, max_entries: int):
    """Return a loguru-compatible callable that writes to Redis."""
    import redis as _redis

    client = _redis.Redis.from_url(
        redis_url,
        socket_connect_timeout=1,
        socket_timeout=1,
        decode_responses=False,
    )

    def _sink(message: Any) -> None:
        record = message.record
        entry: dict[str, Any] = {
            "ts": record["time"].isoformat(),
            "level": record["level"].name,
            "service": service_name,
            "logger": record["name"],
            "message": record["message"],
        }
        if record["exception"] is not None:
            entry["exception"] = "".join(
                traceback.format_exception(*record["exception"])
            ).strip()

        try:
            raw = json.dumps(entry, ensure_ascii=False).encode()
            pipe = client.pipeline(transaction=False)
            pipe.lpush(LOG_STREAM_KEY, raw)
            pipe.ltrim(LOG_STREAM_KEY, 0, max_entries - 1)
            pipe.execute()
        except Exception:  # noqa: BLE001
            # Never crash the application because logging failed.
            pass

    return _sink


def setup_redis_sink(
    redis_url: str | None = None,
    service_name: str | None = None,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
) -> None:
    """Attach a Redis sink to loguru.  No-op when redis_url is empty."""
    global _sink_installed  # noqa: PLW0603
    if _sink_installed:
        return
    url = redis_url or os.getenv("REDIS_URL", "")
    if not url:
        return
    name = service_name or os.getenv("SERVICE_NAME", "unknown")
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    sink = _make_sink(url, name, max_entries)
    # enqueue=True → background thread; catch=True → swallow sink exceptions
    logger.add(sink, level=level, enqueue=True, catch=True)
    _sink_installed = True
