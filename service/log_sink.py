"""Redis log sink for loguru.

All containers share one Redis instance; each appends JSON log entries to the
same list key.  The API's GET /logs reads from that list.

Call ``setup_redis_sink(redis_url, service_name)`` at startup — after
``setup_logging()``, which removes every loguru sink including this one — to
wire up the sink.  The sink runs in a background thread (loguru enqueue=True)
and never blocks the caller.

Entry shape::

    {"ts", "level", "service", "host", "logger", "where", "message",
     "task_id"?, "external_id"?, "celery_task_id"?, "task_name"?, "exception"?}
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any

from loguru import logger

from .logging_config import CONTEXT_KEYS

LOG_STREAM_KEY = "logs:stream"
_DEFAULT_MAX_ENTRIES = 10_000

_sink_installed = False


def mark_sink_removed() -> None:
    """Called by setup_logging() after logger.remove() dropped the sink."""
    global _sink_installed  # noqa: PLW0603
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
    host = socket.gethostname()

    def _sink(message: Any) -> None:
        record = message.record
        extra = record["extra"]
        entry: dict[str, Any] = {
            "ts": record["time"].isoformat(),
            "level": record["level"].name,
            "service": service_name,
            "host": host,
            "logger": extra.get("logger_name") or record["name"],
            "where": f"{record['name']}:{record['function']}:{record['line']}",
            "message": record["message"],
        }
        for key in CONTEXT_KEYS:
            if extra.get(key) is not None:
                entry[key] = extra[key]
        # With enqueue=True loguru strips the traceback object before the
        # record reaches this thread (tracebacks are not picklable), so
        # record["exception"] holds only the last line. The sink is added with
        # format="", which makes the message exactly "\n{exception}" —
        # formatted in the caller thread, so it is the full traceback.
        exception_text = str(message).strip() or extra.get("exception_text")
        if exception_text:
            entry["exception"] = exception_text

        try:
            raw = json.dumps(entry, ensure_ascii=False, default=str).encode()
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
    # enqueue=True → background thread; catch=True → swallow sink exceptions.
    # format="" → see the exception comment in _sink. diagnose=False keeps
    # local variable values (tokens, paths) out of an HTTP-readable log.
    logger.add(
        sink,
        level=level,
        format="",
        enqueue=True,
        catch=True,
        backtrace=False,
        diagnose=False,
    )
    _sink_installed = True
