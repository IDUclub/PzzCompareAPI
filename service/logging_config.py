from __future__ import annotations

import logging
import os
import sys
from contextvars import ContextVar, Token
from typing import Any

from loguru import logger

# Fields attached to every record logged while a Celery task runs, so /logs
# entries can be narrowed down to a single task (see bind_log_context).
CONTEXT_KEYS = ("task_id", "external_id", "celery_task_id", "task_name")

_log_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "log_context", default=None
)

# uvicorn access lines for probe endpoints (Prometheus scrape, Docker
# healthcheck) drowned everything else in /logs. Successful hits to these paths
# are dropped; failing ones (status >= 400) are still logged.
_QUIET_ACCESS_PATHS = frozenset(
    p.strip()
    for p in os.getenv("LOG_ACCESS_SKIP_PATHS", "/health,/metrics,/readiness").split(
        ","
    )
    if p.strip()
)


def bind_log_context(**fields: Any) -> Token:
    """Attach *fields* to every record logged from the current context.

    Returns a token for reset_log_context(). Celery's threads pool reuses
    threads across tasks, so the caller must reset when the task ends.
    """
    current = _log_context.get() or {}
    merged = {**current, **{k: v for k, v in fields.items() if v is not None}}
    return _log_context.set(merged)


def reset_log_context(token: Token) -> None:
    _log_context.reset(token)


def _patch_record(record: dict[str, Any]) -> None:
    extra = record["extra"]
    extra.setdefault("logger_name", record["name"])
    for key, value in (_log_context.get() or {}).items():
        extra.setdefault(key, value)
    extra["ctx"] = " ".join(
        f"{key}={extra[key]}" for key in CONTEXT_KEYS if extra.get(key) is not None
    )


def _stdout_format(record: dict[str, Any]) -> str:
    extra = record["extra"]
    fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[logger_name]} | "
    if extra.get("ctx"):
        fmt += "[{extra[ctx]}] "
    fmt += "{message}\n"
    if extra.get("exception_text"):
        # Traceback parsed out of the pipeline subprocess output.
        fmt += "{extra[exception_text]}\n"
    return fmt + "{exception}"


def _is_access_noise(record: logging.LogRecord) -> bool:
    if record.name != "uvicorn.access":
        return False
    # uvicorn.access args: (client_addr, method, full_path, http_version, status)
    args = record.args
    if not isinstance(args, tuple) or len(args) < 5:
        return False
    path = str(args[2]).split("?", 1)[0]
    try:
        status = int(args[4])
    except (TypeError, ValueError):
        return False
    return path in _QUIET_ACCESS_PATHS and status < 400


class _InterceptHandler(logging.Handler):
    """Route all stdlib logging records through loguru.

    This single handler replaces every handler on every stdlib logger, so
    Celery, SQLAlchemy, uvicorn, and our own code all pass through loguru
    and end up in every configured loguru sink (stdout + Redis).
    """

    def emit(self, record: logging.LogRecord) -> None:
        if _is_access_noise(record):
            return
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk up the call stack past logging internals so loguru reports the
        # actual call site rather than the logging module itself.
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # loguru's record["name"] is the caller's module (e.g. h11_impl); keep
        # the stdlib logger name (e.g. uvicorn.access) — that is what people
        # filter on.
        logger.opt(depth=depth, exception=record.exc_info).bind(
            logger_name=record.name
        ).log(level, record.getMessage())


def setup_logging(
    *, force: bool = False
) -> None:  # noqa: ARG001 — force kept for call-site compat
    from .log_sink import mark_sink_removed

    level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Remove all existing loguru handlers (idempotent — safe to call again).
    # This drops the Redis sink too; callers re-add it via setup_redis_sink.
    logger.remove()
    mark_sink_removed()
    logger.configure(patcher=_patch_record)
    logger.add(
        sys.stdout,
        level=level,
        format=_stdout_format,
        colorize=False,  # plain text so Docker / journald can parse it cleanly
    )

    # Replace all stdlib handlers with our single intercept handler so that
    # Celery, uvicorn, SQLAlchemy etc. go through loguru.
    intercept = _InterceptHandler()
    logging.root.setLevel(0)
    logging.root.handlers = [intercept]
    for name in list(logging.root.manager.loggerDict):
        log = logging.getLogger(name)
        log.handlers = [intercept]
        log.propagate = False
