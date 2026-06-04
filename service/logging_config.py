from __future__ import annotations

import logging
import os
import sys

from loguru import logger


class _InterceptHandler(logging.Handler):
    """Route all stdlib logging records through loguru.

    This single handler replaces every handler on every stdlib logger, so
    Celery, SQLAlchemy, uvicorn, and our own code all pass through loguru
    and end up in every configured loguru sink (stdout + Redis).
    """

    def emit(self, record: logging.LogRecord) -> None:
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

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(*, force: bool = False) -> None:  # noqa: ARG001 — force kept for call-site compat
    level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Remove all existing loguru handlers (idempotent — safe to call again).
    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name} | "
            "{message}"
        ),
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
