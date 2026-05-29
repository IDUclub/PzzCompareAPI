"""Shared HTTP-layer utilities (structured logging)."""
from __future__ import annotations

import json
import logging

logger = logging.getLogger("service.app")


def api_log(stage: str, status: str, **extra: object) -> None:
    """Emit a structured single-line JSON log record for an API event.

    Keeps log output greppable by ``stage`` / ``status`` and consistent
    across all endpoints. ``task_id`` and ``external_id`` are first-class
    fields; everything else goes into ``extra``.
    """
    payload = {
        "task_id": extra.pop("task_id", None),
        "external_id": extra.pop("external_id", None),
        "celery_task_id": None,
        "stage": stage,
        "status": status,
        "duration_ms": extra.pop("duration_ms", None),
        **extra,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))
