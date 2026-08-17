"""Shared HTTP-layer utilities (structured logging, upload streaming)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import HTTPException, Request, UploadFile

logger = logging.getLogger("service.app")


def stream_upload_to_file(
    upload: UploadFile,
    dest: Path,
    max_bytes: int,
    field_name: str,
) -> None:
    """Stream ``upload`` chunk-by-chunk to ``dest``, enforcing ``max_bytes``.

    Avoids buffering the full payload in memory.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dest.open("wb") as fh:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"{field_name} exceeds limit of {max_bytes} bytes",
                )
            fh.write(chunk)


def durable_url(path: str, public_base_url: str, request: Request | None = None) -> str:
    """Stable, never-expiring URL for a stored file.

    Absolute when ``PUBLIC_BASE_URL`` is set (what a link kept in chat history needs),
    otherwise derived from the request, else a relative path.
    """
    if public_base_url:
        return f"{public_base_url}{path}"
    if request is not None:
        return str(request.base_url).rstrip("/") + path
    return path


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
