"""Shared HTTP-layer utilities (structured logging, upload streaming)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import HTTPException, UploadFile

logger = logging.getLogger("service.app")

# Field names are the multipart keys; the message goes to an end user, so name the
# layer the way the UI does.
_FIELD_TITLES = {
    "cadastral_feature_collection_file": "слой земельных участков",
    "buildings_feature_collection_file": "слой зданий и сервисов",
    "pzz_zones_feature_collection_file": "слой зон ПЗЗ",
    "pzz_zone_vri_labels_file": "описания зон ПЗЗ",
    "pzz_descriptions_file": "описания зон ПЗЗ",
    "vri_classifier_file": "классификатор ВРИ",
    "mo_boundaries_feature_collection_file": "слой границ МО",
    "file": "файл",
}


def field_title(field_name: str) -> str:
    """The upload slot as the UI names it, falling back to the multipart key."""
    return _FIELD_TITLES.get(field_name, field_name)


def upload_too_large_error(
    field_name: str, max_bytes: int, filename: str | None = None
) -> HTTPException:
    """413 worded for an end user: which layer, which file, what the limit is."""
    title = field_title(field_name)
    where = f"{title} («{filename}»)" if filename else title
    limit_mb = max_bytes / (1024 * 1024)
    limit = f"{limit_mb:.0f} МБ" if limit_mb >= 10 else f"{limit_mb:.1f} МБ"
    return HTTPException(
        status_code=413,
        detail=f"{where}: файл больше допустимого размера {limit}.",
    )


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
                raise upload_too_large_error(field_name, max_bytes, upload.filename)
            fh.write(chunk)


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
