"""Pre-task file uploads: store the bytes once, reference them by id afterwards.

An MCP tool cannot carry a multi-megabyte GeoJSON as an argument — the model would
have to emit it — so an agent uploads the file first and submits the returned id. The
same id also spares the caller re-sending a layer when a run is repeated with different
parameters.

Uploads live outside the per-task input tree because they exist *before* a task does:
``task_inputs/{external_id}/`` is named after an id that submission has yet to mint.

They stay on the local filesystem rather than going through ``ObjectStorage``: an upload
is short-lived scratch data that is copied into the task's own inputs on submission, and
the storage abstraction addresses objects by the path it returns, not by a key one can
reconstruct from an id. The same caveat as ``task_inputs`` applies — with several API
replicas the directory has to be shared, as it already is in compose.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from ...settings import Settings

logger = logging.getLogger("service.uploads")

_META_NAME = "meta.json"
_PAYLOAD_FALLBACK = "payload.bin"


class UploadError(Exception):
    """An upload could not be stored or resolved."""

    def __init__(self, detail: str, *, status_code: int = 400) -> None:
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


@dataclass(frozen=True)
class StoredUpload:
    """What the caller gets back and later passes as ``*_upload_id``."""

    upload_id: str
    filename: str
    size: int
    content_type: str
    created_at: float
    expires_at: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "upload_id": self.upload_id,
            "filename": self.filename,
            "size": self.size,
            "content_type": self.content_type,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


def uploads_root(settings: Settings) -> Path:
    return Path(settings.uploads_dir)


def new_upload_dir(settings: Settings) -> tuple[str, Path]:
    """Reserve an id and its directory so the caller can stream straight into it."""
    upload_id = uuid4().hex
    directory = uploads_root(settings) / upload_id
    directory.mkdir(parents=True, exist_ok=True)
    return upload_id, directory


def register_upload(
    *,
    upload_id: str,
    payload_path: Path,
    content_type: str,
    owner_id: str,
    settings: Settings,
) -> StoredUpload:
    """Write the metadata that makes an already-streamed file referencable."""
    now = time.time()
    record = StoredUpload(
        upload_id=upload_id,
        filename=payload_path.name,
        size=payload_path.stat().st_size,
        content_type=content_type or "application/octet-stream",
        created_at=now,
        expires_at=now + settings.uploads_max_age_hours * 3600,
    )
    meta = {**record.as_dict(), "owner_id": owner_id}
    (payload_path.parent / _META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "upload stored | id=%s | size=%s | owner=%s", upload_id, record.size, owner_id
    )
    return record


def _read_meta(upload_id: str, settings: Settings) -> tuple[dict[str, Any], Path]:
    root = uploads_root(settings).resolve()
    directory = (root / upload_id).resolve()
    if directory.parent != root or not directory.is_dir():
        raise UploadError(f"Unknown upload_id: {upload_id}", status_code=404)
    try:
        meta = json.loads((directory / _META_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UploadError(f"Unknown upload_id: {upload_id}", status_code=404) from exc
    if not isinstance(meta, dict):
        raise UploadError(f"Unknown upload_id: {upload_id}", status_code=404)
    return meta, directory


def purge_expired_uploads(settings: Settings) -> int:
    """Delete uploads whose TTL has passed. Returns how many were removed.

    Needed because the directory is a volume: it outlives the container by design, so
    nothing reclaims it on its own, and every file a user ever attached would stay.
    An unreadable or half-written directory is removed too — it cannot be resolved
    anyway, and leaving it would keep it forever.
    """
    root = uploads_root(settings)
    if not root.is_dir():
        return 0

    now = time.time()
    removed = 0
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        try:
            meta = json.loads((directory / _META_NAME).read_text(encoding="utf-8"))
            expires_at = float(meta["expires_at"])
        except (OSError, ValueError, KeyError, TypeError):
            # No readable metadata: keep it while it could still be an upload in
            # flight, drop it once it is older than any TTL would have allowed.
            age_limit = settings.uploads_max_age_hours * 3600
            if now - directory.stat().st_mtime < age_limit:
                continue
            expires_at = 0.0

        if expires_at and expires_at > now:
            continue
        shutil.rmtree(directory, ignore_errors=True)
        removed += 1

    if removed:
        logger.info("Removed %s expired uploads from %s", removed, root)
    return removed


def describe_upload(
    upload_id: str, *, owner_id: str, settings: Settings
) -> tuple[StoredUpload, Path]:
    """Return an upload's metadata and file, refusing a foreign or expired one.

    Ownership is checked here rather than at each call site so every entry point gets
    it: an upload id travels through URLs, logs and chat messages, and without the
    check it would be a handle to another user's data.
    """
    meta, directory = _read_meta(upload_id, settings)

    stored_owner = str(meta.get("owner_id") or "")
    if owner_id and stored_owner and stored_owner != owner_id:
        raise UploadError("Upload belongs to another user", status_code=403)

    expires_at = meta.get("expires_at")
    if isinstance(expires_at, (int, float)) and expires_at < time.time():
        raise UploadError(f"Upload {upload_id} has expired", status_code=410)

    filename = str(meta.get("filename") or _PAYLOAD_FALLBACK)
    payload = directory / filename
    if not payload.is_file():
        raise UploadError(f"Upload {upload_id} is no longer stored", status_code=410)

    record = StoredUpload(
        upload_id=upload_id,
        filename=filename,
        size=int(meta.get("size") or payload.stat().st_size),
        content_type=str(meta.get("content_type") or "application/octet-stream"),
        created_at=float(meta.get("created_at") or 0.0),
        expires_at=float(expires_at or 0.0),
    )
    return record, payload


def resolve_upload(upload_id: str, *, owner_id: str, settings: Settings) -> Path:
    """Return the stored file for a submission that references it by id."""
    return describe_upload(upload_id, owner_id=owner_id, settings=settings)[1]
