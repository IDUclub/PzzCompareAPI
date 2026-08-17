"""Upload a file once, reference it by id when submitting a task.

Separate from the classifier router because an upload is not a task: it exists before
one, and the same upload can feed several runs. This is what lets an MCP client submit
a pipeline task at all — a tool argument cannot carry a multi-megabyte GeoJSON.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from ..application.use_cases.uploads import (
    UploadError,
    describe_upload,
    new_upload_dir,
    register_upload,
)
from ..dependencies import get_app_settings
from ..settings import Settings
from .security import AuthUser, get_current_user
from .utils import durable_url, stream_upload_to_file

router = APIRouter(prefix="/uploads", tags=["uploads"])
logger = logging.getLogger("service.api.uploads")


@router.post("", status_code=201)
def create_upload(
    request: Request,
    file: UploadFile = File(...),
    user: AuthUser = Depends(get_current_user),
    app_settings: Settings = Depends(get_app_settings),
) -> dict[str, object]:
    """Store a file and return the id to pass as ``*_upload_id`` on submission.

    The payload is streamed to disk under the configured size limit and never held in
    memory. Its contents are validated at submission, where the schema expected for
    that particular slot is known.

    ``url`` is the durable link for a client that keeps a record of what was attached;
    it resolves back here and, like the upload itself, only for its owner.
    """
    upload_id, directory = new_upload_dir(app_settings)
    payload_path = directory / Path(file.filename or "payload.bin").name
    try:
        stream_upload_to_file(file, payload_path, app_settings.max_upload_bytes, "file")
        record = register_upload(
            upload_id=upload_id,
            payload_path=payload_path,
            content_type=file.content_type or "",
            owner_id=user.user_id,
            settings=app_settings,
        )
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return {
        **record.as_dict(),
        "url": durable_url(
            f"/uploads/{upload_id}", app_settings.public_base_url, request
        ),
    }


@router.get("/{upload_id}")
def download_upload(
    upload_id: str,
    user: AuthUser = Depends(get_current_user),
    app_settings: Settings = Depends(get_app_settings),
) -> FileResponse:
    """Return a stored upload to its owner.

    A token is required even though submission accepts anonymous callers: submitting
    consumes the id inside the service, while this hands the bytes to whoever holds it,
    which would make the id a capability over another user's data.
    """
    try:
        record, payload = describe_upload(
            upload_id, owner_id=user.user_id, settings=app_settings
        )
    except UploadError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return FileResponse(
        payload, media_type=record.content_type, filename=record.filename
    )
