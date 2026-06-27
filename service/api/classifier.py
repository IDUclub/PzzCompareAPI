"""Task submission endpoints (pzz-check, classify-only) and their helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from ..application.use_cases.create_task import create_task
from ..dependencies import get_app_settings, get_db, get_event_repo, get_task_repo
from ..domain.ports.event_repository import EventRepository
from ..domain.ports.task_repository import TaskRepository
from ..infrastructure.geo_ingest import (
    GeoIngestError,
    geo_file_to_geojson_dict,
    is_geojson_filename,
    supported_extensions,
)
from ..infrastructure.storage import get_object_storage
from ..schemas import TaskCreate, TaskOut
from ..settings import Settings
from ..tasks import celery_app, enqueue_pipeline_task, execute_pipeline_task
from .security import verify_token
from .tasks import task_stream_with_chat_generator, task_stream_with_report_generator
from .utils import api_log

router = APIRouter(prefix="/tasks", tags=["classifier"])


def _stream_upload_to_file(
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


def _validate_json_file(
    path: Path,
    expected_type: type[Any] | tuple[type[Any], ...],
    field_name: str,
) -> None:
    """Load ``path`` as JSON and assert its top-level type."""
    try:
        with path.open("rb") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        path.unlink(missing_ok=True)
        api_log("create_task", "invalid_json", field=field_name)
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must contain valid JSON/GeoJSON",
        ) from exc
    if not isinstance(data, expected_type):
        expected_type_name = (
            ", ".join(t.__name__ for t in expected_type)
            if isinstance(expected_type, tuple)
            else expected_type.__name__
        )
        path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be a JSON {expected_type_name}",
        )


def _ingest_upload(
    upload: UploadFile,
    task_dir: Path,
    filename: str,
    expected_json_type: type[Any] | tuple[type[Any], ...],
    field_name: str,
    max_bytes: int,
    external_id: str,
    storage,
) -> str:
    """Stream → validate → persist to storage. Returns the stored path."""
    local_path = task_dir / filename
    _stream_upload_to_file(upload, local_path, max_bytes, field_name)
    _validate_json_file(local_path, expected_json_type, field_name)
    object_key = f"inputs/{external_id}/{filename}"
    stored = storage.upload_file(str(local_path.resolve()), object_key)
    if storage.is_remote():
        local_path.unlink(missing_ok=True)
    return stored


def _ingest_geo_upload(
    upload: UploadFile,
    task_dir: Path,
    filename: str,
    field_name: str,
    max_bytes: int,
    external_id: str,
    storage,
) -> str:
    """Ingest a geo upload, converting non-GeoJSON formats to GeoJSON first.

    GeoJSON (``.geojson`` / ``.json`` / no extension) takes the existing
    stream → validate-as-dict → persist path. Other vector formats
    (GeoPackage, GML, KML, GeoParquet) are streamed to a temp file, read via
    geopandas, reprojected to EPSG:4326, and persisted as GeoJSON — so the
    stored artefact and the worker path are identical to the upload case.
    """
    if is_geojson_filename(upload.filename):
        return _ingest_upload(
            upload, task_dir, filename, dict, field_name, max_bytes, external_id, storage
        )

    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in supported_extensions():
        raise HTTPException(
            status_code=415,
            detail=(
                f"{field_name}: unsupported format '{suffix}'. Supported: "
                + ", ".join(sorted(supported_extensions()))
            ),
        )

    raw_path = task_dir / f"{Path(filename).stem}{suffix}"
    _stream_upload_to_file(upload, raw_path, max_bytes, field_name)
    try:
        feature_collection = geo_file_to_geojson_dict(raw_path)
    except GeoIngestError as exc:
        raw_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"{field_name}: {exc}") from exc
    finally:
        raw_path.unlink(missing_ok=True)

    return persist_geojson_dict(feature_collection, task_dir, filename, external_id, storage)


def persist_geojson_dict(
    data: dict[str, Any] | list[dict[str, Any]],
    task_dir: Path,
    filename: str,
    external_id: str,
    storage,
) -> str:
    """Serialise a Python dict to disk as JSON, upload to storage.

    Used by alternative input sources (e.g. urban_api integration) where we
    already have parsed JSON and don't need stream/validate. Mirrors the
    on-disk layout used by user-upload ingestion so the worker downloads it
    the same way.
    """
    local_path = task_dir / filename
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    object_key = f"inputs/{external_id}/{filename}"
    stored = storage.upload_file(str(local_path.resolve()), object_key)
    if storage.is_remote():
        local_path.unlink(missing_ok=True)
    return stored


def _create_pipeline_task(
    *,
    cadastral_file: UploadFile,
    pzz_zones_file: UploadFile | None,
    labels_file: UploadFile | None,
    classifier_file: UploadFile | None,
    include_pzz_check: bool,
    cadastral_vri_col: str,
    pzz_zone_code_col: str,
    pzz_zone_name_col: str,
    priority: int,
    retry_failed: bool,
    force_recompute: bool,
    idempotency_key: str | None,
    app_settings: Settings,
    task_repo: TaskRepository,
    event_repo: EventRepository,
    session: Session,
) -> TaskOut:
    """Shared logic for both submission endpoints.

    Streams uploads to a per-task scratch dir, validates JSON, persists to
    object storage (MinIO or local fallback), then hands a fully resolved
    ``input_paths`` dict to the ``create_task`` use case.
    """
    external_id = uuid4().hex
    task_dir = Path(app_settings.task_inputs_dir) / external_id
    storage = get_object_storage()

    stored_cadastral = _ingest_geo_upload(
        cadastral_file, task_dir, "cadastral_feature_collection.geojson",
        "cadastral_feature_collection_file",
        app_settings.max_upload_bytes, external_id, storage,
    )

    if pzz_zones_file is not None:
        stored_pzz_zones = _ingest_geo_upload(
            pzz_zones_file, task_dir, "pzz_zones_feature_collection.geojson",
            "pzz_zones_feature_collection_file",
            app_settings.max_upload_bytes, external_id, storage,
        )
    else:
        stored_pzz_zones = ""

    if labels_file is not None:
        stored_labels = _ingest_upload(
            labels_file, task_dir, "pzz_zone_vri_labels.json",
            list, "pzz_zone_vri_labels_file",
            app_settings.max_upload_bytes, external_id, storage,
        )
    elif include_pzz_check:
        stored_labels = str(Path(app_settings.default_pzz_zone_labels_path).resolve())
    else:
        stored_labels = ""

    if classifier_file is not None:
        stored_classifier = _ingest_upload(
            classifier_file, task_dir, "vri_classifier.json",
            (dict, list), "vri_classifier_file",
            app_settings.max_upload_bytes, external_id, storage,
        )
    else:
        stored_classifier = str(Path(app_settings.default_vri_classifier_path).resolve())

    input_paths = {
        "cadastral_data_path": stored_cadastral,
        "pzz_zones_data_path": stored_pzz_zones,
        "pzz_zone_vri_labels_path": stored_labels,
        "vri_classifier_path": stored_classifier,
    }

    payload = TaskCreate(
        include_pzz_check=include_pzz_check,
        cadastral_vri_col=cadastral_vri_col,
        pzz_zone_code_col=pzz_zone_code_col,
        pzz_zone_name_col=pzz_zone_name_col,
        priority=priority,
    )

    namespaced_key: str | None = None
    if idempotency_key:
        mode_prefix = "pzz" if include_pzz_check else "clf"
        namespaced_key = f"{mode_prefix}:{idempotency_key}"

    task = create_task(
        payload=payload,
        settings=app_settings,
        task_repo=task_repo,
        event_repo=event_repo,
        enqueue_task=lambda tid: enqueue_pipeline_task(tid, is_scenario=False),
        idempotency_key=namespaced_key,
        retry_failed=retry_failed,
        force_recompute=force_recompute,
        external_id=external_id,
        input_paths=input_paths,
        session=session,
        revoke_task=celery_app.control.revoke,
    )
    session.flush()
    session.refresh(task)
    api_log(
        "create_task", "accepted",
        task_id=task.id, external_id=task.external_id,
        mode=("pzz_check" if include_pzz_check else "classify_only"),
    )
    return TaskOut.model_validate(task)


_TASK_RERUN_DOCSTRING = """**Coordinate Reference System (CRS) requirement.** All GeoJSON uploads
    must be in **EPSG:4326** (WGS84, latitude/longitude). The pipeline
    reprojects internally to the appropriate UTM zone via
    ``estimate_utm_crs`` for overlay and area computations.

    **Idempotency and re-runs.** When ``Idempotency-Key`` (header or form)
    matches an existing task:

    - ``force_recompute=true`` re-enqueues the existing task if it is in a
      terminal state (``finished`` or ``failed``). The ``external_id``
      stays the same; ``result_path`` / ``error_text`` / timestamps are
      cleared.
    - ``retry_failed=true`` only re-runs ``failed`` tasks.
    - Without either flag, the existing task is returned as-is (no
      recompute, instant cached response).
    """


@router.post("/pzz-check", response_model=TaskOut)
def create_pzz_check_task_endpoint(
    cadastral_feature_collection_file: UploadFile = File(...),
    pzz_zones_feature_collection_file: UploadFile = File(...),
    pzz_zone_vri_labels_file: UploadFile | None = File(default=None),
    vri_classifier_file: UploadFile | None = File(default=None),
    cadastral_vri_col: str = Form(..., min_length=1),
    pzz_zone_code_col: str = Form(..., min_length=1),
    pzz_zone_name_col: str = Form(..., min_length=1),
    priority: int = Form(1, ge=1, le=10),
    retry_failed: bool = Form(False),
    force_recompute: bool = Form(False),
    idempotency_key_form: str | None = Form(default=None, alias="Idempotency-Key"),
    idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
    event_repo: EventRepository = Depends(get_event_repo),
    session: Session = Depends(get_db),
) -> TaskOut:
    """Create a pipeline task that runs the full PZZ compliance check.

    Both cadastral parcels and PZZ zone polygons are required; the pipeline
    performs a spatial overlay to determine each parcel's factual zone and
    validates the cadastral VRI text against the PZZ zone definition.

    """
    return _create_pipeline_task(
        cadastral_file=cadastral_feature_collection_file,
        pzz_zones_file=pzz_zones_feature_collection_file,
        labels_file=pzz_zone_vri_labels_file,
        classifier_file=vri_classifier_file,
        include_pzz_check=True,
        cadastral_vri_col=cadastral_vri_col,
        pzz_zone_code_col=pzz_zone_code_col,
        pzz_zone_name_col=pzz_zone_name_col,
        priority=priority,
        retry_failed=retry_failed,
        force_recompute=force_recompute,
        idempotency_key=idempotency_key_header or idempotency_key_form,
        app_settings=app_settings,
        task_repo=task_repo,
        event_repo=event_repo,
        session=session,
    )


create_pzz_check_task_endpoint.__doc__ = (create_pzz_check_task_endpoint.__doc__ or "") + "\n    " + _TASK_RERUN_DOCSTRING


@router.post("/classify-only", response_model=TaskOut)
def create_classify_only_task_endpoint(
    cadastral_feature_collection_file: UploadFile = File(...),
    vri_classifier_file: UploadFile | None = File(default=None),
    cadastral_vri_col: str = Form(..., min_length=1),
    priority: int = Form(1, ge=1, le=10),
    retry_failed: bool = Form(False),
    force_recompute: bool = Form(False),
    idempotency_key_form: str | None = Form(default=None, alias="Idempotency-Key"),
    idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
    event_repo: EventRepository = Depends(get_event_repo),
    session: Session = Depends(get_db),
) -> TaskOut:
    """Create a pipeline task that classifies VRI text against the Rosreestr classifier only.

    No PZZ zones or spatial overlay — only the cadastral text is matched
    against the federal VRI classifier (string + embedding + optional LLM
    rerank). Useful when zone data is unavailable or out of scope.

    """
    return _create_pipeline_task(
        cadastral_file=cadastral_feature_collection_file,
        pzz_zones_file=None,
        labels_file=None,
        classifier_file=vri_classifier_file,
        include_pzz_check=False,
        cadastral_vri_col=cadastral_vri_col,
        pzz_zone_code_col="",
        pzz_zone_name_col="",
        priority=priority,
        retry_failed=retry_failed,
        force_recompute=force_recompute,
        idempotency_key=idempotency_key_header or idempotency_key_form,
        app_settings=app_settings,
        task_repo=task_repo,
        event_repo=event_repo,
        session=session,
    )


create_classify_only_task_endpoint.__doc__ = (create_classify_only_task_endpoint.__doc__ or "") + "\n    " + _TASK_RERUN_DOCSTRING


@router.post("/pzz-check/stream")
async def create_pzz_check_stream_endpoint(
    request: Request,
    cadastral_feature_collection_file: UploadFile = File(...),
    pzz_zones_feature_collection_file: UploadFile = File(...),
    pzz_zone_vri_labels_file: UploadFile | None = File(default=None),
    vri_classifier_file: UploadFile | None = File(default=None),
    cadastral_vri_col: str = Form(..., min_length=1),
    pzz_zone_code_col: str = Form(..., min_length=1),
    pzz_zone_name_col: str = Form(..., min_length=1),
    priority: int = Form(1, ge=1, le=10),
    retry_failed: bool = Form(False),
    force_recompute: bool = Form(False),
    poll_interval: float = Query(2.0, ge=0.5, le=10.0),
    idempotency_key_form: str | None = Form(default=None, alias="Idempotency-Key"),
    idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
    event_repo: EventRepository = Depends(get_event_repo),
    session: Session = Depends(get_db),
) -> EventSourceResponse:
    """Create a full PZZ-check task AND stream it to completion via SSE.

    Same inputs as POST /tasks/pzz-check. One call uploads, creates the task,
    then streams: ``task`` -> ``task_event``/``status`` -> ``geojson`` (the
    classified FeatureCollection with zone verdicts) -> ``done``.

    The upload flow returns the classified layer only; the object-zone-fit
    summary is a scenario/chatbot concern and is available separately via
    GET /tasks/{external_id}/object-zone-fit if needed.

    Use a fetch-based SSE client (native EventSource cannot POST multipart).
    """
    task_out = await run_in_threadpool(
        _create_pipeline_task,
        cadastral_file=cadastral_feature_collection_file,
        pzz_zones_file=pzz_zones_feature_collection_file,
        labels_file=pzz_zone_vri_labels_file,
        classifier_file=vri_classifier_file,
        include_pzz_check=True,
        cadastral_vri_col=cadastral_vri_col,
        pzz_zone_code_col=pzz_zone_code_col,
        pzz_zone_name_col=pzz_zone_name_col,
        priority=priority,
        retry_failed=retry_failed,
        force_recompute=force_recompute,
        idempotency_key=idempotency_key_header or idempotency_key_form,
        app_settings=app_settings,
        task_repo=task_repo,
        event_repo=event_repo,
        session=session,
    )
    session.commit()
    return EventSourceResponse(
        task_stream_with_report_generator(
            task_out.external_id,
            group_by="zone",
            poll_interval=poll_interval,
            request=request,
            app_settings=app_settings,
            initial=task_out.model_dump(mode="json"),
            include_report=False,
            emit_input_files=True,
        )
    )


@router.post("/classify-only/stream")
async def create_classify_only_stream_endpoint(
    request: Request,
    cadastral_feature_collection_file: UploadFile = File(...),
    vri_classifier_file: UploadFile | None = File(default=None),
    cadastral_vri_col: str = Form(..., min_length=1),
    priority: int = Form(1, ge=1, le=10),
    retry_failed: bool = Form(False),
    force_recompute: bool = Form(False),
    poll_interval: float = Query(2.0, ge=0.5, le=10.0),
    idempotency_key_form: str | None = Form(default=None, alias="Idempotency-Key"),
    idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
    event_repo: EventRepository = Depends(get_event_repo),
    session: Session = Depends(get_db),
) -> EventSourceResponse:
    """Create a classify-only task AND stream it to completion via SSE.

    Same inputs as POST /tasks/classify-only (no PZZ zones). Streams:
    ``task`` -> ``task_event``/``status`` -> ``geojson`` (classified
    FeatureCollection with VRI candidate properties) -> ``done``.

    No ``report`` event: classify-only has no zones, so the object-zone-fit
    summary is not applicable. Use a fetch-based SSE client.
    """
    task_out = await run_in_threadpool(
        _create_pipeline_task,
        cadastral_file=cadastral_feature_collection_file,
        pzz_zones_file=None,
        labels_file=None,
        classifier_file=vri_classifier_file,
        include_pzz_check=False,
        cadastral_vri_col=cadastral_vri_col,
        pzz_zone_code_col="",
        pzz_zone_name_col="",
        priority=priority,
        retry_failed=retry_failed,
        force_recompute=force_recompute,
        idempotency_key=idempotency_key_header or idempotency_key_form,
        app_settings=app_settings,
        task_repo=task_repo,
        event_repo=event_repo,
        session=session,
    )
    session.commit()
    return EventSourceResponse(
        task_stream_with_report_generator(
            task_out.external_id,
            group_by="object",
            poll_interval=poll_interval,
            request=request,
            app_settings=app_settings,
            initial=task_out.model_dump(mode="json"),
            include_report=False,
            emit_input_files=True,
        )
    )


@router.post("/chat/stream")
async def create_chat_stream_endpoint(
    request: Request,
    cadastral_feature_collection_file: UploadFile = File(...),
    pzz_zones_feature_collection_file: UploadFile = File(...),
    pzz_zone_vri_labels_file: UploadFile | None = File(default=None),
    vri_classifier_file: UploadFile | None = File(default=None),
    user_query: str = Form(..., min_length=1),
    cadastral_vri_col: str = Form(..., min_length=1),
    pzz_zone_code_col: str = Form(..., min_length=1),
    pzz_zone_name_col: str = Form(..., min_length=1),
    chat_id: str | None = Form(default=None),
    group_by: str = Form("zone"),
    model: str | None = Form(default=None),
    temperature: float | None = Form(default=None),
    priority: int = Form(1, ge=1, le=10),
    force_recompute: bool = Form(False),
    poll_interval: float = Query(2.0, ge=0.5, le=10.0),
    idempotency_key_form: str | None = Form(default=None, alias="Idempotency-Key"),
    idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
    token: str = Depends(verify_token),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
    event_repo: EventRepository = Depends(get_event_repo),
    session: Session = Depends(get_db),
) -> EventSourceResponse:
    """Run a full PZZ check on uploaded files, then stream a conversational answer.

    The file-upload counterpart of ``POST /scenarios/{id}/chat/stream``. Both
    cadastral parcels and PZZ zones are required (the answer is grounded in the
    object-zone-fit report). Uploads may be any supported geo format (GeoJSON,
    GeoPackage, GML, KML, GeoParquet); they're stored as GeoJSON.

    A Bearer token is REQUIRED — chat history is persisted to ChatStorage under
    the token's user. ``chat_id`` is optional: when omitted a new chat is
    created and announced via a ``chat_created`` SSE event.

    SSE events: ``task``, ``task_event``, ``status``, ``object_zone_fit``,
    ``chat_created``, ``token``, ``error``, ``done``. Use a fetch-based SSE
    client (native EventSource cannot POST multipart or set Authorization).
    """
    if group_by not in ("zone", "object"):
        raise HTTPException(status_code=422, detail="group_by must be 'zone' or 'object'")

    task_out = await run_in_threadpool(
        _create_pipeline_task,
        cadastral_file=cadastral_feature_collection_file,
        pzz_zones_file=pzz_zones_feature_collection_file,
        labels_file=pzz_zone_vri_labels_file,
        classifier_file=vri_classifier_file,
        include_pzz_check=True,
        cadastral_vri_col=cadastral_vri_col,
        pzz_zone_code_col=pzz_zone_code_col,
        pzz_zone_name_col=pzz_zone_name_col,
        priority=priority,
        retry_failed=False,
        force_recompute=force_recompute,
        idempotency_key=idempotency_key_header or idempotency_key_form,
        app_settings=app_settings,
        task_repo=task_repo,
        event_repo=event_repo,
        session=session,
    )
    session.commit()
    return EventSourceResponse(
        task_stream_with_chat_generator(
            task_out.external_id,
            group_by=group_by,
            poll_interval=poll_interval,
            request=request,
            app_settings=app_settings,
            initial=task_out.model_dump(mode="json"),
            token=token,
            user_query=user_query,
            chat_id=chat_id,
            scenario_id=None,
            project_id=None,
            chat_title=user_query[:256],
            model=model,
            temperature=temperature,
            emit_input_files=True,
        )
    )
