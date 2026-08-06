"""Task submission endpoints (pzz-check, classify-only) and their helpers."""

from __future__ import annotations

import json
import logging
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

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
from ..application.use_cases.building_zone_review import (
    build_confirmed_overlay,
    build_disclaimer,
    build_suggestion_messages,
    parse_suggestions,
    remaining_uncovered,
    review_building_zones,
    template_candidates,
)
from ..application.use_cases.convert_zone_table import (
    ConversionError,
    convert_zone_table,
)
from ..application.use_cases.detect_columns import (
    BUILDING_TARGETS,
    CADASTRAL_TARGETS,
    PZZ_ZONE_TARGETS,
    ZONE_CODE_TARGET,
    ZONE_NAME_TARGET,
    ZONE_TABLE_TARGETS,
    detect_columns_for_file,
    profile_columns,
    render_detection_narrative,
    required_columns_resolved,
    sparse_column_warnings,
)
from ..dependencies import (
    build_ollama_chat_client,
    get_app_settings,
    get_db,
    get_event_repo,
    get_task_repo,
)
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
from ..output_version import PIPELINE_OUTPUT_VERSION
from ..settings import Settings
from ..tasks import celery_app, enqueue_pipeline_task, execute_pipeline_task
from .security import AuthUser, get_current_user
from ..infrastructure.ollama_chat_client import OllamaChatError
from ..infrastructure.table_reader import TableReadError, read_table
from .tasks import (
    detection_failed_generator,
    prepend_narrative_generator,
    task_stream_with_chat_generator,
    task_stream_with_report_generator,
    zone_review_generator,
)
from .utils import api_log

router = APIRouter(prefix="/tasks", tags=["classifier"])
logger = logging.getLogger("service.api.classifier")


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


def _json_upload(obj: Any, filename: str) -> UploadFile:
    """Wrap an in-memory JSON object as an ``UploadFile`` so a server-generated
    labels file (the confirmed zone overlay) flows through the same ingest path as
    a real upload. ``_stream_upload_to_file`` reads ``.file`` synchronously, so a
    ``BytesIO`` is sufficient."""
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return UploadFile(file=BytesIO(data), filename=filename)


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
            upload,
            task_dir,
            filename,
            dict,
            field_name,
            max_bytes,
            external_id,
            storage,
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

    return persist_geojson_dict(
        feature_collection, task_dir, filename, external_id, storage
    )


def _upload_to_feature_collection(
    upload: UploadFile,
    task_dir: Path,
    field_name: str,
    max_bytes: int,
) -> dict[str, Any]:
    """Read an upload into a GeoJSON FeatureCollection dict for column detection.

    Accepts the same formats as ``_ingest_geo_upload``. Always seeks the upload
    back to the start afterwards so the task-creation path can re-stream the same
    bytes (detection and ingestion each read the file once).
    """
    try:
        if is_geojson_filename(upload.filename):
            raw = upload.file.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"{field_name} exceeds limit of {max_bytes} bytes",
                )
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"{field_name} must contain valid JSON/GeoJSON",
                ) from exc
            if not isinstance(data, dict):
                raise HTTPException(
                    status_code=400, detail=f"{field_name} must be a GeoJSON object"
                )
            return data

        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in supported_extensions():
            raise HTTPException(
                status_code=415,
                detail=(
                    f"{field_name}: unsupported format '{suffix}'. Supported: "
                    + ", ".join(sorted(supported_extensions()))
                ),
            )
        raw_path = task_dir / f"detect{suffix}"
        _stream_upload_to_file(upload, raw_path, max_bytes, field_name)
        try:
            return geo_file_to_geojson_dict(raw_path)
        except GeoIngestError as exc:
            raise HTTPException(status_code=400, detail=f"{field_name}: {exc}") from exc
        finally:
            raw_path.unlink(missing_ok=True)
    finally:
        upload.file.seek(0)


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
    building_upload: bool = False,
    building_type_col: str | None = None,
    building_service_col: str | None = None,
    building_floors_col: str | None = None,
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
        cadastral_file,
        task_dir,
        "cadastral_feature_collection.geojson",
        "cadastral_feature_collection_file",
        app_settings.max_upload_bytes,
        external_id,
        storage,
    )

    if pzz_zones_file is not None:
        stored_pzz_zones = _ingest_geo_upload(
            pzz_zones_file,
            task_dir,
            "pzz_zones_feature_collection.geojson",
            "pzz_zones_feature_collection_file",
            app_settings.max_upload_bytes,
            external_id,
            storage,
        )
    else:
        stored_pzz_zones = ""

    if building_upload:
        # The building flow's optional "descriptions" file is a zone→permitted-VRI
        # mapping, reusing the labels slot. Two schemas are accepted: the urban_api
        # ``functional_zone_mappings`` dict (numeric functional_zone_type_id) and
        # the ПЗЗ letter-index label list (like pzz_zone_llm_labels_template.json,
        # keyed by «Ж-1»). When absent, the runner falls back to the built-in
        # mapping matching whichever backend the zone codes imply.
        if labels_file is not None:
            stored_labels = _ingest_upload(
                labels_file,
                task_dir,
                "pzz_zone_descriptions.json",
                (dict, list),
                "pzz_descriptions_file",
                app_settings.max_upload_bytes,
                external_id,
                storage,
            )
        else:
            stored_labels = ""
    elif labels_file is not None:
        stored_labels = _ingest_upload(
            labels_file,
            task_dir,
            "pzz_zone_vri_labels.json",
            list,
            "pzz_zone_vri_labels_file",
            app_settings.max_upload_bytes,
            external_id,
            storage,
        )
    elif include_pzz_check:
        stored_labels = str(Path(app_settings.default_pzz_zone_labels_path).resolve())
    else:
        stored_labels = ""

    if classifier_file is not None:
        stored_classifier = _ingest_upload(
            classifier_file,
            task_dir,
            "vri_classifier.json",
            (dict, list),
            "vri_classifier_file",
            app_settings.max_upload_bytes,
            external_id,
            storage,
        )
    else:
        stored_classifier = str(
            Path(app_settings.default_vri_classifier_path).resolve()
        )

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
        building_type_col=building_type_col,
        building_service_col=building_service_col,
        building_floors_col=building_floors_col,
        priority=priority,
    )

    namespaced_key: str | None = None
    if building_upload:
        # "bld:" routes the worker to the deterministic UploadedBuildingPzzRunner
        # (start_task derives it from this prefix, mirroring the "sc:" scenario
        # prefix). Always keyed so routing works even without a caller key — the
        # unique external_id then just means no cross-request dedup.
        key_base = idempotency_key or external_id
        namespaced_key = f"bld:{PIPELINE_OUTPUT_VERSION}:{key_base}"
    elif idempotency_key:
        mode_prefix = "pzz" if include_pzz_check else "clf"
        # ``PIPELINE_OUTPUT_VERSION`` in the key auto-invalidates cache hits when
        # the result format changes — a client's old key stops matching after a bump.
        namespaced_key = f"{mode_prefix}:{PIPELINE_OUTPUT_VERSION}:{idempotency_key}"

    task = create_task(
        payload=payload,
        settings=app_settings,
        task_repo=task_repo,
        event_repo=event_repo,
        enqueue_task=lambda tid: enqueue_pipeline_task(
            tid, is_scenario=False, is_building_upload=building_upload
        ),
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
        "create_task",
        "accepted",
        task_id=task.id,
        external_id=task.external_id,
        mode=(
            "building_pzz_check"
            if building_upload
            else ("pzz_check" if include_pzz_check else "classify_only")
        ),
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


create_pzz_check_task_endpoint.__doc__ = (
    (create_pzz_check_task_endpoint.__doc__ or "") + "\n    " + _TASK_RERUN_DOCSTRING
)


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


create_classify_only_task_endpoint.__doc__ = (
    (create_classify_only_task_endpoint.__doc__ or "")
    + "\n    "
    + _TASK_RERUN_DOCSTRING
)


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


# DEPRECATED alias: the endpoint was renamed /tasks/chat/stream ->
# /tasks/pzz-check/chat/stream for symmetry with /tasks/classify-only/chat/stream.
# The old path is kept temporarily so existing frontends keep working; remove it
# once they migrate. Both decorators bind the same handler.
@router.post(
    "/chat/stream",
    deprecated=True,
    summary="[DEPRECATED] use POST /tasks/pzz-check/chat/stream",
)
@router.post("/pzz-check/chat/stream")
async def create_pzz_check_chat_stream_endpoint(
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
    user: AuthUser = Depends(get_current_user),
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
        raise HTTPException(
            status_code=422, detail="group_by must be 'zone' or 'object'"
        )

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
            user_id=user.user_id,
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


@router.post("/classify-only/chat/stream")
async def create_classify_only_chat_stream_endpoint(
    request: Request,
    cadastral_feature_collection_file: UploadFile = File(...),
    vri_classifier_file: UploadFile | None = File(default=None),
    user_query: str = Form(..., min_length=1),
    cadastral_vri_col: str = Form(..., min_length=1),
    chat_id: str | None = Form(default=None),
    model: str | None = Form(default=None),
    temperature: float | None = Form(default=None),
    priority: int = Form(1, ge=1, le=10),
    force_recompute: bool = Form(False),
    poll_interval: float = Query(2.0, ge=0.5, le=10.0),
    idempotency_key_form: str | None = Form(default=None, alias="Idempotency-Key"),
    idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
    user: AuthUser = Depends(get_current_user),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
    event_repo: EventRepository = Depends(get_event_repo),
    session: Session = Depends(get_db),
) -> EventSourceResponse:
    """Run a classify-only pass on uploaded files, then stream a conversational answer.

    The classify-only counterpart of ``POST /tasks/pzz-check/chat/stream``: no PZZ zones
    and no spatial overlay — the answer is grounded in the classifier-candidate
    summary (top-1 / top-5 VRI per object) instead of the object-zone-fit
    report. Uploads may be any supported geo format; they're stored as GeoJSON.

    A Bearer token is REQUIRED — chat history is persisted to ChatStorage under
    the token's user. ``chat_id`` is optional: when omitted a new chat is
    created and announced via a ``chat_created`` SSE event.

    SSE events: ``task``, ``task_event``, ``status``, ``classify_summary``,
    ``chat_created``, ``token``, ``error``, ``done``. Use a fetch-based SSE
    client (native EventSource cannot POST multipart or set Authorization).
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
            group_by="object",
            poll_interval=poll_interval,
            request=request,
            app_settings=app_settings,
            initial=task_out.model_dump(mode="json"),
            user_id=user.user_id,
            user_query=user_query,
            chat_id=chat_id,
            scenario_id=None,
            project_id=None,
            chat_title=user_query[:256],
            model=model,
            temperature=temperature,
            report_kind="classify",
            emit_input_files=True,
            system_prompt_path=app_settings.chat_system_prompt_classify_path,
        )
    )


def _default_auto_query(include_pzz_check: bool) -> str:
    """Grounding query used when the auto endpoint gets no explicit user_query."""
    if include_pzz_check:
        return (
            "Подготовь подробный разбор результата проверки загруженных земельных "
            "участков на соответствие ПЗЗ в официально-деловом стиле: главный "
            "вывод, итоги по категориям, разбивка по территориальным зонам ПЗЗ и "
            "на что обратить внимание (потенциальные нарушения, земельные участки "
            "на ручной проверке и что делать)."
        )
    return (
        "Подготовь подробный разбор результата классификации ВРИ загруженных "
        "земельных участков в официально-деловом стиле: главный вывод и на что "
        "обратить внимание."
    )


_DEFAULT_BUILDING_AUTO_QUERY = (
    "Подготовь подробный разбор результата проверки загруженных зданий на "
    "соответствие ПЗЗ в официально-деловом стиле: главный вывод, итоги по "
    "категориям, разбивка по территориальным зонам ПЗЗ и на что обратить "
    "внимание (потенциальные нарушения, здания на ручной проверке и что делать)."
)


class _DescriptionsTableError(Exception):
    """A CSV/XLSX descriptions table could not be converted (bad table / missing column)."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


async def _convert_descriptions_if_table(
    descriptions_file: UploadFile | None,
    app_settings: Settings,
    model: str | None,
) -> tuple[UploadFile | None, str]:
    """Convert a CSV/XLSX ``pzz_descriptions_file`` to the label JSON in place.

    A JSON/GeoJSON descriptions file (or none) passes through unchanged. A table is
    read, its columns detected (LLM-first, heuristic backstop) and folded into the
    label schema, then returned as an in-memory JSON ``UploadFile`` so the rest of
    the flow is oblivious to the input format. Returns ``(file, note)`` where
    ``note`` is a leading-narrative line about what was recognised. Raises
    :class:`_DescriptionsTableError` when the table is unreadable or a required
    column can't be resolved, so the caller emits a terminal detection_failed.
    """
    if descriptions_file is None:
        return None, ""
    name = (descriptions_file.filename or "").lower()
    if not (name.endswith(".csv") or name.endswith(".xlsx")):
        return descriptions_file, ""
    data = await descriptions_file.read()
    try:
        rows, headers = read_table(data, descriptions_file.filename or "")
    except TableReadError as exc:
        raise _DescriptionsTableError(
            "Таблицу описаний зон правил землепользования и застройки не удалось "
            f"прочитать: {exc}."
        ) from exc
    feature_collection = {"features": [{"properties": r} for r in rows]}
    async with build_ollama_chat_client(app_settings) as ollama_client:
        suggestions = await detect_columns_for_file(
            ollama_client,
            feature_collection,
            ZONE_TABLE_TARGETS,
            model=model,
            heuristic_first=False,
        )
    resolved = {
        t.key: (suggestions[t.key].value if suggestions.get(t.key) else None)
        for t in ZONE_TABLE_TARGETS
    }
    try:
        zones, report = convert_zone_table(rows, resolved)
    except ConversionError as exc:
        raise _DescriptionsTableError(
            f"В таблице описаний зон правил землепользования и застройки не определить "
            f"обязательную колонку ({exc}). Заголовки: {', '.join(headers)}. "
            "Задайте её или подайте описание в JSON."
        ) from exc
    # Full spellings here: this note leads the narrative, before the model's
    # «ВРИ — …; ПЗЗ — …» line defines the abbreviations.
    role_labels = {
        "zone_code": "код зоны",
        "zone_name": "наименование зоны",
        "permission": "тип разрешения",
        "vri_code": "код вида разрешённого использования",
        "vri_name": "наименование вида разрешённого использования",
    }
    columns = [
        f"{role_labels[key]} — «{col}»"
        for key, col in resolved.items()
        if col and key in role_labels
    ]
    note = (
        "Из таблицы описаний зон правил землепользования и застройки распознано: "
        f"{report.zones_count} зон, {report.vri_count} видов разрешённого использования."
    )
    if columns:
        note += " Подобранные колонки таблицы: " + "; ".join(columns) + "."
    if report.warnings:
        note += " Предупреждения: " + "; ".join(report.warnings)
    return _json_upload(zones, "pzz_descriptions_from_table.json"), note


async def _run_building_pzz_auto(
    *,
    request: Request,
    buildings_file: UploadFile,
    zones_file: UploadFile,
    descriptions_file: UploadFile | None,
    confirmed_zone_map: dict[str, str] | None,
    user_query: str | None,
    chat_id: str | None,
    group_by: str,
    model: str | None,
    temperature: float | None,
    priority: int,
    force_recompute: bool,
    poll_interval: float,
    idempotency_key: str | None,
    user_id: str,
    app_settings: Settings,
    task_repo: TaskRepository,
    event_repo: EventRepository,
    session: Session,
) -> EventSourceResponse:
    """Auto-detect building columns, run the deterministic building PZZ check, stream chat.

    The uploaded-building counterpart of the ``pzz_check`` auto flow: buildings
    (physical_object_type_id / service_type_id + floors) are matched against the
    uploaded PZZ zones; the answer is grounded in the object-zone-fit report.

    A CSV/XLSX ``pzz_descriptions_file`` is converted to the label JSON up front,
    so the frontend can drop a spreadsheet without a separate ``/convert`` call.
    """
    try:
        descriptions_file, table_note = await _convert_descriptions_if_table(
            descriptions_file, app_settings, model
        )
    except _DescriptionsTableError as exc:
        return EventSourceResponse(detection_failed_generator("", exc.detail))

    max_bytes = app_settings.max_upload_bytes
    detect_dir = Path(app_settings.task_inputs_dir) / f"detect-{uuid4().hex}"
    try:
        buildings_fc = await run_in_threadpool(
            _upload_to_feature_collection,
            buildings_file,
            detect_dir,
            "buildings_feature_collection_file",
            max_bytes,
        )
        zones_fc = await run_in_threadpool(
            _upload_to_feature_collection,
            zones_file,
            detect_dir,
            "pzz_zones_feature_collection_file",
            max_bytes,
        )
    finally:
        shutil.rmtree(detect_dir, ignore_errors=True)

    # The zone name is detected too (optional) so the letter-index review can offer
    # a name-based suggestion for zones missing from the built-in template.
    announce_targets = list(BUILDING_TARGETS) + [ZONE_CODE_TARGET, ZONE_NAME_TARGET]
    async with build_ollama_chat_client(app_settings) as ollama_client:
        suggestions = await detect_columns_for_file(
            ollama_client, buildings_fc, BUILDING_TARGETS, model=model
        )
        suggestions.update(
            await detect_columns_for_file(
                ollama_client,
                zones_fc,
                [ZONE_CODE_TARGET, ZONE_NAME_TARGET],
                model=model,
            )
        )
    narrative = render_detection_narrative(suggestions, announce_targets)
    sparse_warnings = sparse_column_warnings(
        suggestions, BUILDING_TARGETS, profile_columns(buildings_fc)
    ) + sparse_column_warnings(
        suggestions, [ZONE_CODE_TARGET, ZONE_NAME_TARGET], profile_columns(zones_fc)
    )
    if sparse_warnings:
        narrative = narrative + "\n\n" + "\n".join(sparse_warnings)
    if table_note:
        narrative = table_note + "\n\n" + narrative

    def _val(key: str) -> str | None:
        s = suggestions.get(key)
        return s.value if s is not None else None

    zone_code = _val("pzz_zone_code_col")
    zone_name = _val("pzz_zone_name_col")
    building_type = _val("building_type_col")
    building_service = _val("building_service_col")
    # Floors is optional (residential falls back without it); a zone code plus at
    # least one building identifier (type or service) is the minimum to classify.
    if not zone_code or not (building_type or building_service):
        return EventSourceResponse(
            detection_failed_generator(
                narrative,
                "нужны колонка кода зоны ПЗЗ и хотя бы одна из колонок «тип» или "
                "«сервис» здания",
            )
        )

    # Real-ПЗЗ (letter-index) zone review: when the zones aren't numeric urban_api
    # ids and the user gave no descriptions file, the check runs on the built-in
    # template — approximate, and only covering a subset of indices. Decide whether
    # to proceed (with a disclaimer), ask to upload proper descriptions, or offer
    # per-zone LLM suggestions to confirm. Confirmed suggestions become an overlay
    # descriptions file fed to the runner.
    review = review_building_zones(
        zones_fc=zones_fc,
        code_col=zone_code,
        name_col=zone_name,
        user_uploaded_descriptions=descriptions_file is not None,
        confirmed_map=confirmed_zone_map,
        template_path=app_settings.default_pzz_zone_labels_path,
        threshold=app_settings.building_pzz_zone_suggest_threshold,
    )
    if review.action == "suggest_upload":
        detail = (
            f"Обобщённый шаблон ПЗЗ не подходит: {len(review.uncovered)} зон не "
            f"найдено ({', '.join(review.uncovered_codes)}). Загрузите описание "
            "разрешённых ВРИ вашего ПЗЗ (поле pzz_descriptions_file) и повторите."
        )
        return EventSourceResponse(
            zone_review_generator(
                narrative, detail, action="suggest_upload", suggestions=None
            )
        )
    if review.action == "confirm":
        candidates = template_candidates(app_settings.default_pzz_zone_labels_path)
        messages, schema = build_suggestion_messages(review.uncovered, candidates)
        llm_available = True
        try:
            async with build_ollama_chat_client(app_settings) as ollama_client:
                parsed = await ollama_client.complete_json(
                    messages, schema=schema, model=model
                )
        except (OllamaChatError, httpx.HTTPError) as exc:
            # No suggestions possible without the model — degrade to the upload
            # ask rather than 500 the whole request.
            logger.warning(
                "zone-suggestion LLM call failed (%s); asking for upload", exc
            )
            parsed, llm_available = {}, False
        if not llm_available:
            detail = (
                "Не удалось подобрать соответствия зон (модель недоступна). "
                f"Не найдены в шаблоне ПЗЗ: {', '.join(review.uncovered_codes)}. "
                "Загрузите описание разрешённых ВРИ вашего ПЗЗ (поле pzz_descriptions_file) "
                "и повторите."
            )
            return EventSourceResponse(
                zone_review_generator(
                    narrative, detail, action="suggest_upload", suggestions=None
                )
            )
        picks = parse_suggestions(review.uncovered, parsed, candidates)
        by_code = {c["code"]: c["name"] for c in candidates}
        payload = [
            {
                "user_code": z.code,
                "user_name": z.name,
                "suggested_code": picks.get(z.code),
                "suggested_name": by_code.get(picks.get(z.code, ""), ""),
            }
            for z in review.uncovered
        ]
        detail = (
            "Часть зон не найдена в шаблоне ПЗЗ. Подтвердите предложенные "
            "соответствия (или загрузите своё описание) и повторите запрос, передав "
            "confirmed_zone_map."
        )
        return EventSourceResponse(
            zone_review_generator(
                narrative, detail, action="confirm", suggestions=payload
            )
        )

    # action == "proceed". Build the confirmed overlay (if any), and prepend the
    # approximate-classification disclaimer when the template/overlay was used.
    effective_descriptions = descriptions_file
    if confirmed_zone_map and review.approximate:
        overlay = build_confirmed_overlay(
            app_settings.default_pzz_zone_labels_path, confirmed_zone_map
        )
        effective_descriptions = _json_upload(
            overlay, "pzz_zone_confirmed_overlay.json"
        )
    if review.approximate:
        narrative = (
            narrative
            + "\n\n"
            + build_disclaimer(
                remaining_uncovered(review.uncovered, confirmed_zone_map)
            )
        )

    task_out = await run_in_threadpool(
        _create_pipeline_task,
        cadastral_file=buildings_file,
        pzz_zones_file=zones_file,
        labels_file=effective_descriptions,
        classifier_file=None,
        include_pzz_check=True,
        cadastral_vri_col="",
        pzz_zone_code_col=zone_code,
        pzz_zone_name_col=zone_name or "",
        priority=priority,
        retry_failed=False,
        force_recompute=force_recompute,
        idempotency_key=idempotency_key,
        app_settings=app_settings,
        task_repo=task_repo,
        event_repo=event_repo,
        session=session,
        building_upload=True,
        building_type_col=building_type,
        building_service_col=building_service,
        building_floors_col=_val("building_floors_col"),
    )
    session.commit()

    effective_query = user_query or _DEFAULT_BUILDING_AUTO_QUERY
    inner = task_stream_with_chat_generator(
        task_out.external_id,
        group_by=group_by,
        poll_interval=poll_interval,
        request=request,
        app_settings=app_settings,
        initial=task_out.model_dump(mode="json"),
        user_id=user_id,
        user_query=effective_query,
        chat_id=chat_id,
        scenario_id=None,
        project_id=None,
        chat_title=(user_query or narrative)[:256],
        model=model,
        temperature=temperature,
        report_kind="object_zone_fit",
        emit_input_files=True,
        system_prompt_path=app_settings.chat_system_prompt_building_path,
    )
    return EventSourceResponse(prepend_narrative_generator(narrative, inner))


@router.post("/auto/chat/stream")
async def create_auto_chat_stream_endpoint(
    request: Request,
    cadastral_feature_collection_file: UploadFile = File(...),
    pzz_zones_feature_collection_file: UploadFile | None = File(default=None),
    pzz_zone_vri_labels_file: UploadFile | None = File(default=None),
    pzz_descriptions_file: UploadFile | None = File(default=None),
    vri_classifier_file: UploadFile | None = File(default=None),
    mode: str = Form("pzz_check"),
    user_query: str | None = Form(default=None),
    chat_id: str | None = Form(default=None),
    group_by: str = Form("zone"),
    model: str | None = Form(default=None),
    temperature: float | None = Form(default=None),
    priority: int = Form(1, ge=1, le=10),
    force_recompute: bool = Form(False),
    confirmed_zone_map: str | None = Form(default=None),
    poll_interval: float = Query(2.0, ge=0.5, le=10.0),
    idempotency_key_form: str | None = Form(default=None, alias="Idempotency-Key"),
    idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
    user: AuthUser = Depends(get_current_user),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
    event_repo: EventRepository = Depends(get_event_repo),
    session: Session = Depends(get_db),
) -> EventSourceResponse:
    """Auto-detect the classifier's input columns, then run + stream the answer.

    The "pick a mode, drop the files, magic happens" flow: instead of the caller
    naming ``cadastral_vri_col`` / ``pzz_zone_code_col`` / ``pzz_zone_name_col``,
    the columns are detected from the uploaded data (heuristic + LLM), announced
    as a leading ``chunk`` ("поле X определено как …"), and the full pipeline runs
    automatically — grounding a conversational answer just like the other chat
    endpoints.

    ``mode``:
      - ``pzz_check`` (default) — needs cadastral + PZZ zones; detects all three
        columns; answer grounded in the object-zone-fit report. A CSV/XLSX
        ``pzz_zone_vri_labels_file`` is converted to the label JSON inline;
      - ``classify_only`` — needs only cadastral; detects the VRI column; answer
        grounded in the classify summary.
      - ``building_pzz_check`` — needs a buildings layer (``cadastral_…_file``:
        Urban-API-shaped ``physical_object_type_id`` / ``service_type_id`` +
        floors) + PZZ zones; detects the type/service/floors + zone-code columns
        and runs the deterministic building PZZ check (no LLM classification).
        Optionally accepts ``pzz_descriptions_file`` (a zone→permitted-VRI mapping);
        without it the built-in mapping is used. Answer grounded in object-zone-fit.

    When a required column can't be determined, no task is started: the narrative
    (which columns are missing) plus an ``error`` and terminal ``done`` are
    emitted so the user can retry with the columns specified.

    A Bearer token is REQUIRED (chat history is persisted). Use a fetch-based SSE
    client (native EventSource cannot POST multipart or set Authorization).
    """
    if mode not in ("pzz_check", "classify_only", "building_pzz_check"):
        raise HTTPException(
            status_code=422,
            detail="mode must be 'pzz_check', 'classify_only' or 'building_pzz_check'",
        )
    if group_by not in ("zone", "object"):
        raise HTTPException(
            status_code=422, detail="group_by must be 'zone' or 'object'"
        )

    if mode == "building_pzz_check":
        if pzz_zones_feature_collection_file is None:
            raise HTTPException(
                status_code=422,
                detail="pzz_zones_feature_collection_file is required for mode=building_pzz_check",
            )
        parsed_zone_map: dict[str, str] | None = None
        if confirmed_zone_map:
            try:
                loaded = json.loads(confirmed_zone_map)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=422, detail="confirmed_zone_map must be a JSON object"
                ) from exc
            if not isinstance(loaded, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in loaded.items()
            ):
                raise HTTPException(
                    status_code=422,
                    detail="confirmed_zone_map must map zone codes to template codes (str→str)",
                )
            parsed_zone_map = loaded or None
        return await _run_building_pzz_auto(
            request=request,
            buildings_file=cadastral_feature_collection_file,
            zones_file=pzz_zones_feature_collection_file,
            descriptions_file=pzz_descriptions_file,
            confirmed_zone_map=parsed_zone_map,
            user_query=user_query,
            chat_id=chat_id,
            group_by=group_by,
            model=model,
            temperature=temperature,
            priority=priority,
            force_recompute=force_recompute,
            poll_interval=poll_interval,
            idempotency_key=idempotency_key_header or idempotency_key_form,
            user_id=user.user_id,
            app_settings=app_settings,
            task_repo=task_repo,
            event_repo=event_repo,
            session=session,
        )

    include_pzz_check = mode == "pzz_check"
    if include_pzz_check and pzz_zones_feature_collection_file is None:
        raise HTTPException(
            status_code=422,
            detail="pzz_zones_feature_collection_file is required for mode=pzz_check",
        )

    # A CSV/XLSX zone-labels file is converted to the label JSON up front (same as
    # the building descriptions flow), so the pipeline receives the list schema it
    # expects. JSON labels pass through unchanged.
    labels_note = ""
    if include_pzz_check:
        try:
            pzz_zone_vri_labels_file, labels_note = (
                await _convert_descriptions_if_table(
                    pzz_zone_vri_labels_file, app_settings, model
                )
            )
        except _DescriptionsTableError as exc:
            return EventSourceResponse(detection_failed_generator("", exc.detail))

    max_bytes = app_settings.max_upload_bytes
    detect_dir = Path(app_settings.task_inputs_dir) / f"detect-{uuid4().hex}"
    try:
        cadastral_fc = await run_in_threadpool(
            _upload_to_feature_collection,
            cadastral_feature_collection_file,
            detect_dir,
            "cadastral_feature_collection_file",
            max_bytes,
        )
        zones_fc: dict[str, Any] | None = None
        if include_pzz_check:
            zones_fc = await run_in_threadpool(
                _upload_to_feature_collection,
                pzz_zones_feature_collection_file,
                detect_dir,
                "pzz_zones_feature_collection_file",
                max_bytes,
            )
    finally:
        shutil.rmtree(detect_dir, ignore_errors=True)

    targets = list(CADASTRAL_TARGETS) + (
        list(PZZ_ZONE_TARGETS) if include_pzz_check else []
    )
    async with build_ollama_chat_client(app_settings) as ollama_client:
        suggestions = await detect_columns_for_file(
            ollama_client, cadastral_fc, CADASTRAL_TARGETS, model=model
        )
        if include_pzz_check and zones_fc is not None:
            suggestions.update(
                await detect_columns_for_file(
                    ollama_client, zones_fc, PZZ_ZONE_TARGETS, model=model
                )
            )
    narrative = render_detection_narrative(
        suggestions, targets, include_pzz_check=include_pzz_check
    )
    sparse_warnings = sparse_column_warnings(
        suggestions, CADASTRAL_TARGETS, profile_columns(cadastral_fc)
    )
    if include_pzz_check and zones_fc is not None:
        sparse_warnings += sparse_column_warnings(
            suggestions, PZZ_ZONE_TARGETS, profile_columns(zones_fc)
        )
    if sparse_warnings:
        narrative = narrative + "\n\n" + "\n".join(sparse_warnings)
    if labels_note:
        narrative = labels_note + "\n\n" + narrative

    if not required_columns_resolved(suggestions, targets):
        return EventSourceResponse(
            detection_failed_generator(
                narrative,
                "не удалось определить обязательные колонки из загруженных данных",
            )
        )

    task_out = await run_in_threadpool(
        _create_pipeline_task,
        cadastral_file=cadastral_feature_collection_file,
        pzz_zones_file=pzz_zones_feature_collection_file if include_pzz_check else None,
        labels_file=pzz_zone_vri_labels_file,
        classifier_file=vri_classifier_file,
        include_pzz_check=include_pzz_check,
        cadastral_vri_col=suggestions["cadastral_vri_col"].value,
        pzz_zone_code_col=(
            suggestions.get("pzz_zone_code_col").value if include_pzz_check else ""
        ),
        pzz_zone_name_col=(
            suggestions.get("pzz_zone_name_col").value if include_pzz_check else ""
        ),
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

    effective_query = user_query or _default_auto_query(include_pzz_check)
    inner = task_stream_with_chat_generator(
        task_out.external_id,
        group_by=group_by if include_pzz_check else "object",
        poll_interval=poll_interval,
        request=request,
        app_settings=app_settings,
        initial=task_out.model_dump(mode="json"),
        user_id=user.user_id,
        user_query=effective_query,
        chat_id=chat_id,
        scenario_id=None,
        project_id=None,
        chat_title=(user_query or narrative)[:256],
        model=model,
        temperature=temperature,
        report_kind="object_zone_fit" if include_pzz_check else "classify",
        emit_input_files=True,
        system_prompt_path=(
            None if include_pzz_check else app_settings.chat_system_prompt_classify_path
        ),
    )
    return EventSourceResponse(prepend_narrative_generator(narrative, inner))
