"""Scenario-driven endpoints that pull data from IDU urban_api.

Instead of file uploads, the user supplies a ``scenario_id`` + (year, source).
The endpoint fetches the scenario's functional zones and residential
physical objects from urban_api, transforms them into the GeoJSON shape
the existing pipeline expects, and kicks off a normal classification
task. The worker is unchanged — it just sees inputs persisted to MinIO,
same as the upload path.

Auth: the user's incoming ``Authorization: Bearer ...`` header is
forwarded to urban_api verbatim.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Path as FastPath,
    Query,
    Request as FastRequest,
    UploadFile,
)
from sqlalchemy import select
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from starlette.responses import FileResponse

from ..application.use_cases.create_task import create_task
from ..db import session_scope
from ..dependencies import get_app_settings, get_db, get_event_repo, get_task_repo
from ..domain.ports.event_repository import EventRepository
from ..domain.ports.task_repository import TaskRepository
from ..infrastructure.functional_zone_names import (
    functional_zone_display_name,
    functional_zone_type_label,
)
from ..infrastructure.pzz_mapping import (
    build_pipeline_zone_labels,
    lookup_zone_summary,
    mapping_version,
)
from ..infrastructure.repositories.sqlalchemy_task_repository import (
    SqlAlchemyTaskRepository,
)
from ..infrastructure.storage import get_object_storage
from ..infrastructure.urban_api_client import UrbanApiClient, UrbanApiError
from ..models import PipelineTask, TaskEvent, TaskStatus
from ..schemas import TaskCreate, TaskEventOut, TaskOut
from ..output_version import PIPELINE_OUTPUT_VERSION
from ..settings import Settings
from ..tasks import enqueue_pipeline_task, execute_pipeline_task
from ..application.use_cases.uploads import uploads_root
from .classifier import _placement_context, _resolve_file_slot, persist_geojson_dict
from ..application.use_cases.resolve_service_vri import resolve_service_vri
from fastapi.concurrency import run_in_threadpool

from .tasks import (
    _TERMINAL_STATUSES,
    SCENARIO_ZONE_NAME_COL,
    build_cancel_task_response,
    build_object_zone_fit_response,
    build_placement_summary_response,
    build_recompute_task_response,
    build_scenario_zone_geo_layers,
    build_task_events_response,
    build_task_result_response,
    get_task_or_404,
    task_stream_with_chat_generator,
    task_stream_with_report_generator,
)
from .security import AuthUser, get_current_user, get_optional_user, verify_token
from .utils import api_log, field_title

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


_CADASTRAL_VRI_COL = "vri_text"
_PZZ_ZONE_CODE_COL = "zone_code"
_PZZ_ZONE_NAME_COL = SCENARIO_ZONE_NAME_COL
_SCENARIO_IDEMPOTENCY_PREFIX = "sc:"
# urban_api physical_object_type ids: 4 = жилой дом (the default), 5 = нежилое здание.
_RESIDENTIAL_PO_TYPE = 4
NON_RESIDENTIAL_PO_TYPE = 5


def _scenario_id_from_idempotency_key(key: str | None) -> int | None:
    if not key or not key.startswith(_SCENARIO_IDEMPOTENCY_PREFIX):
        return None
    parts = key.split(":", 2)
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _get_scenario_task_or_404(
    *,
    scenario_id: int,
    external_id: str,
    task_repo: TaskRepository,
):
    task = get_task_or_404(external_id, task_repo)
    key = task_repo.get_idempotency_key_by_external_id(external_id)
    if _scenario_id_from_idempotency_key(key) != scenario_id:
        raise HTTPException(status_code=404, detail="Scenario task not found")
    return task


def _raise_urban_api_http_exception(exc: UrbanApiError) -> None:
    if exc.status in {401, 403, 404}:
        raise HTTPException(
            status_code=exc.status, detail=f"urban_api: {exc.body}"
        ) from exc
    raise HTTPException(status_code=502, detail=f"urban_api: {exc}") from exc


async def _verify_scenario_access(
    *,
    scenario_id: int,
    token: str,
    app_settings: Settings,
) -> None:
    if not app_settings.urban_api_base_url:
        raise HTTPException(
            status_code=503,
            detail="urban_api integration is not configured.",
        )

    async with UrbanApiClient(
        base_url=app_settings.urban_api_base_url,
        timeout_seconds=app_settings.urban_api_timeout_seconds,
    ) as urban:
        try:
            await urban.get_scenario_info(scenario_id, token=token)
        except UrbanApiError as exc:
            _raise_urban_api_http_exception(exc)


def _flatten_physical_object_features(
    feature_collection: dict[str, Any],
    *,
    vri_col: str,
) -> dict[str, Any]:
    """Add a ``vri_text`` property to each Feature so the pipeline can read it.

    urban_api gives us ``physical_object_type.name`` ("Жилой дом") plus a
    free-form ``properties`` dict that often holds "Количество этажей".
    We collapse those into a single human-readable VRI text string —
    floor count helps the classifier distinguish 2.1.1 / 2.5 / 2.6.
    """
    features = feature_collection.get("features") or []
    for feature in features:
        props = feature.setdefault("properties", {})
        service_type = props.get("service_type")
        if isinstance(service_type, dict):
            # Scenario service (from services_with_geometry): the runner resolves
            # its VRI by service_type_id; this text is for display only.
            type_name = service_type.get("name") or ""
            name = props.get("name") or ""
            if type_name and name and name != type_name:
                props[vri_col] = f"{type_name}, {name}"
            else:
                props[vri_col] = type_name or name or "неизвестный сервис"
            continue
        po_type = props.get("physical_object_type") or {}
        type_name = (po_type.get("name") if isinstance(po_type, dict) else None) or ""
        nested = (
            props.get("properties") if isinstance(props.get("properties"), dict) else {}
        )
        floors = nested.get("Количество этажей") or props.get("Количество этажей")
        if type_name and floors:
            props[vri_col] = f"{type_name}, {floors} этажей"
        elif type_name:
            props[vri_col] = type_name
        else:
            props[vri_col] = "неизвестный объект"
    return feature_collection


def _flatten_functional_zone_features(
    feature_collection: dict[str, Any],
    *,
    code_col: str,
    name_col: str,
) -> dict[str, Any]:
    """Flatten ``functional_zone_type`` into top-level ``zone_code`` / ``zone_name``.

    Pipeline expects scalar string columns; urban_api gives us a nested
    object. We use the type id as the code (stable per scenario). The name is
    the zone's own ``name`` when the project set one, else the Russian type
    label (see ``functional_zone_display_name``), so downstream outputs never
    show an English type code.
    """
    features = feature_collection.get("features") or []
    for feature in features:
        props = feature.setdefault("properties", {})
        zone_type = props.get("functional_zone_type")
        zone_type_id = zone_type.get("id") if isinstance(zone_type, dict) else None
        props[code_col] = str(zone_type_id) if zone_type_id is not None else ""
        props[name_col] = functional_zone_display_name(props)
    return feature_collection


def _persist_and_create_scenario_task(
    *,
    objects_fc: dict[str, Any],
    zones_fc: dict[str, Any],
    scenario_id: int,
    year: int,
    source: str,
    physical_object_type_id: int,
    priority: int,
    namespaced_key: str,
    force_recompute: bool,
    app_settings: Settings,
    task_repo: TaskRepository,
    event_repo: EventRepository,
    session: Session,
    placement_vri_code: str | None = None,
    placement_service_name: str | None = None,
    placement_context: dict[str, Any] | None = None,
    zone_descriptions: dict[str, Any] | None = None,
    vri_classifier: dict[str, Any] | list[Any] | None = None,
) -> PipelineTask:
    """Flatten inputs, persist GeoJSON to storage, and create the task.

    ``zone_descriptions`` / ``vri_classifier`` are the user's own optional
    regulation files (already parsed and validated); when given they replace the
    built-in functional-zone mapping and ВРИ classifier for this task.

    Synchronous and I/O-heavy (three MinIO writes); intended to run in a
    threadpool so the API event loop stays responsive under concurrent submits.
    """
    if not placement_vri_code:
        objects_fc = _flatten_physical_object_features(
            objects_fc, vri_col=_CADASTRAL_VRI_COL
        )
    zones_fc = _flatten_functional_zone_features(
        zones_fc, code_col=_PZZ_ZONE_CODE_COL, name_col=_PZZ_ZONE_NAME_COL
    )

    external_id = uuid4().hex
    task_dir = Path(app_settings.task_inputs_dir) / external_id
    storage = get_object_storage()

    stored_cadastral = persist_geojson_dict(
        objects_fc,
        task_dir,
        "cadastral_feature_collection.geojson",
        external_id,
        storage,
    )
    stored_pzz_zones = persist_geojson_dict(
        zones_fc, task_dir, "pzz_zones_feature_collection.geojson", external_id, storage
    )

    observed_zone_types: dict[str, str | None] = {}
    for feature in zones_fc.get("features") or []:
        props = feature.get("properties") or {}
        code = (props.get(_PZZ_ZONE_CODE_COL) or "").strip()
        if code and code not in observed_zone_types:
            observed_zone_types[code] = props.get(_PZZ_ZONE_NAME_COL) or None
    scenario_labels = build_pipeline_zone_labels(
        observed_zone_types=observed_zone_types
    )
    if zone_descriptions is not None:
        stored_labels = persist_geojson_dict(
            zone_descriptions,
            task_dir,
            "pzz_zone_descriptions.json",
            external_id,
            storage,
        )
    elif placement_vri_code:
        stored_labels = ""
    else:
        stored_labels = persist_geojson_dict(
            scenario_labels, task_dir, "pzz_zone_vri_labels.json", external_id, storage
        )
    if vri_classifier is not None:
        stored_classifier = persist_geojson_dict(
            vri_classifier, task_dir, "vri_classifier.json", external_id, storage
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
    if placement_context is not None:
        input_paths["placement_context_path"] = persist_geojson_dict(
            placement_context, task_dir, "placement_context.json", external_id, storage
        )

    payload = TaskCreate(
        include_pzz_check=True,
        cadastral_vri_col=_CADASTRAL_VRI_COL,
        pzz_zone_code_col=_PZZ_ZONE_CODE_COL,
        pzz_zone_name_col=_PZZ_ZONE_NAME_COL,
        priority=priority,
        placement_vri_code=placement_vri_code,
        placement_service_name=(placement_service_name or "")[:256] or None,
    )

    task = create_task(
        payload=payload,
        settings=app_settings,
        task_repo=task_repo,
        event_repo=event_repo,
        enqueue_task=lambda tid: enqueue_pipeline_task(tid, is_scenario=True),
        idempotency_key=namespaced_key,
        retry_failed=False,
        force_recompute=force_recompute,
        external_id=external_id,
        input_paths=input_paths,
        session=session,
    )
    session.flush()
    session.refresh(task)
    api_log(
        "create_task",
        "accepted",
        task_id=task.id,
        external_id=task.external_id,
        mode="scenario_placement" if placement_vri_code else "scenario_classify",
        scenario_id=scenario_id,
        year=year,
        source=source,
        physical_object_type_id=physical_object_type_id,
        objects_count=len(objects_fc.get("features") or []),
        zones_count=len(zones_fc.get("features") or []),
    )
    return task


async def _build_scenario_classification_task(
    *,
    scenario_id: int,
    year: int,
    source: str,
    physical_object_type_id: int,
    priority: int,
    force_recompute: bool,
    idempotency_key_form: str | None,
    idempotency_key_header: str | None,
    token: str,
    app_settings: Settings,
    task_repo: TaskRepository,
    event_repo: EventRepository,
    session: Session,
    zone_descriptions: dict[str, Any] | None = None,
    vri_classifier: dict[str, Any] | list[Any] | None = None,
) -> PipelineTask:
    """Run PZZ-style classification on a scenario's data fetched from urban_api.

    Flow:
      1. Verify (year, source) exists in scenario's functional_zone_sources.
      2. Fetch functional_zones GeoJSON.
      3. Fetch physical_objects_with_geometry GeoJSON (filtered by type).
      4. Flatten urban_api shape → columns the pipeline expects.
      5. Persist both GeoJSONs to MinIO; create a classification task.

    Returns the same TaskOut shape as /tasks/pzz-check — track via
    /tasks/{external_id} and download via /tasks/{external_id}/result.
    """
    if not app_settings.urban_api_base_url:
        raise HTTPException(
            status_code=503,
            detail="urban_api integration is not configured (URBAN_API_BASE_URL is empty).",
        )

    async with UrbanApiClient(
        base_url=app_settings.urban_api_base_url,
        timeout_seconds=app_settings.urban_api_timeout_seconds,
    ) as urban:
        try:
            scenario_info = await urban.get_scenario_info(scenario_id, token=token)
        except UrbanApiError as exc:
            raise HTTPException(status_code=502, detail=f"urban_api: {exc}") from exc

        scenario_updated_at = (scenario_info or {}).get("updated_at") or "unknown"

        auto_key = (
            f"sc:{scenario_id}:{year}:{source}"
            f":type-{physical_object_type_id}"
            f":upd-{scenario_updated_at}"
            f":m-{mapping_version()}"
            f":pv-{PIPELINE_OUTPUT_VERSION}"
        )
        raw_key = idempotency_key_header or idempotency_key_form
        namespaced_key = f"{auto_key}:{raw_key}" if raw_key else auto_key
        if zone_descriptions is not None:
            namespaced_key += f":d-{_json_digest(zone_descriptions)}"
        if vri_classifier is not None:
            namespaced_key += f":c-{_json_digest(vri_classifier)}"

        existing = task_repo.get_by_idempotency_key(namespaced_key)
        if existing is not None and not force_recompute:
            api_log(
                "create_task",
                "cached_hit",
                task_id=existing.id,
                external_id=existing.external_id,
                mode="scenario_classify",
                scenario_id=scenario_id,
                year=year,
                source=source,
                physical_object_type_id=physical_object_type_id,
                scenario_updated_at=scenario_updated_at,
            )
            return existing

        try:
            sources = await urban.list_functional_zone_sources(scenario_id, token=token)
        except UrbanApiError as exc:
            raise HTTPException(status_code=502, detail=f"urban_api: {exc}") from exc

        wants = (int(year), source)
        available = [
            (int(s.get("year")), s.get("source"))
            for s in sources
            if s.get("year") is not None
        ]
        if wants not in available:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "no functional zones for the requested (year, source)",
                    "scenario_id": scenario_id,
                    "requested": {"year": year, "source": source},
                    "available": [{"year": y, "source": s} for y, s in available],
                },
            )

        try:
            zones_fc = await urban.get_functional_zones(
                scenario_id, year=year, source=source, token=token
            )
            objects_fc = await urban.get_physical_objects_with_geometry(
                scenario_id,
                physical_object_type_id=physical_object_type_id,
                token=token,
            )
        except UrbanApiError as exc:
            raise HTTPException(status_code=502, detail=f"urban_api: {exc}") from exc

        # Services are checked alongside buildings (same as the file building
        # check → separate «здания» / «сервисы» result layers). Non-fatal: a
        # services failure must not block the buildings check.
        try:
            services_fc = await urban.get_services_with_geometry(
                scenario_id, token=token
            )
        except UrbanApiError as exc:
            api_log(
                "create_task",
                "services_fetch_failed",
                mode="scenario_classify",
                scenario_id=scenario_id,
                error=str(exc),
            )
            services_fc = {}

        has_services = any(
            f.get("geometry") is not None
            for f in ((services_fc or {}).get("features") or [])
        )
        if (
            not (objects_fc.get("features") or [])
            and not has_services
            and physical_object_type_id == _RESIDENTIAL_PO_TYPE
        ):
            # No houses and no services: check the non-residential buildings
            # instead of failing. Their ВРИ is unknown without services, so
            # they come out as «Не определено» — the user still sees them.
            try:
                objects_fc = await urban.get_physical_objects_with_geometry(
                    scenario_id,
                    physical_object_type_id=NON_RESIDENTIAL_PO_TYPE,
                    token=token,
                )
            except UrbanApiError as exc:
                raise HTTPException(
                    status_code=502, detail=f"urban_api: {exc}"
                ) from exc
            api_log(
                "create_task",
                "non_residential_fallback",
                mode="scenario_classify",
                scenario_id=scenario_id,
                objects=len(objects_fc.get("features") or []),
            )

    service_features = [
        f
        for f in ((services_fc or {}).get("features") or [])
        if f.get("geometry") is not None
    ]
    if service_features:
        objects_fc = {
            **objects_fc,
            "features": list(objects_fc.get("features") or []) + service_features,
        }

    if not (zones_fc.get("features") or []):
        raise HTTPException(
            status_code=422,
            detail=f"urban_api returned no functional zones for year={year} source={source!r}.",
        )
    if not (objects_fc.get("features") or []):
        raise HTTPException(
            status_code=422,
            detail=(
                f"urban_api returned no physical objects of type {physical_object_type_id} "
                f"and no services for scenario {scenario_id}."
            ),
        )

    return await run_in_threadpool(
        _persist_and_create_scenario_task,
        objects_fc=objects_fc,
        zones_fc=zones_fc,
        scenario_id=scenario_id,
        year=year,
        source=source,
        physical_object_type_id=physical_object_type_id,
        priority=priority,
        namespaced_key=namespaced_key,
        force_recompute=force_recompute,
        app_settings=app_settings,
        task_repo=task_repo,
        event_repo=event_repo,
        session=session,
        zone_descriptions=zone_descriptions,
        vri_classifier=vri_classifier,
    )


def _placement_object_texts(
    feature_collection: dict[str, Any], app_settings: Settings
) -> dict[str, Any]:
    """Resolve controlled Urban API types before the placement text classifier."""
    from ..infrastructure.runners._deterministic_pzz import resolve_po_type_vri

    physical_map = json.loads(
        Path(app_settings.physical_object_type_to_vri_path).read_text(encoding="utf-8")
    )
    service_map = json.loads(
        Path(app_settings.service_type_to_vri_path).read_text(encoding="utf-8")
    ).get("by_service_type_id", {})
    fc = _flatten_physical_object_features(
        feature_collection, vri_col=_CADASTRAL_VRI_COL
    )
    for feature in fc.get("features") or []:
        props = feature.get("properties") or {}
        service_type = props.get("service_type")
        code = None
        if isinstance(service_type, dict):
            type_id = service_type.get("service_type_id", service_type.get("id"))
            code = (service_map.get(str(type_id)) or {}).get("vri_code")
        else:
            physical_type = props.get("physical_object_type") or {}
            type_id = physical_type.get(
                "physical_object_type_id", physical_type.get("id")
            )
            nested = props.get("properties") or {}
            floors = (
                nested.get("Количество этажей") if isinstance(nested, dict) else None
            ) or props.get("Количество этажей")
            if type_id is not None:
                code, _ = resolve_po_type_vri(physical_map, int(type_id), floors)
        if code:
            props[_CADASTRAL_VRI_COL] += f" (ВРИ {code})"
    return fc


_JSON_EXTENSIONS = frozenset({".json", ""})


def _read_json_slot(
    upload: UploadFile | None,
    upload_id: str | None,
    field_name: str,
    *,
    owner_id: str,
    app_settings: Settings,
) -> Any | None:
    """Read one optional JSON upload slot (inline file or ``upload_id``).

    Returns the parsed JSON, or ``None`` when the slot is empty. Synchronous —
    run it in a threadpool.
    """
    if upload is None and not upload_id:
        return None
    title = field_title(field_name)
    scratch = uploads_root(app_settings) / f"resolve-{uuid4().hex}"
    try:
        resolved = _resolve_file_slot(
            upload,
            upload_id,
            field_name,
            required=False,
            owner_id=owner_id,
            app_settings=app_settings,
            scratch=scratch,
        )
        if resolved is None:
            return None
        suffix = Path(resolved.filename or "").suffix.lower()
        if suffix not in _JSON_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"{title}: в режиме проекта принимается только JSON.",
            )
        try:
            resolved.file.seek(0)
            raw = resolved.file.read(app_settings.max_upload_bytes + 1)
        finally:
            if upload_id:
                # Our own copy in the scratch dir; close it so rmtree can delete it.
                resolved.file.close()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if len(raw) > app_settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"{title}: файл слишком большой.")
    try:
        return json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422, detail=f"{title}: некорректный JSON ({exc})."
        ) from exc


def _validate_scenario_zone_descriptions(doc: Any) -> dict[str, Any]:
    """Scenario zones are coded by ``functional_zone_type`` id, so only the
    ``functional_zone_mappings`` schema can match them. The worker would
    silently fall back to the built-in mapping on anything else, so refuse it."""
    entries = doc.get("functional_zone_mappings") if isinstance(doc, dict) else None
    usable = [
        e
        for e in (entries if isinstance(entries, list) else [])
        if isinstance(e, dict)
        and str(e.get("functional_zone_type_id", "")).strip().isdigit()
        and isinstance(e.get("averaged_pzz_profile"), dict)
    ]
    if not usable:
        raise HTTPException(
            status_code=422,
            detail=(
                "Описания зон ПЗЗ: в режиме проекта нужен JSON со списком "
                "«functional_zone_mappings» — записи с functional_zone_type_id и "
                "averaged_pzz_profile (main_vri / conditional_vri / auxiliary_vri), "
                "как во встроенном functional_zones_to_pzz_mapping.json."
            ),
        )
    return doc


def _validate_vri_classifier(doc: Any) -> dict[str, Any] | list[Any]:
    if not isinstance(doc, (dict, list)) or not doc:
        raise HTTPException(
            status_code=422,
            detail="Классификатор ВРИ: ожидается непустой JSON-объект или массив.",
        )
    return doc


async def read_scenario_regulation_files(
    *,
    pzz_descriptions_file: UploadFile | None,
    pzz_descriptions_upload_id: str | None,
    vri_classifier_file: UploadFile | None,
    vri_classifier_upload_id: str | None,
    owner_id: str,
    app_settings: Settings,
) -> tuple[dict[str, Any] | None, dict[str, Any] | list[Any] | None]:
    """The user's optional ПЗЗ descriptions and ВРИ classifier, validated."""
    descriptions = await run_in_threadpool(
        _read_json_slot,
        pzz_descriptions_file,
        pzz_descriptions_upload_id,
        "pzz_descriptions_file",
        owner_id=owner_id,
        app_settings=app_settings,
    )
    classifier = await run_in_threadpool(
        _read_json_slot,
        vri_classifier_file,
        vri_classifier_upload_id,
        "vri_classifier_file",
        owner_id=owner_id,
        app_settings=app_settings,
    )
    if descriptions is not None:
        descriptions = _validate_scenario_zone_descriptions(descriptions)
    if classifier is not None:
        classifier = _validate_vri_classifier(classifier)
    return descriptions, classifier


def _json_digest(doc: Any) -> str:
    canonical = json.dumps(doc, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


async def _build_scenario_placement_task(
    *,
    scenario_id: int,
    year: int,
    source: str,
    service_name: str,
    model: str | None,
    priority: int,
    force_recompute: bool,
    idempotency_key_form: str | None,
    idempotency_key_header: str | None,
    token: str,
    app_settings: Settings,
    task_repo: TaskRepository,
    event_repo: EventRepository,
    session: Session,
    zone_descriptions: dict[str, Any] | None = None,
    vri_classifier: dict[str, Any] | list[Any] | None = None,
) -> PipelineTask:
    if not service_name.strip():
        raise HTTPException(
            status_code=422, detail="service_name is required for placement"
        )
    if not app_settings.urban_api_base_url:
        raise HTTPException(
            status_code=503, detail="urban_api integration is not configured."
        )
    # Check access even on cache hits, before resolving the user's service query.
    async with UrbanApiClient(
        app_settings.urban_api_base_url,
        timeout_seconds=app_settings.urban_api_timeout_seconds,
    ) as urban:
        try:
            info = await urban.get_scenario_info(scenario_id, token=token)
            sources = await urban.list_functional_zone_sources(scenario_id, token=token)
            available = [
                {"year": s.get("year"), "source": s.get("source")} for s in sources
            ]
            if not any(
                s.get("year") == year and s.get("source") == source for s in sources
            ):
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "no functional zones for the requested (year, source)",
                        "requested": {"year": year, "source": source},
                        "available": available,
                    },
                )
            resolved = await resolve_service_vri(
                service_name, app_settings, model=model
            )
            if resolved is None:
                raise HTTPException(
                    status_code=422, detail="Не удалось сопоставить сервис с ВРИ."
                )
            # Isolate services sharing the same ВРИ (school / kindergarten),
            # and placements from classifications.
            query_hash = hashlib.sha256(
                service_name.strip().casefold().encode()
            ).hexdigest()[:16]
            namespaced_key = (
                f"sc:{scenario_id}:{year}:{source}:placement:{resolved.code}:{query_hash}"
                f":upd-{(info or {}).get('updated_at', 'unknown')}"
                f":m-{mapping_version()}:pv-{PIPELINE_OUTPUT_VERSION}"
            )
            # The user's own regulation files change the answer, so they
            # must change the cache key too.
            if zone_descriptions is not None:
                namespaced_key += f":d-{_json_digest(zone_descriptions)}"
            if vri_classifier is not None:
                namespaced_key += f":c-{_json_digest(vri_classifier)}"
            raw_key = idempotency_key_header or idempotency_key_form
            if raw_key:
                namespaced_key += f":{raw_key}"
            existing = task_repo.get_by_idempotency_key(namespaced_key)
            if existing is not None and not force_recompute:
                return existing
            zones_fc = await urban.get_functional_zones(
                scenario_id, year=year, source=source, token=token
            )
            # A housing-only filter would hide production buffers and schools.
            objects_fc = await urban.get_physical_objects_with_geometry(
                scenario_id, token=token
            )
            services_fc = await urban.get_services_with_geometry(
                scenario_id, token=token
            )
        except UrbanApiError as exc:
            _raise_urban_api_http_exception(exc)
    if not zones_fc.get("features"):
        raise HTTPException(
            status_code=422, detail="urban_api returned no functional zones."
        )
    objects_fc = {
        "type": "FeatureCollection",
        "features": [
            f
            for fc in (objects_fc, services_fc)
            for f in fc.get("features") or []
            if f.get("geometry") is not None
        ],
    }
    objects_fc = await run_in_threadpool(
        _placement_object_texts, objects_fc, app_settings
    )
    zones_fc = _flatten_functional_zone_features(
        zones_fc, code_col=_PZZ_ZONE_CODE_COL, name_col=_PZZ_ZONE_NAME_COL
    )
    context = await _placement_context(
        app_settings,
        service_name=resolved.service_name or service_name,
        service_query=service_name,
        zones_fc=zones_fc,
        zone_code_col=_PZZ_ZONE_CODE_COL,
        territory_id=None,
        norms_only=True,
    )
    context["input_kind"] = "scenario_objects"
    mapping_note = (
        "по загруженным пользователем описаниям функциональных зон"
        if zone_descriptions is not None
        else "по справочному соответствию функциональных зон видам использования"
    )
    context["notes"].append(
        "Использованы функциональные зоны и объекты сценария Urban API, а не кадастровые "
        f"участки. Допустимость оценена {mapping_note}, а не по границам зон ПЗЗ. "
        "Буферы построены от геометрии объектов сценария; результат ориентировочный."
    )
    if vri_classifier is not None:
        context["notes"].append(
            "Использован загруженный пользователем классификатор ВРИ."
        )
    return await run_in_threadpool(
        _persist_and_create_scenario_task,
        objects_fc=objects_fc,
        zones_fc=zones_fc,
        scenario_id=scenario_id,
        year=year,
        source=source,
        physical_object_type_id=0,
        priority=priority,
        namespaced_key=namespaced_key,
        force_recompute=force_recompute,
        app_settings=app_settings,
        task_repo=task_repo,
        event_repo=event_repo,
        session=session,
        placement_vri_code=resolved.code,
        placement_service_name=resolved.service_name or service_name,
        placement_context=context,
        zone_descriptions=zone_descriptions,
        vri_classifier=vri_classifier,
    )


@router.post("/{scenario_id}/placement", response_model=TaskOut)
async def place_service_in_scenario_endpoint(
    scenario_id: int = FastPath(..., ge=1),
    year: int = Form(..., ge=1900, le=2100),
    source: str = Form(..., min_length=1),
    service_name: str = Form(..., min_length=1),
    model: str | None = Form(default=None),
    priority: int = Form(1, ge=1, le=10),
    force_recompute: bool = Form(False),
    idempotency_key_form: str | None = Form(default=None, alias="Idempotency-Key"),
    idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
    pzz_descriptions_file: UploadFile | None = File(default=None),
    pzz_descriptions_upload_id: str | None = Form(default=None),
    vri_classifier_file: UploadFile | None = File(default=None),
    vri_classifier_upload_id: str | None = Form(default=None),
    token: str = Depends(verify_token),
    user: AuthUser | None = Depends(get_optional_user),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
    event_repo: EventRepository = Depends(get_event_repo),
    session: Session = Depends(get_db),
) -> TaskOut:
    """Place a service using a scenario's functional zones and all its objects.

    Uses the functional-zone reference mapping and NormGraph service norms.
    Physical objects are an approximate substitute for cadastral parcels.

    Optional, like the file-mode endpoints (each as a file or an ``*_upload_id``):

    * ``pzz_descriptions_file`` — your own zone descriptions instead of the
      built-in mapping. Scenario zones are keyed by ``functional_zone_type`` id,
      so it must be JSON in the ``functional_zone_mappings`` schema (same as
      ``functional_zones_to_pzz_mapping.json``).
    * ``vri_classifier_file`` — your own ВРИ classifier JSON.
    """
    zone_descriptions, vri_classifier = await read_scenario_regulation_files(
        pzz_descriptions_file=pzz_descriptions_file,
        pzz_descriptions_upload_id=pzz_descriptions_upload_id,
        vri_classifier_file=vri_classifier_file,
        vri_classifier_upload_id=vri_classifier_upload_id,
        owner_id=user.user_id if user else "",
        app_settings=app_settings,
    )
    task = await _build_scenario_placement_task(
        scenario_id=scenario_id,
        year=year,
        source=source,
        service_name=service_name,
        model=model,
        priority=priority,
        force_recompute=force_recompute,
        idempotency_key_form=idempotency_key_form,
        idempotency_key_header=idempotency_key_header,
        token=token,
        app_settings=app_settings,
        task_repo=task_repo,
        event_repo=event_repo,
        session=session,
        zone_descriptions=zone_descriptions,
        vri_classifier=vri_classifier,
    )
    session.commit()
    return TaskOut.model_validate(task)


@router.post("/{scenario_id}/classify", response_model=TaskOut)
async def classify_scenario_endpoint(
    scenario_id: int = FastPath(..., ge=1),
    year: int = Form(..., ge=1900, le=2100),
    source: str = Form(..., min_length=1),
    physical_object_type_id: int = Form(4),
    priority: int = Form(1, ge=1, le=10),
    force_recompute: bool = Form(False),
    idempotency_key_form: str | None = Form(default=None, alias="Idempotency-Key"),
    idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
    token: str = Depends(verify_token),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
    event_repo: EventRepository = Depends(get_event_repo),
    session: Session = Depends(get_db),
) -> TaskOut:
    """Create a scenario classification task (asynchronous).

    Fetches the scenario's data from urban_api, persists inputs, and enqueues
    the classification. Returns the TaskOut descriptor; track via
    /scenarios/{id}/tasks/{external_id} or the SSE /stream endpoint.
    """
    task = await _build_scenario_classification_task(
        scenario_id=scenario_id,
        year=year,
        source=source,
        physical_object_type_id=physical_object_type_id,
        priority=priority,
        force_recompute=force_recompute,
        idempotency_key_form=idempotency_key_form,
        idempotency_key_header=idempotency_key_header,
        token=token,
        app_settings=app_settings,
        task_repo=task_repo,
        event_repo=event_repo,
        session=session,
    )
    return TaskOut.model_validate(task)


@router.post("/{scenario_id}/classify/stream")
async def classify_scenario_stream_endpoint(
    request: FastRequest,
    scenario_id: int = FastPath(..., ge=1),
    year: int = Form(..., ge=1900, le=2100),
    source: str = Form(..., min_length=1),
    physical_object_type_id: int = Form(4),
    priority: int = Form(1, ge=1, le=10),
    force_recompute: bool = Form(False),
    group_by: str = Form("zone"),
    poll_interval: float = Query(2.0, ge=0.5, le=10.0),
    idempotency_key_form: str | None = Form(default=None, alias="Idempotency-Key"),
    idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
    token: str = Depends(verify_token),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
    event_repo: EventRepository = Depends(get_event_repo),
    session: Session = Depends(get_db),
) -> EventSourceResponse:
    """Create the scenario classification task AND stream it to completion.

    One call does the whole flow: creates (or reuses a cached) task, then
    streams Server-Sent Events until the task is terminal:

      - ``task``        the created task descriptor (keep external_id to
                        reconnect to /stream if the connection drops);
      - ``task_event``  per pipeline event;
      - ``status``      on status changes;
      - ``geojson``     the classified result FeatureCollection when finished
                        (omitted above ``SSE_INLINE_GEOJSON_MAX_BYTES``);
      - ``report``      the object-zone-fit summary when finished;
      - ``done``        terminal marker; the stream then closes.

    Request setup errors (e.g. invalid year/source -> 422) are returned as a
    normal HTTP status BEFORE the stream starts; only in-flight problems are
    delivered as an ``error`` event.

    Note: native EventSource cannot POST or set Authorization — use a
    fetch-based SSE client.
    """
    if group_by not in ("zone", "object"):
        raise HTTPException(
            status_code=422, detail="group_by must be 'zone' or 'object'"
        )

    task = await _build_scenario_classification_task(
        scenario_id=scenario_id,
        year=year,
        source=source,
        physical_object_type_id=physical_object_type_id,
        priority=priority,
        force_recompute=force_recompute,
        idempotency_key_form=idempotency_key_form,
        idempotency_key_header=idempotency_key_header,
        token=token,
        app_settings=app_settings,
        task_repo=task_repo,
        event_repo=event_repo,
        session=session,
    )
    session.commit()
    initial = TaskOut.model_validate(task).model_dump(mode="json")
    return EventSourceResponse(
        task_stream_with_report_generator(
            task.external_id,
            group_by=group_by,
            poll_interval=poll_interval,
            request=request,
            app_settings=app_settings,
            initial=initial,
            emit_input_files=True,
            input_layers_builder=build_scenario_zone_geo_layers,
            scenario=True,
        )
    )


async def _fetch_project_id(
    scenario_id: int,
    token: str,
    app_settings: Settings,
) -> int | str | None:
    """Resolve the scenario's project_id from urban_api (best-effort).

    Used to tag the ChatStorage chat with its project. Returns None when
    urban_api is not configured or the id can't be found — chat persistence
    still works, just without project_id.
    """
    if not app_settings.urban_api_base_url:
        return None
    try:
        async with UrbanApiClient(
            base_url=app_settings.urban_api_base_url,
            timeout_seconds=app_settings.urban_api_timeout_seconds,
        ) as urban:
            info = await urban.get_scenario_info(scenario_id, token=token)
    except UrbanApiError:
        return None

    info = info or {}
    project_id = info.get("project_id")
    if project_id is None:
        project = info.get("project")
        if isinstance(project, dict):
            project_id = project.get("project_id") or project.get("id")
    return project_id


@router.post("/{scenario_id}/chat/stream")
async def scenario_chat_stream_endpoint(
    request: FastRequest,
    scenario_id: int = FastPath(..., ge=1),
    user_query: str | None = Form(default=None, min_length=1),
    mode: str = Form("pzz_check"),
    service_name: str | None = Form(default=None),
    year: int = Form(..., ge=1900, le=2100),
    source: str = Form(..., min_length=1),
    chat_id: str | None = Form(default=None),
    physical_object_type_id: int = Form(4),
    priority: int = Form(1, ge=1, le=10),
    force_recompute: bool = Form(False),
    group_by: str = Form("zone"),
    model: str | None = Form(default=None),
    temperature: float | None = Form(default=None),
    poll_interval: float = Query(2.0, ge=0.5, le=10.0),
    idempotency_key_form: str | None = Form(default=None, alias="Idempotency-Key"),
    idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
    pzz_descriptions_file: UploadFile | None = File(default=None),
    pzz_descriptions_upload_id: str | None = Form(default=None),
    vri_classifier_file: UploadFile | None = File(default=None),
    vri_classifier_upload_id: str | None = Form(default=None),
    user: AuthUser = Depends(get_current_user),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
    event_repo: EventRepository = Depends(get_event_repo),
    session: Session = Depends(get_db),
) -> EventSourceResponse:
    """Classify a scenario or place a service, then stream a grounded answer.

    ``mode=placement`` accepts ``service_name`` (or ``user_query``), fetches all
    scenario objects, and emits ``placement_summary`` and placement map layers.
    Functional zones use the reference mapping; NormGraph supplies service norms.
    ``physical_object_type_id`` only filters the default ``pzz_check`` mode.
    Both modes take optional ``pzz_descriptions_file`` / ``vri_classifier_file``
    (or their ``*_upload_id``). Descriptions use the ``functional_zone_mappings``
    JSON schema, keyed by functional-zone type id. The classifier supplies names
    for the VRI codes resolved from object and service types in ``pzz_check``.

    One call: builds (or reuses a cached) classification task, waits for it to
    finish, then streams an LLM answer to ``user_query`` grounded in the
    object-zone-fit report — and persists the user+assistant turn to
    ChatStorage. ``chat_id`` is optional: when omitted, a new chat is created
    and announced via a ``chat_created`` SSE event so the frontend can store it.

    SSE events: ``task``, ``task_event``, ``status``, ``object_zone_fit``,
    ``chat_created``, ``token``, ``error``, ``done``.

    Note: native EventSource cannot POST or set Authorization — use a
    fetch-based SSE client.
    """
    if group_by not in ("zone", "object"):
        raise HTTPException(
            status_code=422, detail="group_by must be 'zone' or 'object'"
        )

    if mode not in ("pzz_check", "placement"):
        raise HTTPException(
            status_code=422, detail="mode must be 'pzz_check' or 'placement'"
        )
    if mode == "pzz_check" and not user_query:
        raise HTTPException(
            status_code=422, detail="user_query is required for pzz_check"
        )
    if mode == "placement" and not (service_name or user_query or "").strip():
        raise HTTPException(
            status_code=422,
            detail="service_name (or user_query) is required for placement",
        )

    zone_descriptions, vri_classifier = await read_scenario_regulation_files(
        pzz_descriptions_file=pzz_descriptions_file,
        pzz_descriptions_upload_id=pzz_descriptions_upload_id,
        vri_classifier_file=vri_classifier_file,
        vri_classifier_upload_id=vri_classifier_upload_id,
        owner_id=user.user_id,
        app_settings=app_settings,
    )
    if mode == "placement":
        task = await _build_scenario_placement_task(
            scenario_id=scenario_id,
            year=year,
            source=source,
            service_name=service_name or user_query or "",
            model=model,
            priority=priority,
            force_recompute=force_recompute,
            idempotency_key_form=idempotency_key_form,
            idempotency_key_header=idempotency_key_header,
            token=user.token,
            app_settings=app_settings,
            task_repo=task_repo,
            event_repo=event_repo,
            session=session,
            zone_descriptions=zone_descriptions,
            vri_classifier=vri_classifier,
        )
        session.commit()
        project_id = await _fetch_project_id(scenario_id, user.token, app_settings)
        return EventSourceResponse(
            task_stream_with_chat_generator(
                task.external_id,
                group_by="zone",
                poll_interval=poll_interval,
                request=request,
                app_settings=app_settings,
                initial=TaskOut.model_validate(task).model_dump(mode="json"),
                user_id=user.user_id,
                user_query=user_query
                or f"Где разместить {service_name}? Объясни ограничения.",
                chat_id=chat_id,
                scenario_id=scenario_id,
                project_id=project_id,
                chat_title=(user_query or service_name or "Подбор мест размещения")[
                    :256
                ],
                model=model,
                temperature=temperature,
                report_kind="placement",
                emit_input_files=True,
                input_layers_builder=build_scenario_zone_geo_layers,
                system_prompt_path=app_settings.chat_system_prompt_placement_path,
                scenario=True,
            )
        )

    # The user's token is forwarded to urban_api (still fresh at request start);
    # the user's id (sub) is what ChatStorage needs, sent later via the service
    # token + X-User-Id, so persistence survives the token's expiry mid-compute.
    token = user.token
    project_id = await _fetch_project_id(scenario_id, token, app_settings)

    task = await _build_scenario_classification_task(
        scenario_id=scenario_id,
        year=year,
        source=source,
        physical_object_type_id=physical_object_type_id,
        priority=priority,
        force_recompute=force_recompute,
        idempotency_key_form=idempotency_key_form,
        idempotency_key_header=idempotency_key_header,
        token=token,
        app_settings=app_settings,
        task_repo=task_repo,
        event_repo=event_repo,
        session=session,
        zone_descriptions=zone_descriptions,
        vri_classifier=vri_classifier,
    )
    session.commit()
    initial = TaskOut.model_validate(task).model_dump(mode="json")
    return EventSourceResponse(
        task_stream_with_chat_generator(
            task.external_id,
            group_by=group_by,
            poll_interval=poll_interval,
            request=request,
            app_settings=app_settings,
            initial=initial,
            user_id=user.user_id,
            user_query=user_query,
            chat_id=chat_id,
            scenario_id=scenario_id,
            project_id=project_id,
            chat_title=user_query[:256],
            model=model,
            temperature=temperature,
            emit_input_files=True,
            input_layers_builder=build_scenario_zone_geo_layers,
            system_prompt_path=app_settings.chat_system_prompt_scenario_path,
            scenario=True,
        )
    )


@router.get("/{scenario_id}/zones-info")
async def get_scenario_zones_info(
    scenario_id: int = FastPath(..., ge=1),
    year: int = Query(..., ge=1900, le=2100),
    source: str = Query(..., min_length=1),
    token: str = Depends(verify_token),
    app_settings: Settings = Depends(get_app_settings),
) -> dict[str, Any]:
    """Return a simplified description of the scenario's functional zones.

    Light wrapper over urban_api's /functional_zones — flattens the
    response into one entry per zone with id, type, name and free-form
    properties (which may include a textual description depending on the
    data source). The frontend / chatbot can use this to show "what's
    allowed here" hints.

    Each zone is enriched with ``pzz_summary`` from the static
    functional_zone -> PZZ mapping: a human-readable "что можно строить"
    summary plus structured permitted/conditional/auxiliary VRI lists.
    ``pzz_summary.mapping_status == "no_mapping"`` means we have no PZZ
    analogue for that zone type.
    """
    if not app_settings.urban_api_base_url:
        raise HTTPException(
            status_code=503,
            detail="urban_api integration is not configured.",
        )

    async with UrbanApiClient(
        base_url=app_settings.urban_api_base_url,
        timeout_seconds=app_settings.urban_api_timeout_seconds,
    ) as urban:
        try:
            zones_fc = await urban.get_functional_zones(
                scenario_id, year=year, source=source, token=token
            )
        except UrbanApiError as exc:
            raise HTTPException(status_code=502, detail=f"urban_api: {exc}") from exc

    items: list[dict[str, Any]] = []
    for feature in zones_fc.get("features") or []:
        props = feature.get("properties") or {}
        zone_type = props.get("functional_zone_type") or {}
        if isinstance(zone_type, dict):
            zone_type_id = zone_type.get("id")
            zone_type_name = functional_zone_type_label(zone_type)
        else:
            zone_type_id = None
            zone_type_name = None
        items.append(
            {
                "functional_zone_id": props.get("functional_zone_id"),
                "zone_type_id": zone_type_id,
                "zone_type_name": zone_type_name,
                "name": props.get("name"),
                "year": props.get("year"),
                "source": props.get("source"),
                "properties": props.get("properties"),
                "pzz_summary": lookup_zone_summary(zone_type_id),
            }
        )

    return {
        "scenario_id": scenario_id,
        "year": year,
        "source": source,
        "total": len(items),
        "items": items,
    }


@router.get("/{scenario_id}/tasks/{external_id}", response_model=TaskOut)
async def get_scenario_task_endpoint(
    scenario_id: int = FastPath(..., ge=1),
    external_id: str = FastPath(..., min_length=1),
    token: str = Depends(verify_token),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
) -> TaskOut:
    """Return task status only after the token is allowed to see the scenario."""
    await _verify_scenario_access(
        scenario_id=scenario_id,
        token=token,
        app_settings=app_settings,
    )
    task = _get_scenario_task_or_404(
        scenario_id=scenario_id,
        external_id=external_id,
        task_repo=task_repo,
    )
    return TaskOut.model_validate(task)


@router.get("/{scenario_id}/tasks/{external_id}/placement-summary")
async def get_scenario_placement_summary_endpoint(
    scenario_id: int = FastPath(..., ge=1),
    external_id: str = FastPath(..., min_length=1),
    token: str = Depends(verify_token),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
) -> dict[str, Any]:
    """Placement summary after checking scenario access and task membership."""
    await _verify_scenario_access(
        scenario_id=scenario_id, token=token, app_settings=app_settings
    )
    task = _get_scenario_task_or_404(
        scenario_id=scenario_id, external_id=external_id, task_repo=task_repo
    )
    return build_placement_summary_response(task, external_id, app_settings)


@router.get("/{scenario_id}/tasks/{external_id}/result")
async def get_scenario_task_result_endpoint(
    scenario_id: int = FastPath(..., ge=1),
    external_id: str = FastPath(..., min_length=1),
    token: str = Depends(verify_token),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
) -> FileResponse:
    """Stream a scenario task result after checking access to the scenario."""
    await _verify_scenario_access(
        scenario_id=scenario_id,
        token=token,
        app_settings=app_settings,
    )
    task = _get_scenario_task_or_404(
        scenario_id=scenario_id,
        external_id=external_id,
        task_repo=task_repo,
    )
    return build_task_result_response(task, external_id, app_settings)


@router.get("/{scenario_id}/tasks/{external_id}/object-zone-fit")
async def get_scenario_task_object_zone_fit_endpoint(
    scenario_id: int = FastPath(..., ge=1),
    external_id: str = FastPath(..., min_length=1),
    group_by: str = Query("zone", pattern="^(zone|object)$"),
    token: str = Depends(verify_token),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
) -> dict[str, Any]:
    """Return scenario task object-zone summary after checking scenario access."""
    await _verify_scenario_access(
        scenario_id=scenario_id,
        token=token,
        app_settings=app_settings,
    )
    task = _get_scenario_task_or_404(
        scenario_id=scenario_id,
        external_id=external_id,
        task_repo=task_repo,
    )
    return build_object_zone_fit_response(
        task, external_id, group_by, app_settings, scenario=True
    )


@router.delete("/{scenario_id}/tasks/{external_id}", response_model=TaskOut)
async def cancel_scenario_task_endpoint(
    scenario_id: int = FastPath(..., ge=1),
    external_id: str = FastPath(..., min_length=1),
    token: str = Depends(verify_token),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
    event_repo: EventRepository = Depends(get_event_repo),
    session: Session = Depends(get_db),
) -> TaskOut:
    """Cancel a scenario task after checking access to its scenario.

    Behaviour mirrors ``DELETE /tasks/{external_id}`` (revokes Celery
    message for queued tasks, SIGTERMs the worker for running ones, 409
    on already-terminal tasks) — only the auth+ownership check differs.
    """
    await _verify_scenario_access(
        scenario_id=scenario_id,
        token=token,
        app_settings=app_settings,
    )
    task = _get_scenario_task_or_404(
        scenario_id=scenario_id,
        external_id=external_id,
        task_repo=task_repo,
    )
    return build_cancel_task_response(task, task_repo, event_repo, session)


@router.post("/{scenario_id}/tasks/{external_id}/recompute", response_model=TaskOut)
async def recompute_scenario_task_endpoint(
    scenario_id: int = FastPath(..., ge=1),
    external_id: str = FastPath(..., min_length=1),
    token: str = Depends(verify_token),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
    event_repo: EventRepository = Depends(get_event_repo),
    session: Session = Depends(get_db),
) -> TaskOut:
    """Force-recompute a terminal scenario task using its stored input paths.

    Same semantics as ``POST /tasks/{external_id}/recompute`` — re-runs
    the existing task with the cached MinIO inputs, without re-fetching
    from urban_api. If you need a fresh pull from urban_api (data version
    bumped), call ``POST /scenarios/{id}/classify`` with
    ``force_recompute=true`` instead.
    """
    await _verify_scenario_access(
        scenario_id=scenario_id,
        token=token,
        app_settings=app_settings,
    )
    task = _get_scenario_task_or_404(
        scenario_id=scenario_id,
        external_id=external_id,
        task_repo=task_repo,
    )
    return build_recompute_task_response(task, task_repo, event_repo, session)


def _parse_scenario_idempotency_key(key: str) -> tuple[int, str, int] | None:
    """Parse (year, source, physical_object_type_id) from sc: idempotency key.

    Key format: sc:{scenario_id}:{year}:{source}:type-{po_type_id}:upd-{...}:m-{...}
    """
    if not key or not key.startswith("sc:"):
        return None
    parts = key.split(":")
    try:
        year = int(parts[2])
        source = parts[3]
        po_type_id = 0 if parts[4] == "placement" else int(parts[4].split("-", 1)[1])
        return year, source, po_type_id
    except (ValueError, IndexError):
        return None


async def _scenario_task_sse_generator(
    scenario_id: int,
    external_id: str,
    poll_interval: float,
    request: FastRequest,
    token: str,
    app_settings: Settings,
) -> AsyncIterator[ServerSentEvent]:
    # 1. Parse year/source from idempotency key and emit zones_info immediately
    with session_scope() as session:
        repo = SqlAlchemyTaskRepository(session)
        idempotency_key = repo.get_idempotency_key_by_external_id(external_id) or ""

    parsed = _parse_scenario_idempotency_key(idempotency_key)
    if parsed and app_settings.urban_api_base_url:
        year, source, _ = parsed
        try:
            async with UrbanApiClient(
                base_url=app_settings.urban_api_base_url,
                timeout_seconds=app_settings.urban_api_timeout_seconds,
            ) as urban:
                zones_fc = await urban.get_functional_zones(
                    scenario_id, year=year, source=source, token=token
                )
            items: list[dict[str, Any]] = []
            for feature in zones_fc.get("features") or []:
                props = feature.get("properties") or {}
                zone_type = props.get("functional_zone_type") or {}
                if isinstance(zone_type, dict):
                    zone_type_id = zone_type.get("id")
                    zone_type_name = functional_zone_type_label(zone_type)
                else:
                    zone_type_id = None
                    zone_type_name = None
                items.append(
                    {
                        "functional_zone_id": props.get("functional_zone_id"),
                        "zone_type_id": zone_type_id,
                        "zone_type_name": zone_type_name,
                        "name": props.get("name"),
                        "year": props.get("year"),
                        "source": props.get("source"),
                        "properties": props.get("properties"),
                        "pzz_summary": lookup_zone_summary(zone_type_id),
                    }
                )
            yield ServerSentEvent(
                data=json.dumps(
                    {
                        "scenario_id": scenario_id,
                        "year": year,
                        "source": source,
                        "total": len(items),
                        "items": items,
                    }
                ),
                event="zones_info",
            )
        except Exception:
            pass

    # 2. Poll for task events and status changes
    last_event_id = 0
    last_status: TaskStatus | None = None

    while True:
        if await request.is_disconnected():
            break

        with session_scope() as session:
            task = session.execute(
                select(PipelineTask).where(PipelineTask.external_id == external_id)
            ).scalar_one_or_none()

            if task is None:
                yield ServerSentEvent(
                    data=json.dumps({"error": "Task not found"}), event="error"
                )
                break

            new_events = (
                session.execute(
                    select(TaskEvent)
                    .where(TaskEvent.task_id == task.id, TaskEvent.id > last_event_id)
                    .order_by(TaskEvent.id.asc())
                )
                .scalars()
                .all()
            )

            for ev in new_events:
                last_event_id = ev.id
                yield ServerSentEvent(
                    data=json.dumps(
                        TaskEventOut.model_validate(ev).model_dump(mode="json")
                    ),
                    event="task_event",
                )

            current_status = task.status
            if current_status != last_status:
                last_status = current_status
                yield ServerSentEvent(
                    data=json.dumps(
                        TaskOut.model_validate(task).model_dump(mode="json")
                    ),
                    event="status",
                )

            if current_status in _TERMINAL_STATUSES:
                # 3. On success emit object_zone_fit and result
                if current_status == TaskStatus.finished:
                    try:
                        if task.placement_vri_code:
                            ozone = build_placement_summary_response(
                                task, external_id, app_settings
                            )
                        else:
                            ozone = build_object_zone_fit_response(
                                task, external_id, "zone", app_settings, scenario=True
                            )
                        yield ServerSentEvent(
                            data=json.dumps(ozone, default=str),
                            event=(
                                "placement_summary"
                                if task.placement_vri_code
                                else "object_zone_fit"
                            ),
                        )
                    except Exception:
                        pass
                    try:
                        from pathlib import Path
                        from ..infrastructure.storage import (
                            get_object_storage,
                            is_remote_path,
                        )

                        result_path = task.result_path
                        if result_path:
                            if is_remote_path(result_path):
                                outputs_dir = Path(app_settings.outputs_dir)
                                outputs_dir.mkdir(parents=True, exist_ok=True)
                                cache_path = outputs_dir / result_path.split("/")[-1]
                                if not cache_path.is_file():
                                    get_object_storage().download_file(
                                        result_path, str(cache_path)
                                    )
                                result_path = str(cache_path)
                            yield ServerSentEvent(
                                data=Path(result_path).read_text(encoding="utf-8"),
                                event="result",
                            )
                    except Exception:
                        pass

                yield ServerSentEvent(
                    data=json.dumps({"status": current_status.value}),
                    event="done",
                )
                break

        await asyncio.sleep(poll_interval)


@router.get("/{scenario_id}/tasks/{external_id}/stream")
async def stream_scenario_task_status_endpoint(
    request: FastRequest,
    scenario_id: int = FastPath(..., ge=1),
    external_id: str = FastPath(..., min_length=1),
    poll_interval: float = Query(2.0, ge=0.5, le=10.0),
    token: str = Depends(verify_token),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
) -> EventSourceResponse:
    """Stream scenario task status and events via Server-Sent Events.

    Emits the following event types in order:
    - ``zones_info`` — functional zone data from urban_api (sent immediately on connect)
    - ``task_event`` — each new pipeline event
    - ``status`` — task status on every change
    - ``object_zone_fit`` — zone/object fit summary when task finishes
    - ``done`` — terminal event (``{"status": "finished"|"failed"}``), stream closes
    """
    await _verify_scenario_access(
        scenario_id=scenario_id,
        token=token,
        app_settings=app_settings,
    )
    _get_scenario_task_or_404(
        scenario_id=scenario_id,
        external_id=external_id,
        task_repo=task_repo,
    )
    return EventSourceResponse(
        _scenario_task_sse_generator(
            scenario_id, external_id, poll_interval, request, token, app_settings
        )
    )


@router.get(
    "/{scenario_id}/tasks/{external_id}/events",
    response_model=list[TaskEventOut],
)
async def get_scenario_task_events_endpoint(
    scenario_id: int = FastPath(..., ge=1),
    external_id: str = FastPath(..., min_length=1),
    token: str = Depends(verify_token),
    app_settings: Settings = Depends(get_app_settings),
    task_repo: TaskRepository = Depends(get_task_repo),
    session: Session = Depends(get_db),
) -> list[TaskEventOut]:
    """Return the event timeline for a scenario task (after access check)."""
    await _verify_scenario_access(
        scenario_id=scenario_id,
        token=token,
        app_settings=app_settings,
    )
    task = _get_scenario_task_or_404(
        scenario_id=scenario_id,
        external_id=external_id,
        task_repo=task_repo,
    )
    return build_task_events_response(task, session)
