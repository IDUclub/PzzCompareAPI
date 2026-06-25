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

from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Form, Header, HTTPException, Path as FastPath, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from starlette.responses import FileResponse

from ..application.use_cases.create_task import create_task
from ..auth.exceptions import AuthError
from ..dependencies import get_app_settings, get_auth_client, get_db, get_event_repo, get_task_repo
from ..domain.ports.event_repository import EventRepository
from ..domain.ports.task_repository import TaskRepository
from ..infrastructure.pzz_mapping import (
    build_pipeline_zone_labels,
    lookup_zone_summary,
    mapping_version,
)
from ..infrastructure.storage import get_object_storage
from ..infrastructure.urban_api_client import UrbanApiClient, UrbanApiError
from ..schemas import TaskCreate, TaskEventOut, TaskOut
from ..settings import Settings
from ..tasks import execute_pipeline_task
from .classifier import persist_geojson_dict
from fastapi import Request as FastRequest
from sse_starlette.sse import EventSourceResponse

from .tasks import (
    _task_sse_generator,
    build_cancel_task_response,
    build_object_zone_fit_response,
    build_recompute_task_response,
    build_task_events_response,
    build_task_result_response,
    get_task_or_404,
    task_stream_with_report_generator,
)
from ..models import PipelineTask
from .utils import api_log

router = APIRouter(prefix="/scenarios", tags=["scenarios"])
http_bearer = HTTPBearer()


_CADASTRAL_VRI_COL = "vri_text"
_PZZ_ZONE_CODE_COL = "zone_code"
_PZZ_ZONE_NAME_COL = "zone_name"
_SCENARIO_IDEMPOTENCY_PREFIX = "sc:"


def _get_token_from_header(credentials: HTTPAuthorizationCredentials) -> str:
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing",
        )

    token = credentials.credentials
    if not token:
        raise HTTPException(
            status_code=400,
            detail="Token is missing in the authorization header",
        )

    return token


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer),
) -> str:
    """Extract the Bearer token and verify it (Keycloak JWT) when enabled.

    When ``AUTH_VERIFY`` is false the token is accepted as-is (urban_api
    validates it downstream). When true, the signature + claims are checked
    against the realm JWKS; a rejected token yields 401 so the caller can
    refresh it.
    """
    token = _get_token_from_header(credentials)
    auth_client = get_auth_client()
    if auth_client.config.verify:
        try:
            await auth_client.get_user_from_token(token)
        except AuthError as exc:
            raise HTTPException(status_code=401, detail=exc.detail) from exc
    return token


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
        raise HTTPException(status_code=exc.status, detail=f"urban_api: {exc.body}") from exc
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
        po_type = props.get("physical_object_type") or {}
        type_name = (po_type.get("name") if isinstance(po_type, dict) else None) or ""
        nested = props.get("properties") if isinstance(props.get("properties"), dict) else {}
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
    object. We use the type id as the code (stable per scenario) and
    prefer ``nickname`` (Russian human-readable name like «Жилая зона»)
    over ``name`` (English code like "residential") so downstream outputs
    look natural in chat messages and reports.
    """
    features = feature_collection.get("features") or []
    for feature in features:
        props = feature.setdefault("properties", {})
        zone_type = props.get("functional_zone_type") or {}
        if isinstance(zone_type, dict):
            zone_type_id = zone_type.get("id")
            zone_type_name = zone_type.get("nickname") or zone_type.get("name") or ""
        else:
            zone_type_id = None
            zone_type_name = ""
        props[code_col] = str(zone_type_id) if zone_type_id is not None else ""
        props[name_col] = zone_type_name
    return feature_collection


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
        )
        raw_key = idempotency_key_header or idempotency_key_form
        namespaced_key = f"{auto_key}:{raw_key}" if raw_key else auto_key

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
        available = [(int(s.get("year")), s.get("source")) for s in sources if s.get("year") is not None]
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
                f"for scenario {scenario_id}."
            ),
        )

    objects_fc = _flatten_physical_object_features(objects_fc, vri_col=_CADASTRAL_VRI_COL)
    zones_fc = _flatten_functional_zone_features(
        zones_fc, code_col=_PZZ_ZONE_CODE_COL, name_col=_PZZ_ZONE_NAME_COL
    )

    external_id = uuid4().hex
    task_dir = Path(app_settings.task_inputs_dir) / external_id
    storage = get_object_storage()

    stored_cadastral = persist_geojson_dict(
        objects_fc, task_dir, "cadastral_feature_collection.geojson", external_id, storage
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
    scenario_labels = build_pipeline_zone_labels(observed_zone_types=observed_zone_types)
    stored_labels = persist_geojson_dict(
        scenario_labels, task_dir, "pzz_zone_vri_labels.json", external_id, storage
    )

    input_paths = {
        "cadastral_data_path": stored_cadastral,
        "pzz_zones_data_path": stored_pzz_zones,
        "pzz_zone_vri_labels_path": stored_labels,
        "vri_classifier_path": str(Path(app_settings.default_vri_classifier_path).resolve()),
    }

    payload = TaskCreate(
        include_pzz_check=True,
        cadastral_vri_col=_CADASTRAL_VRI_COL,
        pzz_zone_code_col=_PZZ_ZONE_CODE_COL,
        pzz_zone_name_col=_PZZ_ZONE_NAME_COL,
        priority=priority,
    )

    task = create_task(
        payload=payload,
        settings=app_settings,
        task_repo=task_repo,
        event_repo=event_repo,
        enqueue_task=execute_pipeline_task.delay,
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
        mode="scenario_classify",
        scenario_id=scenario_id,
        year=year,
        source=source,
        physical_object_type_id=physical_object_type_id,
        objects_count=len(objects_fc.get("features") or []),
        zones_count=len(zones_fc.get("features") or []),
    )
    return task


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
      - ``geojson``     the classified result FeatureCollection when finished;
      - ``report``      the object-zone-fit summary when finished;
      - ``done``        terminal marker; the stream then closes.

    Request setup errors (e.g. invalid year/source -> 422) are returned as a
    normal HTTP status BEFORE the stream starts; only in-flight problems are
    delivered as an ``error`` event.

    Note: native EventSource cannot POST or set Authorization — use a
    fetch-based SSE client.
    """
    if group_by not in ("zone", "object"):
        raise HTTPException(status_code=422, detail="group_by must be 'zone' or 'object'")

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
    # Commit so the streaming generator (its own DB session) sees the task and
    # the Celery worker can pick it up.
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
            zone_type_name = zone_type.get("nickname") or zone_type.get("name")
        else:
            zone_type_id = None
            zone_type_name = None
        items.append({
            "functional_zone_id": props.get("functional_zone_id"),
            "zone_type_id": zone_type_id,
            "zone_type_name": zone_type_name,
            "name": props.get("name"),
            "year": props.get("year"),
            "source": props.get("source"),
            "properties": props.get("properties"),
            "pzz_summary": lookup_zone_summary(zone_type_id),
        })

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
    return build_object_zone_fit_response(task, external_id, group_by, app_settings)


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
    """Stream scenario task status and events via Server-Sent Events."""
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
    return EventSourceResponse(_task_sse_generator(external_id, poll_interval, request))


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
