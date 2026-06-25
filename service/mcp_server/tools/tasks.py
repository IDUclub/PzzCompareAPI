"""MCP tools that operate on pipeline tasks (submit / monitor / fetch result).

Each tool maps 1-to-1 to a REST endpoint of the FastAPI service. The
descriptions are written for an LLM consumer — they explain *when* the
tool should be called, list params and result shape, and include a
worked example so the model can match patterns instead of inferring
intent from prose.

Upstream errors are translated to JSON-RPC error codes via
``@map_errors`` so the LLM sees actionable messages, not stack traces.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.dependencies import Depends

from ..api_client import ApiClient
from ..dependencies import get_api_client
from ..exceptions import map_errors


tasks_mcp = FastMCP("PZZ Pipeline Tasks")


def _result_not_ready_response(
    *,
    external_id: str,
    status: str | None,
    error_text: str | None = None,
) -> dict[str, Any]:
    failed = status == "failed"
    return {
        "external_id": external_id,
        "ready": False,
        "status": status,
        "error_text": error_text,
        "next_step": (
            "The task failed; GeoJSON result is unavailable. Show error_text "
            "to the user or call recompute_task after fixing the cause."
            if failed
            else "Poll get_task_status until status is finished, then call "
            "get_task_result again."
        ),
    }


@tasks_mcp.tool(
    name="submit_classify_only_task",
    title="Submit classification-only task",
    description="""Submit cadastral parcels for VRI classification against the federal Rosreestr classifier.

USE WHEN: user has cadastral parcels and wants their permitted-use texts classified, but does NOT have PZZ zone polygons.

PARAMETERS
- cadastral_geojson (object, required): GeoJSON FeatureCollection in EPSG:4326. Each Feature must have the property named by `cadastral_vri_col`.
- cadastral_vri_col (string, required): name of the property in each Feature that holds the cadastral VRI text. Examples: "Вид разреш", "Вид_разрешенного_исп".
- priority (int, 1–10, default 1): scheduling priority. Higher = more capacity reserved.
- force_recompute (bool, default false): when true and an Idempotency-Key matches an existing terminal task, the task is re-enqueued instead of returning the cached result.

RETURNS
- TaskOut object with at least: external_id (string), status (queued/waiting_capacity/running/finished/failed), created_at, priority, cadastral_vri_col.

EXAMPLE CALL
  submit_classify_only_task(
    cadastral_geojson={"type":"FeatureCollection","features":[...]},
    cadastral_vri_col="Вид разреш",
    priority=1
  )

EXAMPLE RESULT
  {"external_id":"a1b2c3d4...","status":"queued","priority":1,"cadastral_vri_col":"Вид разреш", ...}

ERRORS
- -32602 Invalid params: GeoJSON malformed or column name empty.
- -32603 Internal error: upstream API or storage unreachable.

NEXT STEP: poll get_task_status(external_id) until status is "finished" or "failed".""",
    tags={"pipeline", "submit"},
)
@map_errors
async def submit_classify_only_task(
    cadastral_geojson: Annotated[
        dict[str, Any],
        "GeoJSON FeatureCollection of cadastral parcels in EPSG:4326.",
    ],
    cadastral_vri_col: Annotated[
        str,
        "Property name holding the cadastral VRI text (e.g. 'Вид разреш').",
    ],
    priority: Annotated[int, "Scheduling priority 1-10."] = 1,
    force_recompute: Annotated[bool, "Force re-run when Idempotency-Key matches."] = False,
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
    return await api.submit_classify_only(
        cadastral_geojson=cadastral_geojson,
        cadastral_vri_col=cadastral_vri_col,
        priority=priority,
        force_recompute=force_recompute,
    )


@tasks_mcp.tool(
    name="submit_pzz_check_task",
    title="Submit full PZZ compliance check",
    description="""Submit cadastral parcels for the full PZZ compliance check (spatial overlay + classification).

USE WHEN: user has BOTH cadastral parcels AND the PZZ zone polygons for the same territory. The pipeline overlays parcels onto PZZ zones to determine each parcel's factual zone, then validates its VRI text against that zone's permitted-use list.

PARAMETERS
- cadastral_geojson (object, required): GeoJSON FeatureCollection in EPSG:4326.
- pzz_zones_geojson (object, required): GeoJSON FeatureCollection in EPSG:4326 with zone polygons.
- cadastral_vri_col (string, required): name of the property holding the cadastral VRI text in cadastral features.
- pzz_zone_code_col (string, required): name of the property in PZZ features holding the zone code (e.g. "Индекс_зоны").
- pzz_zone_name_col (string, required): name of the property in PZZ features holding the human-readable zone name (e.g. "Код_объекта").
- priority (int, 1–10, default 1).
- force_recompute (bool, default false).

RETURNS
- TaskOut object with external_id, status, include_pzz_check=true, column names.

EXAMPLE CALL
  submit_pzz_check_task(
    cadastral_geojson={...},
    pzz_zones_geojson={...},
    cadastral_vri_col="Вид разреш",
    pzz_zone_code_col="Индекс_зоны",
    pzz_zone_name_col="Код_объекта"
  )

ERRORS
- -32602 Invalid params: GeoJSON not in EPSG:4326, missing columns, or empty.
- -32603 Internal error: upstream API or storage unreachable.

NEXT STEP: poll get_task_status(external_id) until terminal status, then call get_task_result.""",
    tags={"pipeline", "submit"},
)
@map_errors
async def submit_pzz_check_task(
    cadastral_geojson: Annotated[dict[str, Any], "Cadastral parcels GeoJSON in EPSG:4326."],
    pzz_zones_geojson: Annotated[dict[str, Any], "PZZ zones GeoJSON in EPSG:4326."],
    cadastral_vri_col: Annotated[str, "Property name with cadastral VRI text."],
    pzz_zone_code_col: Annotated[str, "Property name with PZZ zone code."],
    pzz_zone_name_col: Annotated[str, "Property name with PZZ zone human-readable name."],
    priority: Annotated[int, "Scheduling priority 1-10."] = 1,
    force_recompute: Annotated[bool, "Force re-run."] = False,
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
    return await api.submit_pzz_check(
        cadastral_geojson=cadastral_geojson,
        pzz_zones_geojson=pzz_zones_geojson,
        cadastral_vri_col=cadastral_vri_col,
        pzz_zone_code_col=pzz_zone_code_col,
        pzz_zone_name_col=pzz_zone_name_col,
        priority=priority,
        force_recompute=force_recompute,
    )


@tasks_mcp.tool(
    name="get_task_status",
    title="Get task status",
    description="""Return the current status descriptor of a task.

USE WHEN: polling a task created by submit_* to detect when it reaches a terminal state.

PARAMETERS
- external_id (string, required): identifier returned by submit_*.

RETURNS
- TaskOut: external_id, status (queued / waiting_capacity / running / finished / failed), created_at, started_at, finished_at, result_path (when finished), error_text (when failed).

ERRORS
- -32602 Invalid params: unknown external_id (404 from API).""",
    tags={"pipeline", "read"},
    annotations={"readOnlyHint": True},
)
@map_errors
async def get_task_status(
    external_id: Annotated[str, "Task identifier from submit_*."],
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
    return await api.get_task(external_id)


@tasks_mcp.tool(
    name="list_tasks",
    title="List tasks",
    description="""List recent tasks with optional status filter and offset/limit pagination.

USE WHEN: surveying queue state, finding the most recent finished task, or auditing failures.

PARAMETERS
- status (string, optional): one of queued / waiting_capacity / running / finished / failed.
- limit (int, default 20, max 100).
- offset (int, default 0).

RETURNS
- TaskListOut: { items: [TaskOut, ...], total, limit, offset }.

EXAMPLE
  list_tasks(status="failed", limit=10) → 10 most recent failed tasks.""",
    tags={"pipeline", "read"},
    annotations={"readOnlyHint": True},
)
@map_errors
async def list_tasks(
    status: Annotated[
        str | None,
        "Filter by status. Omit for all statuses.",
    ] = None,
    limit: Annotated[int, "Page size, 1-100."] = 20,
    offset: Annotated[int, "Pagination offset."] = 0,
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
    return await api.list_tasks(status=status, limit=limit, offset=offset)


@tasks_mcp.tool(
    name="get_task_events",
    title="Get task event timeline",
    description="""Return the chronological event timeline of a task.

USE WHEN: debugging a slow or failed task — events log stage transitions, capacity wait, queue retries, etc.

PARAMETERS
- external_id (string, required).

RETURNS
- list of TaskEventOut: { stage, status, details, created_at }.""",
    tags={"pipeline", "read"},
    annotations={"readOnlyHint": True},
)
@map_errors
async def get_task_events(
    external_id: Annotated[str, "Task identifier."],
    api: ApiClient = Depends(get_api_client),
) -> list[dict[str, Any]]:
    return await api.get_task_events(external_id)


@tasks_mcp.tool(
    name="get_task_result",
    title="Download task result",
    description="""Download and return the result GeoJSON of a finished task as a parsed JSON object.

USE WHEN: the task is in `finished` status and the user wants to inspect or summarise the classification result.

PARAMETERS
- external_id (string, required).

RETURNS
- GeoJSON FeatureCollection. Each Feature carries the original parcel geometry plus added classification columns (Top1 / Top5 candidates, verdict, reason, etc.).

WARNING
- Result may be large (MBs). For raw download prefer the HTTP endpoint /tasks/{external_id}/result with a streaming client.

ERRORS
- -32602 Invalid params: task not found.

If the task is still queued/running, or if it failed, this tool returns
{ ready:false, status, error_text, next_step } instead of raising a tool
error.""",
    tags={"pipeline", "read"},
    annotations={"readOnlyHint": True},
)
@map_errors
async def get_task_result(
    external_id: Annotated[str, "Task identifier."],
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
    task = await api.get_task(external_id)
    status = task.get("status")
    if status != "finished":
        return _result_not_ready_response(
            external_id=external_id,
            status=status,
            error_text=task.get("error_text"),
        )

    return await api.get_task_result(external_id)


@tasks_mcp.tool(
    name="cancel_task",
    title="Cancel an active task",
    description="""Cancel a task that is queued, waiting for capacity, or running.

USE WHEN: user wants to stop a task before it finishes. The worker process is SIGTERM'd for running tasks.

PARAMETERS
- external_id (string, required).

RETURNS
- TaskOut with status updated to `failed` and error_text "Cancelled by client".

ERRORS
- -32602 Invalid params: task already in terminal state (409).""",
    tags={"pipeline", "control"},
)
@map_errors
async def cancel_task(
    external_id: Annotated[str, "Task identifier."],
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
    return await api.cancel_task(external_id)


@tasks_mcp.tool(
    name="recompute_task",
    title="Force re-run a terminal task",
    description="""Re-enqueue a task that already reached a terminal state (finished or failed).

USE WHEN: user wants to re-run an existing task without re-uploading files (e.g. after fixing a config issue or after a re-deploy of the pipeline).

PARAMETERS
- external_id (string, required).

RETURNS
- TaskOut with status `queued` and new celery_task_id; result_path / error_text / timestamps are cleared.

ERRORS
- -32602 Invalid params: task is still active (queued / running / waiting_capacity).""",
    tags={"pipeline", "control"},
)
@map_errors
async def recompute_task(
    external_id: Annotated[str, "Task identifier."],
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
    return await api.recompute_task(external_id)
