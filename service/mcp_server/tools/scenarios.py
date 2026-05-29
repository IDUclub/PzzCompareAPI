"""MCP tools for the scenario classification flow (urban_api-backed).

Parameter split, per IDUclub convention:

- **Hidden metadata** (NOT visible to the LLM) is read from the MCP
  request ``_meta`` and the HTTP ``Authorization`` header:
    * ``scenario_id`` — which project/scenario the user has open; injected
      by the frontend into ``_meta``.
    * Bearer token — the user's urban_api JWT, forwarded to our REST layer
      which forwards it to urban_api.
  These depend on *where the user is* and must not be hallucinated by the
  model, so they never appear in the tool schema.

- **Visible parameters** (the LLM can reason about / ask the user / read
  from urban_api): ``year``, ``source``, ``physical_object_type_id``,
  ``priority``, ``force_recompute``, ``external_id``, ``group_by``.

Long runs report progress via ``ctx.report_progress`` so a client on the
streamable-HTTP transport sees live updates.
"""
from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from fastmcp.server.dependencies import get_http_headers
from mcp import ErrorData, McpError

from ..api_client import ApiClient
from ..dependencies import get_api_client
from ..exceptions import map_errors


scenarios_mcp = FastMCP("PZZ Pipeline Scenarios")


def extract_token() -> str | None:
    """Pull the user's Bearer token from the incoming HTTP Authorization header.

    ``get_http_headers`` strips ``authorization`` by default, so we ask for
    it explicitly. Returns None when absent (no HTTP request / no header).
    """
    headers = get_http_headers(include={"authorization"})
    auth = headers.get("authorization", "")
    return auth[7:].strip() if auth.startswith("Bearer ") else None


def _scenario_id_from_meta(ctx: Context) -> int:
    """Read scenario_id from request _meta; raise a clear error if missing.

    The frontend must inject ``scenario_id`` into the MCP request ``_meta``
    (it knows which project the user has open). We never let the LLM supply
    it — that would let the model classify an arbitrary scenario.
    """
    rc = getattr(ctx, "request_context", None)
    meta = getattr(rc, "meta", None) if rc else None
    raw = getattr(meta, "scenario_id", None) if meta else None
    if raw is None:
        raise McpError(ErrorData(
            code=-32602,
            message=(
                "MISSING_SCENARIO_CONTEXT: scenario_id was not provided in the "
                "request _meta. The frontend must inject the current scenario_id "
                "into the MCP request metadata; it cannot be supplied as a tool "
                "argument."
            ),
        ))
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise McpError(ErrorData(
            code=-32602,
            message=f"INVALID_SCENARIO_CONTEXT: scenario_id in _meta is not an integer: {raw!r}",
        ))


@scenarios_mcp.tool(
    name="classify_scenario",
    title="Classify scenario objects against PZZ zones",
    description="""Run PZZ-compliance classification for the CURRENT scenario's objects.

USE WHEN: the user asks to check whether their scenario's buildings sit in
suitable functional zones, or wants the "is this object in the right zone"
report. Works on the scenario the user currently has open.

CONTEXT (provided automatically, do NOT ask the user for these):
- scenario_id: taken from the request metadata (the open project).
- authorization: the user's token, taken from the request headers.

PARAMETERS (you choose / ask the user):
- year (int, required): data year of the scenario's functional zones.
- source (string, required): one of "User", "OSM", "PZZ" — the zone data source.
- physical_object_type_id (int, default 4): which object type to classify. 4 = residential buildings (Жилой дом). Other ids come from urban_api's physical-object hierarchy.
- priority (int 1-10, default 1).
- force_recompute (bool, default false): re-run even if a cached result for this exact (scenario, year, source, type, data-version) exists.

BEHAVIOUR: asynchronous. Returns immediately with a task descriptor
(external_id + status, usually "queued"). The pipeline runs in the
background. Poll get_scenario_classification_status(external_id) until
status is "finished", then call get_scenario_classification_report.

RETURNS: { external_id, status, priority, ... }.

TIP: to discover valid (year, source) pairs first, call
get_scenario_zone_sources is NOT available — instead, if you pass an
unavailable (year, source) this tool returns an error listing what IS
available.

ERRORS:
- -32002 AUTH_TOKEN_EXPIRED: token rejected — ask frontend for a fresh one.
- -32602 MISSING_SCENARIO_CONTEXT: frontend didn't inject scenario_id.
- -32602 Invalid params: (year, source) not available for this scenario.""",
    tags={"scenario", "submit"},
)
@map_errors
async def classify_scenario(
    year: Annotated[int, "Data year of the scenario's functional zones, e.g. 2026."],
    source: Annotated[str, "Zone data source: 'User', 'OSM' or 'PZZ'."],
    physical_object_type_id: Annotated[
        int, "Object type to classify. 4 = residential buildings."
    ] = 4,
    priority: Annotated[int, "Scheduling priority 1-10."] = 1,
    force_recompute: Annotated[bool, "Re-run even if a cached result exists."] = False,
    ctx: Context = None,
    token: str | None = Depends(extract_token),
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
    scenario_id = _scenario_id_from_meta(ctx)
    return await api.submit_scenario_classify(
        scenario_id=scenario_id,
        year=year,
        source=source,
        physical_object_type_id=physical_object_type_id,
        priority=priority,
        force_recompute=force_recompute,
        token=token,
    )


@scenarios_mcp.tool(
    name="classify_scenario_and_wait",
    title="Classify scenario objects and wait for the report",
    description="""Run scenario classification and BLOCK until it finishes, then return the report.

USE WHEN: the user wants the result in one step and the dataset is small
enough to finish within a couple of minutes. For large datasets prefer
classify_scenario (non-blocking) + polling, since this tool gives up after
a timeout.

Same context + parameters as classify_scenario, plus:
- group_by (string, default "zone"): "zone" groups objects by their zone
  (with PZZ permitted-use summary); "object" returns a flat list.
- timeout_seconds (int, default 180, max 600): how long to wait before
  giving up and returning the latest status instead of the report.

PROGRESS: emits progress updates while polling — a client on the
streamable-HTTP transport sees them live.

RETURNS: the object-zone-fit report
  { task_external_id, summary, chat_message, zones | objects }
when finished; otherwise { external_id, status, timed_out: true }.""",
    tags={"scenario", "submit"},
)
@map_errors
async def classify_scenario_and_wait(
    year: Annotated[int, "Data year, e.g. 2026."],
    source: Annotated[str, "Zone data source: 'User', 'OSM' or 'PZZ'."],
    physical_object_type_id: Annotated[int, "Object type. 4 = residential."] = 4,
    group_by: Annotated[str, "'zone' (default) or 'object'."] = "zone",
    priority: Annotated[int, "Scheduling priority 1-10."] = 1,
    force_recompute: Annotated[bool, "Re-run even if cached."] = False,
    timeout_seconds: Annotated[int, "Max wait before returning status, 1-600."] = 180,
    ctx: Context = None,
    token: str | None = Depends(extract_token),
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
    scenario_id = _scenario_id_from_meta(ctx)
    submitted = await api.submit_scenario_classify(
        scenario_id=scenario_id,
        year=year,
        source=source,
        physical_object_type_id=physical_object_type_id,
        priority=priority,
        force_recompute=force_recompute,
        token=token,
    )
    external_id = submitted.get("external_id")
    if not external_id:
        return submitted

    deadline = max(1, min(int(timeout_seconds), 600))
    waited = 0
    poll_interval = 3
    terminal = {"finished", "failed"}
    status = submitted.get("status", "queued")

    while waited < deadline and status not in terminal:
        await asyncio.sleep(poll_interval)
        waited += poll_interval
        if ctx is not None:
            await ctx.report_progress(
                progress=min(waited, deadline),
                total=deadline,
                message=f"status={status}",
            )
        task = await api.get_scenario_task(
            scenario_id=scenario_id, external_id=external_id, token=token
        )
        status = task.get("status", status)

    if status == "finished":
        return await api.get_scenario_object_zone_fit(
            scenario_id=scenario_id,
            external_id=external_id,
            group_by=group_by,
            token=token,
        )
    if status == "failed":
        task = await api.get_scenario_task(
            scenario_id=scenario_id, external_id=external_id, token=token
        )
        return {
            "external_id": external_id,
            "status": "failed",
            "error_text": task.get("error_text"),
        }
    return {"external_id": external_id, "status": status, "timed_out": True}


@scenarios_mcp.tool(
    name="get_scenario_classification_status",
    title="Get scenario classification status",
    description="""Poll the status of a scenario classification task.

USE WHEN: a classify_scenario call returned an external_id and you need to
know when it's done.

PARAMETERS:
- external_id (string, required): from classify_scenario.

CONTEXT (automatic): scenario_id (_meta), token (header).

RETURNS: TaskOut { external_id, status, started_at, finished_at, error_text }.
status one of queued / waiting_capacity / running / finished / failed.""",
    tags={"scenario", "read"},
    annotations={"readOnlyHint": True},
)
@map_errors
async def get_scenario_classification_status(
    external_id: Annotated[str, "Task id from classify_scenario."],
    ctx: Context = None,
    token: str | None = Depends(extract_token),
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
    scenario_id = _scenario_id_from_meta(ctx)
    return await api.get_scenario_task(
        scenario_id=scenario_id, external_id=external_id, token=token
    )


@scenarios_mcp.tool(
    name="get_scenario_classification_report",
    title="Get scenario classification report",
    description="""Return the structured "which objects are in the wrong zone" report.

USE WHEN: a scenario task has finished and you want to present results to
the user.

PARAMETERS:
- external_id (string, required): finished task id.
- group_by (string, default "zone"): "zone" groups objects by their actual
  zone and attaches the PZZ permitted-use summary; "object" returns a flat
  list of objects.

CONTEXT (automatic): scenario_id (_meta), token (header).

RETURNS:
  {
    task_external_id,
    summary: { total, in_correct_zone, in_wrong_zone, unclear },
    chat_message: plain-text summary ready to show the user,
    zones: [ { zone_name, pzz_summary, summary, objects:[...] } ]  // group_by=zone
      // or objects: [...] when group_by=object
  }
The `chat_message` field is the easiest thing to relay to the user verbatim.

ERRORS: -32602 if the task isn't finished yet (poll status first).""",
    tags={"scenario", "read"},
    annotations={"readOnlyHint": True},
)
@map_errors
async def get_scenario_classification_report(
    external_id: Annotated[str, "Finished task id."],
    group_by: Annotated[str, "'zone' (default) or 'object'."] = "zone",
    ctx: Context = None,
    token: str | None = Depends(extract_token),
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
    scenario_id = _scenario_id_from_meta(ctx)
    return await api.get_scenario_object_zone_fit(
        scenario_id=scenario_id,
        external_id=external_id,
        group_by=group_by,
        token=token,
    )


@scenarios_mcp.tool(
    name="get_scenario_zones_info",
    title="Get scenario zones with permitted-use reference",
    description="""List the current scenario's functional zones with a "what can be built here" reference.

USE WHEN: the user asks what zones exist in their project, or what is
allowed / forbidden in a zone — WITHOUT running classification.

PARAMETERS:
- year (int, required).
- source (string, required): "User" / "OSM" / "PZZ".

CONTEXT (automatic): scenario_id (_meta), token (header).

RETURNS: { scenario_id, year, source, total, items: [ {
  zone_type_id, zone_type_name,
  pzz_summary: { mapping_status, allowed_construction_summary, main_vri, conditional_vri, auxiliary_vri }
} ] }
`pzz_summary.allowed_construction_summary` is human-readable text you can
relay to the user. mapping_status "no_mapping" means no PZZ reference for
that zone type (fields will be null).""",
    tags={"scenario", "read"},
    annotations={"readOnlyHint": True},
)
@map_errors
async def get_scenario_zones_info(
    year: Annotated[int, "Data year, e.g. 2026."],
    source: Annotated[str, "Zone data source: 'User' / 'OSM' / 'PZZ'."],
    ctx: Context = None,
    token: str | None = Depends(extract_token),
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
    scenario_id = _scenario_id_from_meta(ctx)
    return await api.get_scenario_zones_info(
        scenario_id=scenario_id, year=year, source=source, token=token
    )


@scenarios_mcp.tool(
    name="recompute_scenario_classification",
    title="Recompute a scenario classification",
    description="""Force re-run a finished/failed scenario task using its stored inputs.

USE WHEN: the user wants to re-run an existing scenario task without
re-fetching from urban_api. If the scenario DATA changed upstream, prefer
classify_scenario with force_recompute=true instead (it re-pulls).

PARAMETERS:
- external_id (string, required): terminal (finished/failed) task id.

CONTEXT (automatic): scenario_id (_meta), token (header).

RETURNS: TaskOut with status "queued". Poll status afterwards.
ERRORS: -32602 if the task is still active (409 upstream).""",
    tags={"scenario", "control"},
)
@map_errors
async def recompute_scenario_classification(
    external_id: Annotated[str, "Terminal task id to re-run."],
    ctx: Context = None,
    token: str | None = Depends(extract_token),
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
    scenario_id = _scenario_id_from_meta(ctx)
    return await api.recompute_scenario_task(
        scenario_id=scenario_id, external_id=external_id, token=token
    )
