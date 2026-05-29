"""MCP tools for the scenario classification flow (urban_api-backed).

Parameter model:

- **Visible parameters** (the LLM supplies / asks the user / reads from
  urban_api): ``scenario_id``, ``year``, ``source``,
  ``physical_object_type_id``, ``priority``, ``force_recompute``,
  ``external_id``, ``group_by``. ``scenario_id`` identifies which project /
  scenario to operate on and is passed explicitly by the caller.

- **Hidden auth** (NOT in the tool schema): the user's Bearer token is read
  from the HTTP ``Authorization`` header and forwarded to our REST layer,
  which forwards it to urban_api.

Long runs report progress via ``ctx.report_progress`` so a client on the
streamable-HTTP transport sees live updates.
"""
from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from fastmcp.dependencies import Depends
from fastmcp.server.dependencies import get_http_headers

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


@scenarios_mcp.tool(
    name="classify_scenario",
    title="Classify scenario objects against PZZ zones",
    description="""Run PZZ-compliance classification for a scenario's objects.

USE WHEN: the user asks to check whether their scenario's buildings sit in
suitable functional zones, or wants the "is this object in the right zone"
report.

AUTH (automatic, do NOT ask the user): the user's token is taken from the
request Authorization header.

PARAMETERS (you supply / ask the user):
- scenario_id (int, required): the project/scenario to classify.
- year (int, required): data year of the scenario's functional zones.
- source (string, required): one of "User", "OSM", "PZZ" — the zone data source.
- physical_object_type_id (int, default 4): which object type to classify. 4 = residential buildings (Жилой дом). Other ids come from urban_api's physical-object hierarchy.
- priority (int 1-10, default 1).
- force_recompute (bool, default false): re-run even if a cached result for this exact (scenario, year, source, type, data-version) exists.

BEHAVIOUR: asynchronous. Returns immediately with a task descriptor
(external_id + status, usually "queued"). The pipeline runs in the
background. Poll get_scenario_classification_status(scenario_id, external_id)
until status is "finished", then call get_scenario_classification_report.

RETURNS: { external_id, status, priority, ... }.

TIP: if you pass an unavailable (year, source) this tool returns an error
listing what IS available for the scenario.

ERRORS:
- -32002 AUTH_TOKEN_EXPIRED: token rejected — ask frontend for a fresh one.
- -32602 Invalid params: (year, source) not available for this scenario.""",
    tags={"scenario", "submit"},
)
@map_errors
async def classify_scenario(
    scenario_id: Annotated[int, "The project/scenario id to classify."],
    year: Annotated[int, "Data year of the scenario's functional zones, e.g. 2026."],
    source: Annotated[str, "Zone data source: 'User', 'OSM' or 'PZZ'."],
    physical_object_type_id: Annotated[
        int, "Object type to classify. 4 = residential buildings."
    ] = 4,
    priority: Annotated[int, "Scheduling priority 1-10."] = 1,
    force_recompute: Annotated[bool, "Re-run even if a cached result exists."] = False,
    token: str | None = Depends(extract_token),
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
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

AUTH (automatic): the user's token from the Authorization header.

Same parameters as classify_scenario (scenario_id, year, source,
physical_object_type_id, priority, force_recompute), plus:
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
    scenario_id: Annotated[int, "The project/scenario id to classify."],
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
- scenario_id (int, required): the scenario the task belongs to.
- external_id (string, required): from classify_scenario.

AUTH (automatic): the user's token from the Authorization header.

RETURNS: TaskOut { external_id, status, started_at, finished_at, error_text }.
status one of queued / waiting_capacity / running / finished / failed.""",
    tags={"scenario", "read"},
    annotations={"readOnlyHint": True},
)
@map_errors
async def get_scenario_classification_status(
    scenario_id: Annotated[int, "The scenario id the task belongs to."],
    external_id: Annotated[str, "Task id from classify_scenario."],
    token: str | None = Depends(extract_token),
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
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
- scenario_id (int, required): the scenario the task belongs to.
- external_id (string, required): finished task id.
- group_by (string, default "zone"): "zone" groups objects by their actual
  zone and attaches the PZZ permitted-use summary; "object" returns a flat
  list of objects.

AUTH (automatic): the user's token from the Authorization header.

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
    scenario_id: Annotated[int, "The scenario id the task belongs to."],
    external_id: Annotated[str, "Finished task id."],
    group_by: Annotated[str, "'zone' (default) or 'object'."] = "zone",
    token: str | None = Depends(extract_token),
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
    return await api.get_scenario_object_zone_fit(
        scenario_id=scenario_id,
        external_id=external_id,
        group_by=group_by,
        token=token,
    )


@scenarios_mcp.tool(
    name="get_scenario_zones_info",
    title="Get scenario zones with permitted-use reference",
    description="""List a scenario's functional zones with a "what can be built here" reference.

USE WHEN: the user asks what zones exist in their project, or what is
allowed / forbidden in a zone — WITHOUT running classification.

PARAMETERS:
- scenario_id (int, required): the project/scenario to inspect.
- year (int, required).
- source (string, required): "User" / "OSM" / "PZZ".

AUTH (automatic): the user's token from the Authorization header.

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
    scenario_id: Annotated[int, "The project/scenario id to inspect."],
    year: Annotated[int, "Data year, e.g. 2026."],
    source: Annotated[str, "Zone data source: 'User' / 'OSM' / 'PZZ'."],
    token: str | None = Depends(extract_token),
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
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
- scenario_id (int, required): the scenario the task belongs to.
- external_id (string, required): terminal (finished/failed) task id.

AUTH (automatic): the user's token from the Authorization header.

RETURNS: TaskOut with status "queued". Poll status afterwards.
ERRORS: -32602 if the task is still active (409 upstream).""",
    tags={"scenario", "control"},
)
@map_errors
async def recompute_scenario_classification(
    scenario_id: Annotated[int, "The scenario id the task belongs to."],
    external_id: Annotated[str, "Terminal task id to re-run."],
    token: str | None = Depends(extract_token),
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
    return await api.recompute_scenario_task(
        scenario_id=scenario_id, external_id=external_id, token=token
    )
