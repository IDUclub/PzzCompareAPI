"""MCP tool for the point check of a ВРИ against the ПЗЗ of a territory.

Wraps ``POST /pzz/regulations/check-vri``: the compliance agent asks whether a use
may stand in a zone under the ПЗЗ that NormGraph read for the project's territory.
The user's Bearer token (for Urban API) is read from the HTTP ``Authorization``
header, as in the scenario tools.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.dependencies import Depends

from ..api_client import ApiClient
from ..dependencies import get_api_client
from ..exceptions import map_errors
from .scenarios import extract_token

regulations_mcp = FastMCP("PZZ Pipeline Regulations")


@regulations_mcp.tool(
    name="check_vri_in_pzz",
    title="Check a ВРИ against the ПЗЗ of a territory",
    description="""Check whether a use (ВРИ) may stand in a zone under the land-use rules (ПЗЗ) of a territory.

USE WHEN: compliance checking needs the ПЗЗ verdict for one use in one zone — e.g.
"may a 5-floor residential building stand in the residential zone of project 938?".
The ПЗЗ is the real document of the territory (NormGraph), not a template.

PARAMETERS (one of each group):
- territory: project_id (the project's territory) OR territory_id (Urban API).
- use: vri_code (e.g. "2.1") OR physical_object_type_id (Urban API object type;
  4 = residential building, its ВРИ follows the floors) OR service_type_id.
- zone: functional_zone_type_id (Urban API functional zone type: every ПЗЗ zone of
  that type is checked) OR pzz_zone_code (an exact ПЗЗ zone, e.g. "Ж-2.15").
- floors, height (optional): checked against the zones' limits.

AUTH (automatic): the user's token from the Authorization header.

RETURNS:
  { document: {name, version, label, territory_name, ...},
    vri: {code, name, basis},
    zone: {pzz_zone_code} | {functional_zone_type, mapping: {method, zone_codes}},
    verdict, verdict_label, reason,
    zones: [ {code, name, verdict, verdict_label, reason,
              parameters: {status, reason}} ] }
verdict: allowed_main / allowed_conditional / allowed_auxiliary / not_allowed;
"depends_on_zone" when a functional zone type covers ПЗЗ zones that disagree (the
exact ПЗЗ zone decides — see `zones`); "unclear" when there is no ВРИ for the object
or no ПЗЗ zone of the type. parameters.status: "Соответствует" / "Превышены" /
"Не проверено". mapping.method "llm" — the ПЗЗ zones of the type were chosen by the
language model, "code_prefix" — by code prefix (model unavailable).
The first check against a document can take up to a minute (zone mapping).

ERRORS:
- -32002 AUTH_TOKEN_EXPIRED: token rejected — ask frontend for a fresh one.
- -32602: no ПЗЗ for the territory (404), unknown zone code or type, bad params.
- -32603: NormGraph / Urban API unavailable, or the check is not configured.""",
    tags={"regulations", "read"},
    annotations={"readOnlyHint": True},
)
@map_errors
async def check_vri_in_pzz(
    project_id: Annotated[int | None, "Urban API project id."] = None,
    territory_id: Annotated[
        int | None, "Urban API territory id (instead of project_id)."
    ] = None,
    vri_code: Annotated[str | None, "ВРИ code, e.g. '2.1'."] = None,
    physical_object_type_id: Annotated[
        int | None, "Urban API object type; 4 = residential building."
    ] = None,
    service_type_id: Annotated[int | None, "Urban API service type."] = None,
    functional_zone_type_id: Annotated[
        int | None, "Urban API functional zone type."
    ] = None,
    pzz_zone_code: Annotated[str | None, "Exact ПЗЗ zone code, e.g. 'Ж-2.15'."] = None,
    floors: Annotated[int | None, "Above-ground floors of the object."] = None,
    height: Annotated[float | None, "Height of the object, metres."] = None,
    token: str | None = Depends(extract_token),
    api: ApiClient = Depends(get_api_client),
) -> dict[str, Any]:
    body = {
        "project_id": project_id,
        "territory_id": territory_id,
        "vri_code": vri_code,
        "physical_object_type_id": physical_object_type_id,
        "service_type_id": service_type_id,
        "functional_zone_type_id": functional_zone_type_id,
        "pzz_zone_code": pzz_zone_code,
        "floors": floors,
        "height": height,
    }
    return await api.check_vri(
        body={k: v for k, v in body.items() if v is not None}, token=token
    )
