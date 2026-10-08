"""Point check of a ВРИ against the ПЗЗ of a territory, for the compliance agent.

``POST /pzz/regulations/check-vri``: may this use stand in this zone under the ПЗЗ of
the project's (or a given) territory? The ПЗЗ comes from NormGraph, the territory
from Urban API with the caller's token. Separate from the scenario flow, whose checks
and results stay on the default template.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator

from ..application.use_cases.check_vri import (
    Regulations,
    check,
    is_residential,
    resolve_vri,
)
from ..application.use_cases.pzz_zone_types import zone_types
from ..dependencies import build_normgraph_client, get_app_settings
from ..infrastructure.chat_llm_client import build_chat_llm_client
from ..infrastructure.normgraph_client import NormGraphError
from ..infrastructure.pzz_regulations import (
    PzzNotFound,
    resolve_territory_pzz,
    zone_type_mapping,
)
from ..infrastructure.urban_api_client import UrbanApiClient, UrbanApiError
from ..settings import Settings
from .security import verify_token

logger = logging.getLogger("service.pzz_regulations")

router = APIRouter(prefix="/pzz/regulations", tags=["pzz-regulations"])


class VriCheckIn(BaseModel):
    """One of each: the territory, the use and the zone."""

    project_id: int | None = Field(
        default=None, ge=1, description="Urban API project: its territory's ПЗЗ."
    )
    territory_id: int | None = Field(
        default=None, ge=1, description="Urban API territory, instead of a project."
    )
    vri_code: str | None = Field(
        default=None, min_length=1, description="ВРИ code of the classifier, e.g. 2.1."
    )
    physical_object_type_id: int | None = Field(
        default=None, description="Urban API object type, mapped to a ВРИ."
    )
    service_type_id: int | None = Field(
        default=None, description="Urban API service type, mapped to a ВРИ."
    )
    functional_zone_type_id: int | None = Field(
        default=None,
        description="Urban API functional zone type: checked against its ПЗЗ zones.",
    )
    pzz_zone_code: str | None = Field(
        default=None, min_length=1, description="Exact ПЗЗ zone code, e.g. Ж-2.15."
    )
    floors: int | None = Field(default=None, ge=1, description="Above-ground floors.")
    height: float | None = Field(default=None, gt=0, description="Height, metres.")

    @model_validator(mode="after")
    def _one_of_each(self) -> "VriCheckIn":
        groups = {
            "project_id / territory_id": (self.project_id, self.territory_id),
            "vri_code / physical_object_type_id / service_type_id": (
                self.vri_code,
                self.physical_object_type_id,
                self.service_type_id,
            ),
            "functional_zone_type_id / pzz_zone_code": (
                self.functional_zone_type_id,
                self.pzz_zone_code,
            ),
        }
        for names, values in groups.items():
            if sum(value is not None for value in values) != 1:
                raise ValueError(f"exactly one of {names} is required")
        return self


@lru_cache(maxsize=8)
def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _type_entry(types: list[dict[str, Any]], type_id: int) -> dict[str, Any]:
    entry = next((t for t in types if t["id"] == type_id), None)
    if entry is None:
        raise HTTPException(
            status_code=422,
            detail=f"unknown functional_zone_type_id={type_id}; known: "
            + ", ".join(str(t["id"]) for t in types),
        )
    return entry


@router.post("/check-vri")
async def check_vri_endpoint(
    body: VriCheckIn,
    token: str = Depends(verify_token),
    app_settings: Settings = Depends(get_app_settings),
) -> dict[str, Any]:
    """Check a use against the ПЗЗ of a territory; see ``check_vri`` for the verdicts."""
    if not app_settings.urban_api_base_url:
        raise HTTPException(status_code=503, detail="URBAN_API_BASE_URL is not set")
    normgraph = build_normgraph_client(app_settings)
    if normgraph is None:
        raise HTTPException(
            status_code=503,
            detail="NormGraph is not configured (NORMGRAPH_BASE_URL and the "
            "Keycloak service account)",
        )
    try:
        async with (
            UrbanApiClient(
                base_url=app_settings.urban_api_base_url,
                timeout_seconds=app_settings.urban_api_timeout_seconds,
            ) as urban,
            normgraph,
        ):
            payload = await resolve_territory_pzz(
                urban=urban,
                normgraph=normgraph,
                project_id=body.project_id,
                territory_id=body.territory_id,
                token=token,
            )
    except PzzNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UrbanApiError as exc:
        logger.warning("ВРИ check: Urban API failed: %s", exc)
        status = exc.status if exc.status in (401, 403, 404) else 502
        raise HTTPException(
            status_code=status, detail="Urban API rejected or failed the request"
        ) from exc
    except NormGraphError as exc:
        logger.warning("ВРИ check: NormGraph failed: %s", exc)
        raise HTTPException(status_code=502, detail="NormGraph is unavailable") from exc
    except httpx.HTTPError as exc:
        logger.warning("ВРИ check: Urban API / NormGraph unreachable: %s", exc)
        raise HTTPException(
            status_code=502, detail="Urban API or NormGraph is unreachable"
        ) from exc

    regs = Regulations(payload)
    vri = resolve_vri(
        vri_code=body.vri_code,
        physical_object_type_id=body.physical_object_type_id,
        service_type_id=body.service_type_id,
        floors=body.floors,
        po2vri=_load_json(app_settings.physical_object_type_to_vri_path),
        service_map=_load_json(app_settings.service_type_to_vri_path).get(
            "by_service_type_id", {}
        ),
    )
    zone_info: dict[str, Any]
    if body.pzz_zone_code is not None:
        zone = regs.zone(body.pzz_zone_code)
        if zone is None:
            raise HTTPException(
                status_code=404,
                detail=f"no zone {body.pzz_zone_code} in {regs.label}; zones: "
                + ", ".join(z["code"] for z in regs.zones),
            )
        zones = [zone]
        where = f"ПЗЗ {zone['code']}"
        zone_info = {"pzz_zone_code": zone["code"]}
    else:
        types = zone_types(_load_json(app_settings.default_fz_to_pzz_mapping_path))
        entry = _type_entry(types, body.functional_zone_type_id)
        mapping = await zone_type_mapping(
            payload,
            types=types,
            document_label=regs.label,
            llm_factory=lambda: build_chat_llm_client(app_settings),
        )
        codes = mapping.codes(entry["id"])
        zones = [regs.zone(code) for code in codes]
        where = f"ПЗЗ типа «{entry['nickname']}»"
        zone_info = {
            "functional_zone_type": {
                "id": entry["id"],
                "name": entry["name"],
                "nickname": entry["nickname"],
            },
            "mapping": {"method": mapping.method, "zone_codes": codes},
        }

    result = check(
        regs,
        vri,
        [z for z in zones if z is not None],
        where=where,
        floors=body.floors,
        height=body.height,
        residential=is_residential(vri["code"], body.physical_object_type_id),
    )
    if vri["code"] and not vri["name"]:
        vri["name"] = regs.use_name(vri["code"])
    return {
        "document": {**payload["document"], "label": regs.label},
        "territory": payload["territory"],
        "requested_territory": payload["requested_territory"],
        "vri": vri,
        "zone": zone_info,
        **result,
    }
