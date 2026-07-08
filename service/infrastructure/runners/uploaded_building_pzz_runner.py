"""Deterministic PZZ check for user-uploaded buildings — no LLM, no embeddings.

The building counterpart of the cadastral upload flow: instead of a parcel's VRI
text, the user uploads *buildings* (Urban-API-shaped: ``physical_object_type_id``
/ ``service_type_id`` + floors) plus their own PZZ zones, and optionally a zone
descriptions file. Each building is resolved to a VRI code and tested against its
containing zone's permitted-VRI set — the same verdict logic as the scenario
runner (see ``_deterministic_pzz``), driven by user-named columns instead of
urban_api's fixed schema.

Resolution priority for a building's VRI:
    1. residential  -> floor-band VRI (uses the floors column);
    2. service       -> service_type_id → VRI (service_type_to_vri.json);
    3. otherwise     -> physical_object_type_id → VRI.

Zone permitted-VRI set comes from the uploaded descriptions file when supplied,
else the built-in fz_to_pzz mapping (fallback). Heavy geo deps load lazily.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from service.domain import PipelineRequest
from service.infrastructure.runners._deterministic_pzz import (
    build_zone_gdf,
    clean_result_properties,
    join_objects_to_zones,
    load_zone_mapping,
    resolve_po_type_vri,
    verdict as compute_verdict,
)
from service.infrastructure.runners.pipeline_runner import PipelineRunner, _build_output_glob
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from service.settings import Settings

logger = logging.getLogger("service.tasks")

_FLOORS_FIELD = "Количество этажей"
_RESIDENTIAL_PO_TYPE = 4  # urban_api "жилой дом"
_RESIDENTIAL_TEXT = ("жил", "residential", "жилое", "жилой")


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class UploadedBuildingPzzRunner(PipelineRunner):
    """Classify user-uploaded buildings against user-uploaded PZZ zones."""

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._po2vri = json.loads(
            Path(settings.physical_object_type_to_vri_path).read_text(encoding="utf-8")
        )
        raw_service = json.loads(
            Path(settings.service_type_to_vri_path).read_text(encoding="utf-8")
        )
        self._service_map: dict[str, Any] = raw_service.get("by_service_type_id", {})

    def _load_zone_mapping(self, request: PipelineRequest):
        """User descriptions file when supplied+usable, else built-in fallback."""
        descriptions_path = request.pzz_zone_vri_labels_path
        if descriptions_path and Path(descriptions_path).is_file():
            try:
                allowed, nick = load_zone_mapping(descriptions_path)
                if allowed:
                    return allowed, nick
                logger.warning(
                    "uploaded zone descriptions had no usable mappings; "
                    "falling back to built-in fz_to_pzz mapping"
                )
            except (json.JSONDecodeError, OSError, KeyError) as exc:
                logger.warning("failed to read uploaded zone descriptions (%s); fallback", exc)
        return load_zone_mapping(self._settings.default_fz_to_pzz_mapping_path)

    def _extract(self, props: dict[str, Any], request: PipelineRequest):
        """Return (po_type_id, is_residential, service_type_id, floors, label)."""
        nested = props.get("properties") if isinstance(props.get("properties"), dict) else {}

        type_raw = props.get(request.building_type_col) if request.building_type_col else None
        # Unwrap the urban_api nested ``physical_object_type`` object (the detected
        # column may point straight at it) to its id.
        if isinstance(type_raw, dict):
            type_raw = type_raw.get("physical_object_type_id")
        if type_raw is None:
            type_raw = (props.get("physical_object_type") or {}).get("physical_object_type_id")

        service_raw = props.get(request.building_service_col) if request.building_service_col else None
        if isinstance(service_raw, dict):
            service_raw = service_raw.get("service_type_id")

        floors = props.get(request.building_floors_col) if request.building_floors_col else None
        if floors is None:
            floors = nested.get(_FLOORS_FIELD, props.get(_FLOORS_FIELD))

        po_type_id = _as_int(type_raw)
        is_residential = po_type_id == _RESIDENTIAL_PO_TYPE
        if po_type_id is None and isinstance(type_raw, str):
            low = type_raw.strip().lower()
            if any(low.startswith(t) for t in _RESIDENTIAL_TEXT):
                is_residential = True
        service_type_id = _as_int(service_raw)

        label = " / ".join(
            str(x) for x in (type_raw, service_raw) if x not in (None, "")
        ) or None
        return po_type_id, is_residential, service_type_id, floors, label

    def _resolve_vri(
        self, po_type_id: int | None, is_residential: bool, service_type_id: int | None, floors: Any
    ) -> tuple[str | None, str | None]:
        if is_residential:
            code, name = resolve_po_type_vri(self._po2vri, _RESIDENTIAL_PO_TYPE, floors)
            if code:
                return code, name
        if service_type_id is not None:
            entry = self._service_map.get(str(service_type_id))
            if entry and entry.get("vri_code"):
                return entry["vri_code"], entry.get("vri_name") or None
        if po_type_id is not None:
            code, name = resolve_po_type_vri(self._po2vri, po_type_id, floors)
            if code:
                return code, name
        return None, None

    def run(self, request: PipelineRequest) -> str:
        output_dir = Path(request.outputs_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        buildings = json.loads(Path(request.cadastral_data_path).read_text(encoding="utf-8"))
        zones = json.loads(Path(request.pzz_zones_data_path).read_text(encoding="utf-8"))
        code_col = request.pzz_zone_code_col or "zone_code"
        zone_allowed, zone_nick = self._load_zone_mapping(request)

        zgdf = build_zone_gdf(zones, code_col)
        feats = [f for f in (buildings.get("features") or []) if f.get("geometry") is not None]
        fz_by_obj = join_objects_to_zones(feats, zgdf)

        for i, feature in enumerate(feats):
            props = feature.get("properties") or {}
            po_type_id, is_residential, service_type_id, floors, label = self._extract(props, request)
            vri, vri_name = self._resolve_vri(po_type_id, is_residential, service_type_id, floors)

            fz = fz_by_obj.get(i)
            machine_verdict, reason, mcode, _ = compute_verdict(vri, fz, zone_allowed, zone_nick)
            feature["properties"] = clean_result_properties(
                vri_text=label,
                fz_type_id=fz,
                zone_nick=zone_nick,
                machine_verdict=machine_verdict,
                reason=reason,
                matched_vri_code=mcode,
                matched_vri_name=vri_name,
            )

        result = {"type": "FeatureCollection", "features": feats}
        out_path = output_dir / f"pzz_compare_spatial_first_{request.task_external_id}.geojson"
        out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        logger.info(
            json.dumps({
                "stage": "uploaded_building_pzz", "status": "finished",
                "external_id": request.task_external_id,
                "buildings": len(feats), "zones": len(zgdf), "matched_zone": len(fz_by_obj),
            })
        )
        return _build_output_glob(output_dir, request.task_external_id)
