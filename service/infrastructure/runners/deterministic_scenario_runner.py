"""Deterministic scenario PZZ classifier — no LLM, no embeddings.

For scenario tasks the inputs come from urban_api's controlled vocabularies:
each physical object has a stable ``physical_object_type_id`` and each
functional zone a stable ``functional_zone_type_id``. That lets us decide PZZ
fit with pure dictionary lookups instead of the string/embed/LLM cascade:

    object.physical_object_type_id (+floors)  --dict-->  VRI code
    object's zone.functional_zone_type_id     --PZZ map-> allowed VRI set
    verdict = VRI in allowed set (main / conditional / auxiliary) else not_allowed

It produces a result GeoJSON with the same verdict columns the LLM pipeline
emits, so ``/tasks/{id}/object-zone-fit`` and the scenario report read it
unchanged. The building→VRI→verdict primitives are shared with the
uploaded-building runner via ``_deterministic_pzz``. Heavy geo deps are imported
lazily so the API process (which never runs this) does not pay for them.
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
from service.infrastructure.runners.pipeline_runner import (
    PipelineRunner,
    _build_output_glob,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from service.settings import Settings

logger = logging.getLogger("service.tasks")

_FLOORS_FIELD = "Количество этажей"


class DeterministicScenarioRunner(PipelineRunner):
    """Classify a scenario's objects against PZZ zones via dictionary lookups."""

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._po2vri = json.loads(
            Path(settings.physical_object_type_to_vri_path).read_text(encoding="utf-8")
        )
        self._zone_allowed, self._zone_nick = load_zone_mapping(
            settings.default_fz_to_pzz_mapping_path
        )

    def run(self, request: PipelineRequest) -> str:
        output_dir = Path(request.outputs_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        objects = json.loads(
            Path(request.cadastral_data_path).read_text(encoding="utf-8")
        )
        zones = json.loads(
            Path(request.pzz_zones_data_path).read_text(encoding="utf-8")
        )
        code_col = request.pzz_zone_code_col or "zone_code"
        vri_col = request.cadastral_vri_col or "vri_text"

        zgdf = build_zone_gdf(zones, code_col)
        feats = [
            f for f in (objects.get("features") or []) if f.get("geometry") is not None
        ]
        fz_by_obj = join_objects_to_zones(feats, zgdf)

        # --- classify + annotate features ---
        for i, feature in enumerate(feats):
            old_props = feature.get("properties") or {}
            po_type = (old_props.get("physical_object_type") or {}).get(
                "physical_object_type_id"
            )
            nested = (
                old_props.get("properties")
                if isinstance(old_props.get("properties"), dict)
                else {}
            )
            floors = nested.get(_FLOORS_FIELD, old_props.get(_FLOORS_FIELD))
            vri, vri_name = (None, None)
            if po_type is not None:
                vri, vri_name = resolve_po_type_vri(self._po2vri, int(po_type), floors)

            fz = fz_by_obj.get(i)
            machine_verdict, reason, mcode, _ = compute_verdict(
                vri, fz, self._zone_allowed, self._zone_nick
            )
            feature["properties"] = clean_result_properties(
                vri_text=old_props.get(vri_col),
                fz_type_id=fz,
                zone_nick=self._zone_nick,
                machine_verdict=machine_verdict,
                reason=reason,
                matched_vri_code=mcode,
                matched_vri_name=vri_name,
            )

        result = {"type": "FeatureCollection", "features": feats}
        out_path = (
            output_dir / f"pzz_compare_spatial_first_{request.task_external_id}.geojson"
        )
        out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        logger.info(
            json.dumps(
                {
                    "stage": "deterministic_scenario",
                    "status": "finished",
                    "external_id": request.task_external_id,
                    "objects": len(feats),
                    "zones": len(zgdf),
                    "matched_zone": len(fz_by_obj),
                }
            )
        )
        return _build_output_glob(output_dir, request.task_external_id)
