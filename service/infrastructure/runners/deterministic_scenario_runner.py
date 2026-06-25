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
unchanged. Heavy geo deps are imported lazily so the API process (which never
runs this) does not pay for them.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from service.domain import PipelineRequest
from service.infrastructure.runners.pipeline_runner import PipelineRunner, _build_output_glob
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from service.settings import Settings

logger = logging.getLogger("service.tasks")

# Output property columns the object-zone-fit endpoint reads.
_COL_VRI_TEXT = "ВРИ_ЕГРН"
_COL_ZONE_CODE = "Код фактической зоны нахождения кадастра"
_COL_ZONE_NAME = "Название фактической зоны нахождения кадастра"
_COL_VERDICT = "Вердикт_ПЗЗ"
_COL_REASON = "Причина"
_COL_MATCHED_VRI_NAME = "Подобранный_ВРИ"
_COL_MATCHED_VRI_CODE = "Код_подобранного_ВРИ"

_FLOORS_FIELD = "Количество этажей"


def _is_allowed(vri: str, allowed: set[str]) -> bool:
    """Exact or hierarchical membership (an umbrella code allows its children)."""
    for a in allowed:
        if a == vri or vri.startswith(a + ".") or a.startswith(vri + "."):
            return True
    return False


class DeterministicScenarioRunner(PipelineRunner):
    """Classify a scenario's objects against PZZ zones via dictionary lookups."""

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._po2vri = json.loads(
            Path(settings.physical_object_type_to_vri_path).read_text(encoding="utf-8")
        )
        self._zone_allowed, self._zone_nick = self._load_zone_mapping(
            settings.default_fz_to_pzz_mapping_path
        )

    @staticmethod
    def _load_zone_mapping(path: str) -> tuple[dict[int, dict[str, set[str]]], dict[int, str]]:
        """Return {fz_type_id: {section: {vri_code}}} and {fz_type_id: nickname}."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed: dict[int, dict[str, set[str]]] = {}
        nick: dict[int, str] = {}
        for e in raw.get("functional_zone_mappings", []):
            fz = e.get("functional_zone_type_id")
            if fz is None:
                continue
            prof = e.get("averaged_pzz_profile", {})
            allowed[int(fz)] = {
                section: {v["vri_code"] for v in (prof.get(key) or []) if v.get("vri_code")}
                for section, key in (
                    ("main", "main_vri"),
                    ("conditional", "conditional_vri"),
                    ("auxiliary", "auxiliary_vri"),
                )
            }
            nick[int(fz)] = e.get("db_zone_nickname") or str(fz)
        return allowed, nick

    def _object_vri(self, po_type_id: int, floors: Any) -> tuple[str | None, str | None]:
        """Return (vri_code, vri_name) for a physical object type, floor-aware."""
        rule = self._po2vri["by_type_id"].get(str(po_type_id))
        if rule is None:
            return None, None
        if rule.get("strategy") == "residential_floor_bands":
            bands = self._po2vri["residential_floor_bands"]
            try:
                f = int(floors) if floors is not None else None
            except (TypeError, ValueError):
                f = None
            if f is None:
                return bands["fallback_vri_code"], None
            for band in bands["bands"]:
                mx = band["max_floors"]
                if mx is None or f <= mx:
                    return band["vri_code"], band.get("vri_name")
            return bands["fallback_vri_code"], None
        return rule.get("vri_code"), rule.get("vri_name")

    def _verdict(self, vri: str | None, fz_type_id: int | None) -> tuple[str, str, str, str]:
        """Return (verdict, reason, matched_vri_code, matched_vri_name)."""
        if fz_type_id is None:
            return "no_actual_zone", "Объект не пересекается ни с одной функциональной зоной.", "", ""
        zone_name = self._zone_nick.get(fz_type_id, str(fz_type_id))
        if vri is None:
            return "unclear", "Для типа объекта нет сопоставленного ВРИ в словаре.", "", ""
        sections = self._zone_allowed.get(fz_type_id)
        if not sections or not any(sections.values()):
            return "no_zone_metadata", f"Для зоны «{zone_name}» нет описания разрешённых ВРИ.", vri, ""
        for section in ("main", "conditional", "auxiliary"):
            if _is_allowed(vri, sections.get(section) or set()):
                return (
                    f"allowed_{section}",
                    f"ВРИ {vri} разрешён в зоне «{zone_name}» ({section}).",
                    vri, "",
                )
        return "not_allowed", f"ВРИ {vri} не входит в разрешённые в зоне «{zone_name}».", vri, ""

    def run(self, request: PipelineRequest) -> str:
        import geopandas as gpd
        from shapely.geometry import shape

        output_dir = Path(request.outputs_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        objects = json.loads(Path(request.cadastral_data_path).read_text(encoding="utf-8"))
        zones = json.loads(Path(request.pzz_zones_data_path).read_text(encoding="utf-8"))
        code_col = request.pzz_zone_code_col or "zone_code"
        vri_col = request.cadastral_vri_col or "vri_text"

        # --- zones GeoDataFrame (geometry + functional_zone_type_id) ---
        zg_geom, zg_fz = [], []
        for f in zones.get("features") or []:
            props = f.get("properties") or {}
            raw = props.get(code_col)
            if not raw and isinstance(props.get("functional_zone_type"), dict):
                raw = props["functional_zone_type"].get("id")
            if f.get("geometry") is None or raw in (None, ""):
                continue
            try:
                zg_fz.append(int(raw))
            except (TypeError, ValueError):
                continue
            zg_geom.append(shape(f["geometry"]))
        zgdf = gpd.GeoDataFrame({"fz_type_id": zg_fz}, geometry=zg_geom, crs="EPSG:4326")

        # --- objects: keep original features so we can annotate + re-emit ---
        feats = [f for f in (objects.get("features") or []) if f.get("geometry") is not None]
        o_geom = [shape(f["geometry"]) for f in feats]
        ogdf = gpd.GeoDataFrame({"_i": list(range(len(feats)))}, geometry=o_geom, crs="EPSG:4326")

        # --- spatial join on representative point (one deterministic zone) ---
        fz_by_obj: dict[int, int] = {}
        if len(ogdf) and len(zgdf):
            metric = zgdf.estimate_utm_crs()
            pts = ogdf.to_crs(metric).copy()
            pts["geometry"] = pts.representative_point()
            joined = gpd.sjoin(
                pts, zgdf.to_crs(metric)[["fz_type_id", "geometry"]],
                how="left", predicate="within",
            )
            joined = joined[~joined.index.duplicated(keep="first")]
            for idx, row in joined.iterrows():
                fz = row.get("fz_type_id")
                if fz is not None and fz == fz:  # not NaN
                    fz_by_obj[int(ogdf.loc[idx, "_i"])] = int(fz)

        # --- classify + annotate features ---
        for i, feature in enumerate(feats):
            props = feature.setdefault("properties", {})
            po_type = (props.get("physical_object_type") or {}).get("physical_object_type_id")
            nested = props.get("properties") if isinstance(props.get("properties"), dict) else {}
            floors = nested.get(_FLOORS_FIELD, props.get(_FLOORS_FIELD))
            vri, vri_name = (None, None)
            if po_type is not None:
                vri, vri_name = self._object_vri(int(po_type), floors)

            fz = fz_by_obj.get(i)
            verdict, reason, mcode, _ = self._verdict(vri, fz)

            props[_COL_VRI_TEXT] = props.get(vri_col)
            props[_COL_ZONE_CODE] = str(fz) if fz is not None else ""
            props[_COL_ZONE_NAME] = self._zone_nick.get(fz, "") if fz is not None else ""
            props[_COL_VERDICT] = verdict
            props[_COL_REASON] = reason
            props[_COL_MATCHED_VRI_CODE] = mcode
            props[_COL_MATCHED_VRI_NAME] = vri_name or ""

        result = {"type": "FeatureCollection", "features": feats}
        out_path = output_dir / f"pzz_compare_spatial_first_{request.task_external_id}.geojson"
        out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        logger.info(
            json.dumps({
                "stage": "deterministic_scenario", "status": "finished",
                "external_id": request.task_external_id,
                "objects": len(feats), "zones": len(zgdf), "matched_zone": len(fz_by_obj),
            })
        )
        return _build_output_glob(output_dir, request.task_external_id)
