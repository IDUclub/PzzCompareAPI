"""Shared building→VRI→PZZ-verdict primitives for the deterministic runners.

Both the urban_api scenario runner and the uploaded-building runner decide PZZ
fit the same way — a building's VRI code (floor-aware for residential) is tested
against a functional zone's permitted-VRI set via dictionary lookups, no LLM.
The pieces that differ (where the ids/attributes come from, the mapping source)
live in the runners; the pieces that don't live here.

Heavy geo deps are imported lazily by the callers, so this module stays cheap.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# --- result property columns the object-zone-fit endpoint / reports read ------
COL_VRI_TEXT = "ВРИ_ЕГРН"
COL_ZONE_CODE = "Код фактической зоны нахождения кадастра"
COL_ZONE_NAME = "Название фактической зоны нахождения кадастра"
COL_VERDICT = "Вердикт_ПЗЗ"
COL_REASON = "Причина"
COL_MATCHED_VRI_NAME = "Подобранный_ВРИ"
COL_MATCHED_VRI_CODE = "Код_подобранного_ВРИ"

# Machine verdict -> human-readable Russian label (mirrors the pipeline's
# ``status_to_russian_label``; duplicated here to keep the API/worker side free
# of a ``pipeline_modules`` import).
VERDICT_RU = {
    "allowed_main": "Разрешен",
    "allowed_conditional": "Условно разрешен",
    "allowed_auxiliary": "Разрешен как вспомогательный",
    "not_allowed": "Не разрешен",
    "unclear": "Требуется ручная проверка",
    "no_actual_zone": "Нет пересечения с ПЗЗ",
    "no_zone_metadata": "Нет описания зоны в шаблоне",
}


def is_allowed(vri: str, allowed: set[str]) -> bool:
    """Exact or hierarchical membership (an umbrella code allows its children)."""
    for a in allowed:
        if a == vri or vri.startswith(a + ".") or a.startswith(vri + "."):
            return True
    return False


def load_zone_mapping(
    path: str,
) -> tuple[dict[int, dict[str, set[str]]], dict[int, str]]:
    """Load a functional_zone_type_id → permitted-VRI mapping file.

    Returns ``({fz_type_id: {section: {vri_code}}}, {fz_type_id: nickname})``.
    Accepts the built-in ``functional_zones_to_pzz_mapping.json`` and any
    user-supplied descriptions file in the same schema (``functional_zone_mappings``
    with an ``averaged_pzz_profile`` per entry).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed: dict[int, dict[str, set[str]]] = {}
    nick: dict[int, str] = {}
    for e in raw.get("functional_zone_mappings", []):
        fz = e.get("functional_zone_type_id")
        if fz is None:
            continue
        try:
            fz_id = int(fz)
        except (TypeError, ValueError):
            continue
        prof = e.get("averaged_pzz_profile", {})
        allowed[fz_id] = {
            section: {v["vri_code"] for v in (prof.get(key) or []) if v.get("vri_code")}
            for section, key in (
                ("main", "main_vri"),
                ("conditional", "conditional_vri"),
                ("auxiliary", "auxiliary_vri"),
            )
        }
        nick[fz_id] = e.get("db_zone_nickname") or str(fz_id)
    return allowed, nick


def resolve_po_type_vri(
    po2vri: dict[str, Any], po_type_id: int, floors: Any
) -> tuple[str | None, str | None]:
    """Return ``(vri_code, vri_name)`` for a physical_object_type_id, floor-aware.

    Residential types (``strategy == residential_floor_bands``) pick the VRI band
    by floor count, falling back to the configured code when floors are missing.
    """
    rule = po2vri["by_type_id"].get(str(po_type_id))
    if rule is None:
        return None, None
    if rule.get("strategy") == "residential_floor_bands":
        bands = po2vri["residential_floor_bands"]
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


def verdict(
    vri: str | None,
    fz_type_id: int | None,
    zone_allowed: dict[int, dict[str, set[str]]],
    zone_nick: dict[int, str],
) -> tuple[str, str, str, str]:
    """Return ``(machine_verdict, reason, matched_vri_code, matched_vri_name)``."""
    if fz_type_id is None:
        return "no_actual_zone", "Объект не пересекается ни с одной функциональной зоной.", "", ""
    zone_name = zone_nick.get(fz_type_id, str(fz_type_id))
    if vri is None:
        return "unclear", "Для типа объекта нет сопоставленного ВРИ в словаре.", "", ""
    sections = zone_allowed.get(fz_type_id)
    if not sections or not any(sections.values()):
        return "no_zone_metadata", f"Для зоны «{zone_name}» нет описания разрешённых ВРИ.", vri, ""
    for section in ("main", "conditional", "auxiliary"):
        if is_allowed(vri, sections.get(section) or set()):
            return (
                f"allowed_{section}",
                f"ВРИ {vri} разрешён в зоне «{zone_name}» ({section}).",
                vri, "",
            )
    return "not_allowed", f"ВРИ {vri} не входит в разрешённые в зоне «{zone_name}».", vri, ""


def build_zone_gdf(zones: dict[str, Any], code_col: str):
    """Build a zones GeoDataFrame carrying ``fz_type_id`` from ``code_col``.

    Tolerates the urban_api nested shape (``functional_zone_type.id``) as a
    fallback when the flat code column is absent. Imports geopandas lazily.
    """
    import geopandas as gpd
    from shapely.geometry import shape

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
    return gpd.GeoDataFrame({"fz_type_id": zg_fz}, geometry=zg_geom, crs="EPSG:4326")


def join_objects_to_zones(feats: list[dict[str, Any]], zgdf) -> dict[int, int]:
    """Spatial-join object features to their containing zone.

    Returns ``{feature_index: fz_type_id}`` using each object's representative
    point and a ``within`` predicate (one deterministic zone per object).
    """
    import geopandas as gpd
    from shapely.geometry import shape

    o_geom = [shape(f["geometry"]) for f in feats]
    ogdf = gpd.GeoDataFrame({"_i": list(range(len(feats)))}, geometry=o_geom, crs="EPSG:4326")

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
    return fz_by_obj


def clean_result_properties(
    *,
    vri_text: Any,
    fz_type_id: int | None,
    zone_nick: dict[int, str],
    machine_verdict: str,
    reason: str,
    matched_vri_code: str,
    matched_vri_name: str | None,
) -> dict[str, Any]:
    """Build the whitelist of 7 PZZ result columns for one feature.

    Drops all input passthrough fields. ``Вердикт_ПЗЗ`` keeps its field name but
    holds the human-readable Russian label (frontend colors by it unchanged).
    """
    return {
        COL_VRI_TEXT: vri_text,
        COL_ZONE_CODE: str(fz_type_id) if fz_type_id is not None else "",
        COL_ZONE_NAME: zone_nick.get(fz_type_id, "") if fz_type_id is not None else "",
        COL_VERDICT: VERDICT_RU.get(machine_verdict, "Требуется ручная проверка"),
        COL_REASON: reason,
        COL_MATCHED_VRI_CODE: matched_vri_code,
        COL_MATCHED_VRI_NAME: matched_vri_name or "",
    }
