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
COL_RESOLUTION_BASIS = "Основание_подбора_ВРИ"
# Object kind, filled only by the building runner so its result can be split into
# separate «здания» / «сервисы» download layers. Absent from parcel results.
COL_CATEGORY = "Категория_объекта"
CATEGORY_BUILDING = "Здание"
CATEGORY_SERVICE = "Сервис"

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


def normalise_zone_code(value: Any) -> str:
    """Fold a ПЗЗ zone index to a match key: strip, casefold, ё→е, keep alnum only.

    «Ж-1» / «ж 1» / «Ж–1» (en-dash) all collapse to «ж1», so trivial spelling
    drift between the uploaded zone layer and the label mapping still matches.
    The original spelling is preserved for display separately (the runner keeps a
    normalised→raw map and passes it as ``zone_code_display``). An empty/blank
    code folds to ``""`` (never a match key).
    """
    if value in (None, ""):
        return ""
    text = str(value).strip().casefold().replace("ё", "е")
    return "".join(ch for ch in text if ch.isalnum())


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


def load_pzz_label_mapping(
    path: str,
) -> tuple[dict[str, dict[str, set[str]]], dict[str, str]]:
    """Load a PZZ letter-index → permitted-VRI mapping (the regular ``pzz_check``
    labels schema, e.g. ``pzz_zone_llm_labels_template.json``).

    The file is a list of zone entries, each ``{"zone_code": "Ж-1", "zone_name":
    ..., "main": [...], "conditional": [...], "auxiliary": [...]}`` where every VRI
    item carries a ``vri_code``. Returns ``({norm_code: {section: {vri_code}}},
    {norm_code: zone_name})`` keyed by the *normalised* index (see
    ``normalise_zone_code``), so it lines up with the zone layer's own code column
    — no urban_api ``functional_zone_type_id`` needed. This is the building
    counterpart to the ``pzz_check`` zone side: a user uploads a real ПЗЗ with
    letter indices instead of urban_api ids.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw if isinstance(raw, list) else (
        raw.get("zones") or raw.get("labels") or raw.get("pzz_zones") or []
    )
    allowed: dict[str, dict[str, set[str]]] = {}
    nick: dict[str, str] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        code = e.get("zone_code")
        key = normalise_zone_code(code)
        if not key:
            continue
        allowed[key] = {
            section: {
                v["vri_code"]
                for v in (e.get(section) or [])
                if isinstance(v, dict) and v.get("vri_code")
            }
            for section in ("main", "conditional", "auxiliary")
        }
        nick[key] = e.get("zone_name") or str(code).strip()
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
    fz_type_id: Any | None,
    zone_allowed: dict[Any, dict[str, set[str]]],
    zone_nick: dict[Any, str],
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


def build_zone_gdf(zones: dict[str, Any], code_col: str, *, numeric: bool = True):
    """Build a zones GeoDataFrame carrying the zone key from ``code_col``.

    ``numeric=True`` (urban_api scenario / building flow): the key is an integer
    ``functional_zone_type_id`` — tolerates the nested ``functional_zone_type.id``
    shape and drops zones whose code isn't an int.
    ``numeric=False`` (real ПЗЗ flow): the key is the code string verbatim (e.g.
    «Ж-1»), matched against a label mapping; the nested/int coercion doesn't apply.
    Imports geopandas lazily.
    """
    import geopandas as gpd
    from shapely.geometry import shape

    zg_geom, zg_fz = [], []
    for f in zones.get("features") or []:
        props = f.get("properties") or {}
        raw = props.get(code_col)
        if numeric and not raw and isinstance(props.get("functional_zone_type"), dict):
            raw = props["functional_zone_type"].get("id")
        if f.get("geometry") is None or raw in (None, ""):
            continue
        if numeric:
            try:
                key: Any = int(raw)
            except (TypeError, ValueError):
                continue
        else:
            key = normalise_zone_code(raw)
            if not key:
                continue
        zg_fz.append(key)
        zg_geom.append(shape(f["geometry"]))
    return gpd.GeoDataFrame({"fz_type_id": zg_fz}, geometry=zg_geom, crs="EPSG:4326")


def join_objects_to_zones(feats: list[dict[str, Any]], zgdf) -> dict[int, Any]:
    """Spatial-join object features to their containing zone.

    Returns ``{feature_index: fz_type_id}`` using each object's representative
    point and a ``within`` predicate (one deterministic zone per object). The
    zone key is an ``int`` for numeric (urban_api) codes and the code ``str`` for
    PZZ letter indices — matching how ``build_zone_gdf`` stored it.
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
            if fz is None or fz != fz:  # None / NaN
                continue
            # Numeric codes survive the left-join as floats (3 -> 3.0); coerce back
            # to int. String codes (PZZ indices) pass through untouched.
            key = fz if isinstance(fz, str) else int(fz)
            fz_by_obj[int(ogdf.loc[idx, "_i"])] = key
    return fz_by_obj


def clean_result_properties(
    *,
    vri_text: Any,
    fz_type_id: Any | None,
    zone_nick: dict[Any, str],
    machine_verdict: str,
    reason: str,
    matched_vri_code: str,
    matched_vri_name: str | None,
    resolution_basis: str | None = None,
    category: str | None = None,
    zone_code_display: str | None = None,
) -> dict[str, Any]:
    """Build the whitelist of PZZ result columns for one feature.

    Drops all input passthrough fields. ``Вердикт_ПЗЗ`` keeps its field name but
    holds the human-readable Russian label (frontend colors by it unchanged).
    ``resolution_basis`` (how the ВРИ was picked — by floors / service / type) is
    filled by the building runner and left empty by flows where it doesn't apply.
    ``category`` («Здание»/«Сервис») is filled only by the building runner so the
    result can be split into two download layers; omitted entirely otherwise.
    ``zone_code_display`` overrides the shown zone code — the building runner passes
    the user's verbatim ПЗЗ index (e.g. «Ж-1») when the join key is a normalised
    form; numeric flows leave it ``None`` and the code is shown as-is.
    """
    props: dict[str, Any] = {
        COL_VRI_TEXT: vri_text,
        COL_ZONE_CODE: (
            zone_code_display if zone_code_display is not None
            else (str(fz_type_id) if fz_type_id is not None else "")
        ),
        COL_ZONE_NAME: zone_nick.get(fz_type_id, "") if fz_type_id is not None else "",
        COL_VERDICT: VERDICT_RU.get(machine_verdict, "Требуется ручная проверка"),
        COL_REASON: reason,
        COL_MATCHED_VRI_CODE: matched_vri_code,
        COL_MATCHED_VRI_NAME: matched_vri_name or "",
        COL_RESOLUTION_BASIS: resolution_basis or "",
    }
    if category is not None:
        props[COL_CATEGORY] = category
    return props
