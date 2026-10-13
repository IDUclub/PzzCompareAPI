"""Deterministic service-placement check — where may a given ВРИ go under the ПЗЗ.

The user names a service in plain words («школа», «где поставить АЗС»); the API
resolves it to a ВРИ code up front (``resolve_service_vri``) and hands the code
here. Every PZZ zone polygon is then classed by whether that ВРИ is permitted:

    A   — main permitted use;
    B   — conditional or auxiliary use;
    B*  — permitted (A/B), but objects of this type already stand in the zone;
    C   — not permitted, or inside a production/warehouse sanitary buffer (СЗЗ);
    D   — the zone has no permitted-use description.

An optional cadastre layer refines the picture: production/warehouse parcels
buffered by 100 m form a СЗЗ that turns the overlapping part of A/B into C (for
sensitive uses only — housing, health, education, social, leisure), parcels of
the target type mark their zone B*, and housing buffered by 500 m flags which
suitable zones lie within reach of residents. No LLM, no embeddings; heavy geo
deps load lazily.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from service.domain import PipelineRequest
from service.infrastructure.runners._deterministic_pzz import (
    COL_CATEGORY,
    COL_REASON,
    build_zone_gdf,
    verdict as compute_verdict,
    zone_code_display_map,
    zone_codes_are_numeric,
)
from service.infrastructure.runners.pipeline_runner import (
    PipelineRunner,
    _build_output_glob,
)
from service.infrastructure.runners.uploaded_building_pzz_runner import (
    _enrich_zone_names,
    load_zone_mapping_for_request,
)

if TYPE_CHECKING:
    from service.settings import Settings

logger = logging.getLogger("service.tasks")

PLACEMENT_SUMMARY_KEY = "placement_summary"

COL_CLASS = "Класс_пригодности"
COL_SUITABILITY = "Пригодность"
COL_ZONE_CODE = "Код_зоны"
COL_ZONE_NAME = "Название_зоны"
COL_VRI_CODE = "Код_ВРИ"
COL_AREA_HA = "Площадь_га"
COL_NEAR_HOUSING = "Рядом_жильё_500м"
COL_PARCEL_VRI = "ВРИ_участка"

CATEGORY_CLASSES = "Классы пригодности"
CATEGORY_SZZ = "СЗЗ производственных объектов"
CATEGORY_REACH = "Зона доступности жилья"
CATEGORY_EXISTING = "Существующие объекты этого типа"

CLASS_ORDER = ("A", "B", "B*", "C", "D")
CLASS_RU = {
    "A": "Разрешено (основной вид использования)",
    "B": "Разрешено с условиями (условный или вспомогательный вид)",
    "B*": "Разрешено, но объекты этого типа в зоне уже есть",
    "C": "Не разрешено",
    "D": "Нет описания зоны",
}
_VERDICT_CLASS = {
    "allowed_main": "A",
    "allowed_conditional": "B",
    "allowed_auxiliary": "B",
    "not_allowed": "C",
}

SZZ_BUFFER_M = 100.0
REACH_BUFFER_M = 500.0
SZZ_REASON = (
    "Попадает в санитарно-защитную зону (100 м) производственных/складских участков."
)
# ВРИ families a production СЗЗ excludes (СанПиН 2.2.1/2.1.1.1200-03): housing,
# health care, education, culture/leisure, recreation.
_SZZ_SENSITIVE_PREFIXES = ("2", "3.2", "3.4", "3.5", "3.6", "5")
# Hand-written cadastre text hints for common target families — the cadastre's
# free-text VRI rarely carries the classifier name verbatim («для школы»).
_TARGET_HINTS = {
    "2": r"жил|ижс|многокварт",
    "3.4": r"здравоохран|больниц|поликлин|амбулатор|медицин|фельдшер|госпитал",
    "3.5": r"образован|школ|детск\w* сад|дошкольн|просвещ|гимнази|лице[йя]|колледж",
    "3.6": r"культур|клуб|библиотек|музе|театр",
    "3.7": r"религиоз|храм|церк|мечет|часовн",
    "4.4": r"магазин|торгов",
    "4.6": r"общественн\w* питан|кафе|ресторан|столов",
    "4.7": r"гостиниц|отел",
    "4.9.1": r"заправ|азс|автосервис|автомойк",
    "5.1": r"спорт|стадион|бассейн|физкульт",
    "6": r"производ|промышлен|склад",
}
_PROD_RE = re.compile(r"производ|промышлен|склад")
_HOUSING_RE = re.compile(r"жил|ижс|многокварт")
_NOT_HOUSING_RE = re.compile(r"нежил")
_MIN_PART_M2 = 1.0


def _code_pattern(prefix: str) -> re.Pattern[str]:
    """A ВРИ code or any of its sub-codes as a standalone token (``3.5`` ⊃ ``3.5.1``)."""
    return re.compile(rf"(?<![\d.]){re.escape(prefix)}(?:\.\d+)*(?![\d.])")


def _target_regex(code: str, service_name: str) -> re.Pattern[str] | None:
    """Text pattern for parcels of the target type: family hints + the stem of
    the user's own service words (≥5 letters, trimmed of a 2-letter ending)."""
    parts: list[str] = []
    for prefix, pattern in _TARGET_HINTS.items():
        if code == prefix or code.startswith(prefix + "."):
            parts.append(pattern)
    for word in re.findall(r"[а-яё]+", service_name.casefold().replace("ё", "е")):
        if len(word) >= 5:
            parts.append(re.escape(word[: max(4, len(word) - 2)]))
    return re.compile("|".join(parts)) if parts else None


def _load_vri_name(path: str, code: str) -> str:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    entry = (raw.get("by_code") or {}).get(code) or {}
    return str(entry.get("name") or "").strip()


def classify_parcel(
    text: str, target_code: str, target_re: re.Pattern[str] | None
) -> str:
    """``target`` / ``production`` / ``housing`` / ``other`` from free-text VRI."""
    low = text.casefold().replace("ё", "е")
    if _code_pattern(target_code).search(low) or (target_re and target_re.search(low)):
        return "target"
    if _PROD_RE.search(low) or _code_pattern("6").search(low):
        return "production"
    if not _NOT_HOUSING_RE.search(low) and (
        _HOUSING_RE.search(low) or _code_pattern("2").search(low)
    ):
        return "housing"
    return "other"


def is_szz_sensitive(code: str) -> bool:
    return any(code == p or code.startswith(p + ".") for p in _SZZ_SENSITIVE_PREFIXES)


def _ha(geom) -> float:
    return round(float(geom.area) / 1e4, 2)


class PlacementPzzRunner(PipelineRunner):
    """Class every PZZ zone by whether a target ВРИ may be placed there."""

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings

    def _load_cadastre(self, request: PipelineRequest, metric):
        """Cadastre parcels as a metric GeoDataFrame with a ``kind`` column, or None."""
        path = request.cadastral_data_path
        if not path or not Path(path).is_file():
            return None
        import geopandas as gpd
        from shapely.geometry import shape

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        col = request.cadastral_vri_col
        target_re = _target_regex(
            request.placement_vri_code, request.placement_service_name
        )
        geoms, texts, kinds = [], [], []
        for f in raw.get("features") or []:
            if f.get("geometry") is None:
                continue
            props = f.get("properties") or {}
            value = props.get(col) if col else None
            text = str(value).strip() if value not in (None, "") else ""
            geoms.append(shape(f["geometry"]))
            texts.append(text)
            kinds.append(
                classify_parcel(text, request.placement_vri_code, target_re)
                if text
                else "other"
            )
        if not geoms:
            return None
        gdf = gpd.GeoDataFrame(
            {"vri": texts, "kind": kinds}, geometry=geoms, crs="EPSG:4326"
        )
        gdf["geometry"] = gdf.geometry.make_valid()
        return gdf.to_crs(metric)

    def run(self, request: PipelineRequest) -> str:
        import geopandas as gpd
        from shapely.geometry import mapping

        output_dir = Path(request.outputs_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        target = request.placement_vri_code
        vri_name = _load_vri_name(
            request.vri_classifier_path or self._settings.default_vri_classifier_path,
            target,
        )
        zones = json.loads(
            Path(request.pzz_zones_data_path).read_text(encoding="utf-8")
        )
        code_col = request.pzz_zone_code_col or "zone_code"
        numeric_zones = zone_codes_are_numeric(zones, code_col)
        zone_allowed, zone_nick, used_default_mapping = load_zone_mapping_for_request(
            self._settings, request, numeric_zones
        )
        code_display = {} if numeric_zones else zone_code_display_map(zones, code_col)
        display_nick = _enrich_zone_names(
            zone_nick,
            code_display,
            zones,
            code_col,
            request.pzz_zone_name_col,
            numeric_zones,
        )
        uncovered_zones = (
            sorted(
                {code_display.get(k, k) for k in code_display if k not in zone_allowed}
            )
            if not numeric_zones
            else []
        )

        zgdf = build_zone_gdf(
            zones,
            code_col,
            numeric=numeric_zones,
            name_col=request.pzz_zone_name_col or None,
        )
        zgdf["geometry"] = zgdf.geometry.make_valid()
        metric = zgdf.estimate_utm_crs() if len(zgdf) else "EPSG:3857"
        zm = zgdf.to_crs(metric)

        parcels = self._load_cadastre(request, metric)
        szz = reach = existing = None
        parcel_counts: dict[str, int] = {}
        if parcels is not None:
            parcel_counts = {
                str(k): int(n) for k, n in parcels["kind"].value_counts().items()
            }
            prod = parcels[parcels["kind"] == "production"]
            if len(prod) and is_szz_sensitive(target):
                szz = prod.buffer(SZZ_BUFFER_M).union_all()
            homes = parcels[parcels["kind"] == "housing"]
            if len(homes):
                reach = homes.buffer(REACH_BUFFER_M).union_all()
            existing = parcels[parcels["kind"] == "target"]

        # Zones already holding parcels of the target type (by representative point).
        existing_by_zone: dict[int, int] = {}
        if existing is not None and len(existing) and len(zm):
            pts = existing[["geometry"]].copy()
            pts["geometry"] = existing.geometry.representative_point()
            joined = gpd.sjoin(pts, zm[["geometry"]], how="inner", predicate="within")
            for idx, n in joined.groupby("index_right").size().items():
                existing_by_zone[int(idx)] = int(n)

        # (metric geometry, class, reason, zone code, zone name)
        parts: list[tuple[Any, str, str, str, str]] = []
        for idx, row in zm.iterrows():
            key = row["fz_type_id"]
            label = row["zone_name"] or display_nick.get(key) or ""
            machine, reason, _, _ = compute_verdict(
                target, key, zone_allowed, display_nick, label or None
            )
            cls = _VERDICT_CLASS.get(machine, "D")
            code_txt = str(key) if numeric_zones else code_display.get(key, str(key))
            name_txt = label or code_txt
            geom = row.geometry
            if cls in ("A", "B") and existing_by_zone.get(int(idx)):
                cls = "B*"
                reason = (
                    f"{reason} В зоне уже есть объекты этого типа "
                    f"(участков: {existing_by_zone[int(idx)]})."
                )
            if cls in ("A", "B", "B*") and szz is not None and geom.intersects(szz):
                inside = geom.intersection(szz)
                if inside.area >= _MIN_PART_M2:
                    parts.append((inside, "C", SZZ_REASON, code_txt, name_txt))
                geom = geom.difference(szz)
            if geom.is_empty or geom.area < _MIN_PART_M2:
                continue
            parts.append((geom, cls, reason, code_txt, name_txt))

        features: list[dict[str, Any]] = []
        classes: dict[str, dict[str, float]] = {
            c: {"zones": 0, "ha": 0.0} for c in CLASS_ORDER
        }
        best: list[tuple[float, str, str]] = []
        if parts:
            near = [
                bool(reach is not None and p[0].intersects(reach)) for p in parts
            ]
            areas = [_ha(p[0]) for p in parts]
            geoms4326 = gpd.GeoSeries([p[0] for p in parts], crs=metric).to_crs(
                "EPSG:4326"
            )
            for (_, cls, reason, code_txt, name_txt), ha, is_near, g in zip(
                parts, areas, near, geoms4326
            ):
                props: dict[str, Any] = {
                    COL_CLASS: cls,
                    COL_SUITABILITY: CLASS_RU[cls],
                    COL_REASON: reason,
                    COL_ZONE_CODE: code_txt,
                    COL_ZONE_NAME: name_txt,
                    COL_VRI_CODE: target,
                    COL_AREA_HA: ha,
                    COL_CATEGORY: CATEGORY_CLASSES,
                }
                if reach is not None and cls in ("A", "B", "B*"):
                    props[COL_NEAR_HOUSING] = "да" if is_near else "нет"
                features.append(
                    {"type": "Feature", "properties": props, "geometry": mapping(g)}
                )
                bucket = classes[cls]
                bucket["zones"] += 1
                bucket["ha"] = round(bucket["ha"] + ha, 2)
                if cls == "A":
                    best.append((ha, code_txt, name_txt))

        def _overlay(geom, category: str) -> None:
            if geom is None or geom.is_empty:
                return
            g4326 = gpd.GeoSeries([geom], crs=metric).to_crs("EPSG:4326").iloc[0]
            features.append(
                {
                    "type": "Feature",
                    "properties": {COL_CATEGORY: category, COL_AREA_HA: _ha(geom)},
                    "geometry": mapping(g4326),
                }
            )

        _overlay(szz, CATEGORY_SZZ)
        _overlay(reach, CATEGORY_REACH)
        if existing is not None and len(existing):
            for _, r in existing.to_crs("EPSG:4326").iterrows():
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            COL_CATEGORY: CATEGORY_EXISTING,
                            COL_PARCEL_VRI: r["vri"][:300],
                        },
                        "geometry": mapping(r.geometry),
                    }
                )

        best.sort(key=lambda t: -t[0])
        summary = {
            "vri_code": target,
            "vri_name": vri_name,
            "service_name": request.placement_service_name,
            "classes": classes,
            "zones_total": int(len(zm)),
            "best_zones": [
                {"code": code, "name": name, "ha": ha} for ha, code, name in best[:5]
            ],
            "szz_applied": szz is not None,
            "szz_ha": _ha(szz) if szz is not None else 0.0,
            "reach_ha": _ha(reach) if reach is not None else 0.0,
            "parcels": parcel_counts,
            "cadastre_used": parcels is not None,
            "used_default_mapping": used_default_mapping,
            "uncovered_zones": uncovered_zones,
        }
        result = {
            "type": "FeatureCollection",
            "features": features,
            PLACEMENT_SUMMARY_KEY: summary,
        }
        out_path = (
            output_dir / f"pzz_compare_spatial_first_{request.task_external_id}.geojson"
        )
        out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        logger.info(
            json.dumps(
                {
                    "stage": "placement_pzz",
                    "status": "finished",
                    "external_id": request.task_external_id,
                    "vri_code": target,
                    "zones": len(zm),
                    "classes": {c: classes[c]["zones"] for c in CLASS_ORDER},
                    "cadastre_used": parcels is not None,
                    "szz_applied": szz is not None,
                    "used_default_mapping": used_default_mapping,
                }
            )
        )
        return _build_output_glob(output_dir, request.task_external_id)
