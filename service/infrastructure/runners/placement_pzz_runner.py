"""Deterministic service-placement check — where may a given ВРИ go under the ПЗЗ.

The user names a service in plain words («школа», «где поставить АЗС»); the API
resolves it to a ВРИ code and gathers its NormGraph context (the ПЗЗ document and
the norms of the service, ``placement_context``) up front. Every PZZ zone polygon
is then classed (``service.domain.placement``):

    A / B   — main / conditional or auxiliary permitted use;
    A* / B* — the same, but an object of this type already stands in the zone;
    C       — not permitted, inside a production СЗЗ, or cut by a strict norm;
    D       — the zone has no permitted-use description.

Zone permissions: a user descriptions file, else the ПЗЗ from NormGraph, else
the built-in mapping. An optional cadastre layer refines the picture:
production parcels buffered by the СЗЗ of their hazard class (read from the name
of the zone they stand in), the strict norms of the service cut their buffers out
of suitable zones, soft «within N m of housing» norms mark how much of a zone is
in reach, and parcels of the target type mark their zone A*/B*. The zone's own
limits (floors, height, coverage) are listed, never lower the class. No LLM, no
network; heavy geo deps load lazily.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from service.application.use_cases.check_vri import RESIDENTIAL_VRI, zone_limits
from service.domain import PipelineRequest
from service.domain.placement import (
    CLASS_ORDER,
    CLASS_RU,
    DEFAULT_HAZARD_CLASS,
    EFFECT_EXCLUDE_NEAR,
    EFFECT_PREFER_NEAR,
    LAYER_HOUSING,
    LAYER_TARGET,
    SOFT,
    STRICT,
    SUITABLE_CLASSES,
    PlacementRule,
    hazard_class,
    szz_width_m,
    with_existing,
)
from service.infrastructure.runners._deterministic_pzz import (
    COL_CATEGORY,
    COL_REASON,
    build_zone_gdf,
    normalise_zone_code,
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
COL_ZONE_LIMITS = "Параметры_зоны"
COL_REACH_SHARE = "Доля_рядом_с_жильём_%"
COL_PARCEL_VRI = "ВРИ_участка"
COL_REQUIREMENT = "Требование"
COL_SOURCE = "Источник"

CATEGORY_CLASSES = "Классы пригодности"
CATEGORY_SZZ = "СЗЗ производственных объектов"
CATEGORY_NORMS = "Буферы строгих норм"
CATEGORY_REACH = "Зона доступности жилья"
CATEGORY_EXISTING = "Существующие объекты этого типа"

REGULATIONS_USER = "user"
REGULATIONS_NORMGRAPH = "normgraph"
REGULATIONS_DEFAULT = "default"

NORM_APPLIED = "applied"
NORM_NOT_APPLIED = "not_applied"
NORM_LISTED = "listed"

_VERDICT_CLASS = {
    "allowed_main": "A",
    "allowed_conditional": "B",
    "allowed_auxiliary": "B",
    "not_allowed": "C",
}
_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}
SZZ_SOURCE = "СанПиН 2.2.1/2.1.1.1200-03"
SZZ_REASON = (
    "Попадает в санитарно-защитную зону производственных/складских участков "
    "(ширина по классу опасности, СанПиН 2.2.1/2.1.1.1200-03)."
)

# ВРИ families a production СЗЗ excludes (СанПиН 2.2.1/2.1.1.1200-03): housing,
# health care, education, culture/leisure, recreation.
_SZZ_SENSITIVE_PREFIXES = ("2", "3.2", "3.4", "3.5", "3.6", "5")
# Hand-written cadastre text hints for common target families — the cadastre's
# free-text VRI rarely carries the classifier name verbatim («для школы»).
_TARGET_HINTS = {
    "2": r"жил|ижс|многокварт",
    "3.4": r"здравоохран|больниц|поликлин|амбулатор|медицин|фельдшер|госпитал",
    "3.5": (
        r"образован|(?<![а-я])школ|детск\w* сад|дошкольн|просвещ|гимнази"
        r"|лице[йя]|колледж"
    ),
    "3.6": r"культур|клуб|библиотек|музе|театр",
    "3.7": r"религиоз|храм|церк|мечет|часовн",
    "4.4": r"магазин|торгов",
    "4.6": r"общественн\w* питан|кафе|ресторан|столов",
    "4.7": r"гостиниц|отел",
    "4.9.1": r"заправ|азс|автосервис|автомойк",
    "5.1": r"спорт|стадион|бассейн|физкульт",
    "6": r"производ|промышлен|склад",
}
# A school and a kindergarten share ВРИ 3.5.1, yet a kindergarten nearby is not
# «a school already stands here».
_SCHOOL_RE = re.compile(r"(?<![а-я])школ|общеобразоват|лице[йя]|гимнази")
_PRESCHOOL_RE = re.compile(r"дошкол|детск\w* сад|ясл")
# The 3.5 family hint also catches ВРИ 3.5.2 (colleges, universities, driving
# schools), which is not «a school already stands here» for a 3.5.1 target.
_GENERAL_EDUCATION_CODE = "3.5.1"
_VOCATIONAL_RE = re.compile(
    r"высш|профессиональн|колледж|училищ|техникум|университет|институт|академи"
    r"|автошкол|(?:авто|водител)\w*\s+школ\w*"
)
_PROD_RE = re.compile(r"производ|промышлен|склад")
_HOUSING_RE = re.compile(r"жил|ижс|многокварт")
_NOT_HOUSING_RE = re.compile(r"нежил")
_MIN_PART_M2 = 1.0


def _fold(text: str) -> str:
    return text.casefold().replace("ё", "е")


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
    for word in re.findall(r"[а-я]+", _fold(service_name)):
        if len(word) >= 5:
            parts.append(r"(?<![а-я])" + re.escape(word[: max(4, len(word) - 2)]))
    return re.compile("|".join(parts)) if parts else None


def _excluded_target_regex(service_name: str) -> re.Pattern[str] | None:
    """Parcels that match the target family but are another service of it."""
    name = _fold(service_name)
    if _SCHOOL_RE.search(name) and not _PRESCHOOL_RE.search(name):
        return _PRESCHOOL_RE
    return None


def _load_vri_name(path: str, code: str) -> str:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    entry = (raw.get("by_code") or {}).get(code) or {}
    return str(entry.get("name") or "").strip()


def load_placement_context(path: str) -> dict[str, Any]:
    """The NormGraph context the API stored for the task; empty when absent."""
    if not path or not Path(path).is_file():
        return {}
    try:
        context = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("placement context unreadable (%s); running without it", exc)
        return {}
    return context if isinstance(context, dict) else {}


def _is_general_education(code: str) -> bool:
    return code == _GENERAL_EDUCATION_CODE or code.startswith(
        _GENERAL_EDUCATION_CODE + "."
    )


def _is_vocational_only(low: str) -> bool:
    """Vocational or higher education with no school or kindergarten named apart
    from it («автомобильная школа» is not a school)."""
    if not _VOCATIONAL_RE.search(low):
        return False
    rest = _VOCATIONAL_RE.sub(" ", low)
    return not (_SCHOOL_RE.search(rest) or _PRESCHOOL_RE.search(rest))


def classify_parcel(
    text: str,
    target_code: str,
    target_re: re.Pattern[str] | None,
    excluded_re: re.Pattern[str] | None = None,
) -> str:
    """``target`` / ``production`` / ``housing`` / ``other`` from free-text VRI.

    ``excluded_re`` drops parcels of a sibling service from the target (a
    kindergarten when placing a school) unless the text names a school too.
    A 3.5.1 target never takes a purely vocational or higher-education parcel.
    """
    low = _fold(text)
    if _code_pattern(target_code).search(low) or (target_re and target_re.search(low)):
        if _is_general_education(target_code) and _is_vocational_only(low):
            return "other"
        if excluded_re is None or not excluded_re.search(low) or _SCHOOL_RE.search(low):
            return "target"
        return "other"
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


def _number(value: float) -> str:
    return f"{value:g}"


def _citation(rule: PlacementRule) -> str:
    clause = f", п. {rule.clause}" if rule.clause else ""
    return f"{rule.document}{clause}" if rule.document else "NormGraph"


def _limits_text(limits: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"{limit['label']} {_number(limit['value'])} {limit['unit']}"
        for limit in limits
    )


def normgraph_zone_mapping(
    zones: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, set[str]]], dict[str, str], dict[str, dict[str, Any]]]:
    """``(allowed, names, zones)`` keyed by the normalised zone code."""
    allowed: dict[str, dict[str, set[str]]] = {}
    names: dict[str, str] = {}
    by_key: dict[str, dict[str, Any]] = {}
    for zone in zones:
        key = normalise_zone_code(zone.get("code"))
        if not key:
            continue
        sections: dict[str, set[str]] = {
            "main": set(),
            "conditional": set(),
            "auxiliary": set(),
        }
        for use in zone.get("uses") or []:
            if use.get("section") in sections:
                sections[use["section"]].update(use.get("codes") or [])
        allowed[key] = sections
        names[key] = str(zone.get("name") or zone.get("code") or "")
        by_key[key] = zone
    return allowed, names, by_key


@dataclass
class _Cut:
    """A strict exclusion: its geometry and why a zone part inside it is C."""

    geom: Any
    reason: str
    category: str
    requirement: str
    source: str
    rules: list[PlacementRule] = field(default_factory=list)
    ha: float = 0.0


@dataclass
class _NormOutcome:
    rule: PlacementRule
    status: str
    note: str = ""
    ha: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strictness": self.rule.strictness,
            "requirement": self.rule.requirement,
            "document": self.rule.document,
            "clause": self.rule.clause,
            "reviewed": self.rule.reviewed,
            "status": self.status,
            "note": self.note or self.rule.reason,
            "ha": self.ha,
        }


def norm_cuts(
    rules: list[PlacementRule], layers: dict[str, Any]
) -> tuple[list[_Cut], list[_NormOutcome], Any, float | None]:
    """Strict cuts, the outcome of every rule, the soft housing reach and its distance.

    Strict rules with the same effect, layer and distance share one cut, citing
    every source; the soft reach takes the tightest «within N m of housing».
    """
    cuts: dict[tuple[str, str | None, float | None], _Cut] = {}
    outcomes: list[_NormOutcome] = []
    reach_distances: list[float] = []
    for rule in rules:
        layer = layers.get(rule.layer or "")
        if rule.strictness == STRICT:
            if layer is None or layer.is_empty:
                outcomes.append(
                    _NormOutcome(rule, NORM_NOT_APPLIED, "нет данных кадастра")
                )
                continue
            key = (rule.effect, rule.layer, rule.distance_m)
            if key not in cuts:
                geom = (
                    layer.buffer(rule.distance_m)
                    if rule.effect == EFFECT_EXCLUDE_NEAR and rule.distance_m
                    else layer
                )
                cuts[key] = _Cut(
                    geom=geom,
                    reason="",
                    category=CATEGORY_NORMS,
                    requirement=rule.requirement,
                    source=_citation(rule),
                )
            elif _citation(rule) not in cuts[key].source:
                cuts[key].source += f"; {_citation(rule)}"
            cuts[key].rules.append(rule)
            outcomes.append(_NormOutcome(rule, NORM_APPLIED))
        elif rule.strictness == SOFT and rule.effect == EFFECT_PREFER_NEAR:
            if layer is None or layer.is_empty or not rule.distance_m:
                outcomes.append(
                    _NormOutcome(rule, NORM_NOT_APPLIED, "нет данных кадастра")
                )
                continue
            reach_distances.append(rule.distance_m)
            outcomes.append(_NormOutcome(rule, NORM_APPLIED))
        else:
            outcomes.append(_NormOutcome(rule, NORM_LISTED))
    for cut in cuts.values():
        cut.reason = f"Строгая норма: {cut.requirement} ({cut.source})."
    distance = min(reach_distances) if reach_distances else None
    reach = layers[LAYER_HOUSING].buffer(distance) if distance else None
    return list(cuts.values()), outcomes, reach, distance


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
        excluded_re = _excluded_target_regex(request.placement_service_name)
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
                classify_parcel(
                    text, request.placement_vri_code, target_re, excluded_re
                )
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

    def _is_default_labels(self, path: str) -> bool:
        """The API stores the built-in labels path when the user sent no zone
        descriptions, so that file is not the user's own."""
        default = getattr(self._settings, "default_pzz_zone_labels_path", "")
        if not path or not default:
            return False
        return Path(path).resolve() == Path(default).resolve()

    def _zone_permissions(
        self, request: PipelineRequest, context: dict[str, Any], numeric: bool
    ) -> tuple[dict, dict, dict[str, dict[str, Any]], str]:
        """``(allowed, names, NormGraph zones, source)``: user file > NormGraph > built-in."""
        allowed, nick, used_default = load_zone_mapping_for_request(
            self._settings, request, numeric
        )
        used_default = used_default or self._is_default_labels(
            request.pzz_zone_vri_labels_path
        )
        regulations = context.get("regulations") or {}
        ng_zones: dict[str, dict[str, Any]] = {}
        if not numeric and regulations.get("zones"):
            ng_allowed, ng_names, ng_zones = normgraph_zone_mapping(
                regulations["zones"]
            )
            if used_default:
                return ng_allowed, ng_names, ng_zones, REGULATIONS_NORMGRAPH
        source = REGULATIONS_DEFAULT if used_default else REGULATIONS_USER
        return allowed, nick, ng_zones, source

    @staticmethod
    def _szz(prod, zm, hazards: dict[int, int | None]):
        """``(geometry, by_class, parcels without a class)``: production parcels
        buffered by the СЗЗ of the hazard class of the zone each one stands in."""
        import geopandas as gpd

        pts = prod[["geometry"]].copy()
        pts["geometry"] = prod.geometry.representative_point()
        joined = gpd.sjoin(pts, zm[["geometry"]], how="left", predicate="within")
        zone_of = joined[~joined.index.duplicated(keep="first")]["index_right"]
        classes = [
            hazards.get(int(zone_of[i])) if zone_of[i] == zone_of[i] else None
            for i in prod.index
        ]
        geom = gpd.GeoSeries(
            [g.buffer(szz_width_m(c)) for g, c in zip(prod.geometry, classes)],
            crs=prod.crs,
        ).union_all()
        counts: dict[int, int] = {}
        for c in classes:
            counts[c or DEFAULT_HAZARD_CLASS] = (
                counts.get(c or DEFAULT_HAZARD_CLASS, 0) + 1
            )
        by_class = [
            {"hazard_class": _ROMAN[c], "width_m": szz_width_m(c), "parcels": n}
            for c, n in sorted(counts.items())
        ]
        return geom, by_class, sum(1 for c in classes if c is None)

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
        context = load_placement_context(request.placement_context_path)
        zones = json.loads(
            Path(request.pzz_zones_data_path).read_text(encoding="utf-8")
        )
        code_col = request.pzz_zone_code_col or "zone_code"
        numeric_zones = zone_codes_are_numeric(zones, code_col)
        zone_allowed, zone_nick, ng_zones, regulations_source = self._zone_permissions(
            request, context, numeric_zones
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
        labels = {
            int(idx): str(row["zone_name"] or display_nick.get(row["fz_type_id"]) or "")
            for idx, row in zm.iterrows()
        }
        hazards = {idx: hazard_class(label) for idx, label in labels.items()}

        parcels = self._load_cadastre(request, metric)
        szz = None
        szz_by_class: list[dict[str, Any]] = []
        szz_defaulted = 0
        layers: dict[str, Any] = {}
        existing = None
        parcel_counts: dict[str, int] = {}
        if parcels is not None:
            parcel_counts = {
                str(k): int(n) for k, n in parcels["kind"].value_counts().items()
            }
            prod = parcels[parcels["kind"] == "production"]
            if len(prod) and is_szz_sensitive(target):
                szz, szz_by_class, szz_defaulted = self._szz(prod, zm, hazards)
            homes = parcels[parcels["kind"] == "housing"]
            existing = parcels[parcels["kind"] == "target"]
            if len(homes):
                layers[LAYER_HOUSING] = homes.geometry.union_all()
            if len(existing):
                layers[LAYER_TARGET] = existing.geometry.union_all()

        rules = [PlacementRule.from_dict(r) for r in context.get("rules") or []]
        cuts, outcomes, reach, reach_m = norm_cuts(rules, layers)
        if szz is not None:
            cuts.insert(
                0,
                _Cut(
                    geom=szz,
                    reason=SZZ_REASON,
                    category=CATEGORY_SZZ,
                    requirement="",
                    source=SZZ_SOURCE,
                ),
            )

        existing_by_zone: dict[int, int] = {}
        if existing is not None and len(existing) and len(zm):
            pts = existing[["geometry"]].copy()
            pts["geometry"] = existing.geometry.representative_point()
            joined = gpd.sjoin(pts, zm[["geometry"]], how="inner", predicate="within")
            for idx, n in joined.groupby("index_right").size().items():
                existing_by_zone[int(idx)] = int(n)

        # (metric geometry, class, reason, zone code, zone name, zone limits)
        parts: list[tuple[Any, str, str, str, str, str]] = []
        for idx, row in zm.iterrows():
            key = row["fz_type_id"]
            label = labels[int(idx)]
            machine, reason, _, _ = compute_verdict(
                target, key, zone_allowed, display_nick, label or None
            )
            cls = _VERDICT_CLASS.get(machine, "D")
            code_txt = str(key) if numeric_zones else code_display.get(key, str(key))
            name_txt = label or code_txt
            if cls not in SUITABLE_CLASSES:
                parts.append((row.geometry, cls, reason, code_txt, name_txt, ""))
                continue
            limits = ""
            if key in ng_zones:
                limits = _limits_text(
                    zone_limits(
                        ng_zones[key], target, residential=target in RESIDENTIAL_VRI
                    )
                )
            if existing_by_zone.get(int(idx)):
                cls = with_existing(cls)
                reason = (
                    f"{reason} В зоне уже есть объекты этого типа "
                    f"(участков: {existing_by_zone[int(idx)]})."
                )
            geom = row.geometry
            for cut in cuts:
                if geom.is_empty or not geom.intersects(cut.geom):
                    continue
                inside = geom.intersection(cut.geom)
                if inside.area >= _MIN_PART_M2:
                    parts.append((inside, "C", cut.reason, code_txt, name_txt, ""))
                    cut.ha += inside.area / 1e4
                geom = geom.difference(cut.geom)
            if geom.is_empty or geom.area < _MIN_PART_M2:
                continue
            parts.append((geom, cls, reason, code_txt, name_txt, limits))

        for cut in cuts:
            for outcome in outcomes:
                if outcome.rule in cut.rules:
                    outcome.ha = round(cut.ha, 2)

        features: list[dict[str, Any]] = []
        classes: dict[str, dict[str, float]] = {
            c: {"zones": 0, "ha": 0.0} for c in CLASS_ORDER
        }
        best: list[tuple[float, str, str, str]] = []
        reach_zones = 0
        parts = [p for p in parts if not p[0].is_empty and p[0].area >= _MIN_PART_M2]
        if parts:
            geoms4326 = gpd.GeoSeries([p[0] for p in parts], crs=metric).to_crs(
                "EPSG:4326"
            )
            for (geom, cls, reason, code_txt, name_txt, limits), g in zip(
                parts, geoms4326
            ):
                ha = _ha(geom)
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
                if limits:
                    props[COL_ZONE_LIMITS] = limits
                if reach is not None and cls in SUITABLE_CLASSES:
                    share = 0.0
                    if geom.intersects(reach):
                        share = geom.intersection(reach).area / geom.area * 100
                    props[COL_REACH_SHARE] = round(share)
                    reach_zones += int(share > 0)
                features.append(
                    {"type": "Feature", "properties": props, "geometry": mapping(g)}
                )
                bucket = classes[cls]
                bucket["zones"] += 1
                bucket["ha"] = round(bucket["ha"] + ha, 2)
                if cls == "A":
                    best.append((ha, code_txt, name_txt, limits))

        def _overlay(geom, properties: dict[str, Any]) -> None:
            if geom is None or geom.is_empty:
                return
            g4326 = gpd.GeoSeries([geom], crs=metric).to_crs("EPSG:4326").iloc[0]
            features.append(
                {
                    "type": "Feature",
                    "properties": {**properties, COL_AREA_HA: _ha(geom)},
                    "geometry": mapping(g4326),
                }
            )

        for cut in cuts:
            properties: dict[str, Any] = {COL_CATEGORY: cut.category}
            if cut.category == CATEGORY_NORMS:
                properties[COL_REQUIREMENT] = cut.requirement
                properties[COL_SOURCE] = cut.source
            _overlay(cut.geom, properties)
        if reach is not None and reach_m:
            _overlay(
                reach,
                {
                    COL_CATEGORY: CATEGORY_REACH,
                    COL_REQUIREMENT: (
                        f"в пределах {_number(reach_m)} м от жилых участков"
                    ),
                },
            )
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
        document = (context.get("regulations") or {}).get("document") or {}
        summary = {
            "vri_code": target,
            "vri_name": vri_name,
            "service_name": request.placement_service_name,
            "classes": classes,
            "zones_total": int(len(zm)),
            "best_zones": [
                {"code": code, "name": name, "ha": ha, "limits": limits}
                for ha, code, name, limits in best[:5]
            ],
            "regulations": {
                "source": regulations_source,
                "document": document.get("name") or "",
                "version": document.get("version") or "",
            },
            "szz_applied": szz is not None,
            "szz_ha": _ha(szz) if szz is not None else 0.0,
            "szz_by_class": szz_by_class,
            "szz_default_class_parcels": szz_defaulted,
            "norms": [o.to_dict() for o in outcomes],
            "reach_m": reach_m,
            "reach_ha": _ha(reach) if reach is not None else 0.0,
            "reach_zones": reach_zones,
            "parcels": parcel_counts,
            "cadastre_used": parcels is not None
            and context.get("input_kind") != "scenario_objects",
            "scenario_objects_used": parcels is not None
            and context.get("input_kind") == "scenario_objects",
            "used_default_mapping": regulations_source == REGULATIONS_DEFAULT,
            "uncovered_zones": uncovered_zones,
            "notes": [str(n) for n in context.get("notes") or []],
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
                    "regulations": regulations_source,
                    "norms": {
                        s: sum(1 for o in outcomes if o.status == s)
                        for s in (NORM_APPLIED, NORM_NOT_APPLIED, NORM_LISTED)
                    },
                }
            )
        )
        return _build_output_glob(output_dir, request.task_external_id)
