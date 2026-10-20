"""Service placement: suitability classes, СЗЗ widths and norms turned into rules.

A placement answers «where may this service go». Every zone of the ПЗЗ gets a
suitability class; the norms NormGraph holds for the service are inverted from
«check an existing object» into «where a new one may stand»:

- ``distance_from_source`` (buffered) — «not closer than N m to X»: a strict cut
  of the N m buffer around layer X;
- ``distance_from_source`` (source geometry) — «not on X»: a strict cut of X;
- ``presence_within`` / ``accessibility_within`` — «within N m of X»: soft, it
  only flags the parts of suitable zones within reach of X;
- ``object_attribute_threshold`` — a requirement to the building itself: soft,
  listed for the designer;
- anything else, or a layer the request has no data for — not checked, listed.

Pure: no I/O, no geo libraries.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

CLASS_A = "A"
CLASS_A_EXISTING = "A*"
CLASS_B = "B"
CLASS_B_EXISTING = "B*"
CLASS_C = "C"
CLASS_D = "D"
CLASS_ORDER = (CLASS_A, CLASS_A_EXISTING, CLASS_B, CLASS_B_EXISTING, CLASS_C, CLASS_D)
SUITABLE_CLASSES = frozenset({CLASS_A, CLASS_A_EXISTING, CLASS_B, CLASS_B_EXISTING})
CLASS_RU = {
    CLASS_A: "Подходит (основной вид использования)",
    CLASS_A_EXISTING: "Подходит, но объект этого типа в зоне уже есть",
    CLASS_B: "Подходит с ограничением (условно разрешённый или вспомогательный вид)",
    CLASS_B_EXISTING: "Подходит с ограничением, но объект этого типа в зоне уже есть",
    CLASS_C: "Не подходит",
    CLASS_D: "Нет описания зоны",
}
_EXISTING_CLASS = {CLASS_A: CLASS_A_EXISTING, CLASS_B: CLASS_B_EXISTING}

STRICT = "strict"
SOFT = "soft"
UNCHECKED = "unchecked"

EFFECT_EXCLUDE_NEAR = "exclude_near"
EFFECT_EXCLUDE_ON = "exclude_on"
EFFECT_PREFER_NEAR = "prefer_near"
EFFECT_NONE = "none"

LAYER_HOUSING = "housing"
LAYER_TARGET = "target"
_LAYER_GENITIVE = {
    LAYER_HOUSING: "жилых участков",
    LAYER_TARGET: "участков существующих объектов этого типа",
}
_LAYER_PREPOSITIONAL = {
    LAYER_HOUSING: "жилых участках",
    LAYER_TARGET: "участках существующих объектов этого типа",
}

# СанПиН 2.2.1/2.1.1.1200-03, п. 7.1: the approximate СЗЗ by hazard class.
SZZ_WIDTH_BY_HAZARD_CLASS = {1: 1000.0, 2: 500.0, 3: 300.0, 4: 100.0, 5: 50.0}
# A production zone that names no class gets IV — the common case of a
# «зона производственных объектов» without a class in its name.
DEFAULT_HAZARD_CLASS = 4

_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
_LOOKALIKES = str.maketrans({"І": "I", "Ӏ": "I"})
_CLASS_TOKEN = r"(?:IV|V|I{1,3}|[1-5])"
_CLASS_SEPARATOR = r"\s*(?:-|–|—|,|и)\s*"
_HAZARD_RE = re.compile(
    rf"(?<![A-Za-z0-9])({_CLASS_TOKEN}(?:{_CLASS_SEPARATOR}{_CLASS_TOKEN})*)"
    r"(?![A-Za-z0-9])\s*-?\s*(?i:(?:го\s+)?класс)",
)
_HOUSING_ENTITY_RE = re.compile(r"жил|многокварт|ижс")
_NOT_HOUSING_ENTITY_RE = re.compile(r"нежил")
_MAX_TEXT = 300


def hazard_class(text: str | None) -> int | None:
    """The most dangerous hazard class a zone name mentions («III-V класса» -> 3)."""
    if not text:
        return None
    classes: list[int] = []
    for match in _HAZARD_RE.finditer(text.translate(_LOOKALIKES)):
        for token in re.split(_CLASS_SEPARATOR, match.group(1)):
            value = int(token) if token.isdigit() else _ROMAN.get(token)
            if value is not None:
                classes.append(value)
    return min(classes) if classes else None


def szz_width_m(hazard: int | None) -> float:
    return SZZ_WIDTH_BY_HAZARD_CLASS[hazard or DEFAULT_HAZARD_CLASS]


def with_existing(suitability: str) -> str:
    """A* / B* for a suitable zone that already holds an object of this type."""
    return _EXISTING_CLASS.get(suitability, suitability)


@dataclass(frozen=True, slots=True)
class PlacementRule:
    """One norm of the service, as it limits where a new object may stand."""

    restriction_id: str
    strictness: str
    effect: str
    requirement: str
    text: str
    document: str
    clause: str
    reviewed: bool
    layer: str | None = None
    distance_m: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlacementRule":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in raw.items() if k in fields})


def casefold_entity(text: str | None) -> str:
    return (text or "").casefold().replace("ё", "е").strip()


def layer_kind(entity: str | None, service_entities: frozenset[str]) -> str | None:
    """``target`` for the service itself, ``housing`` for residential layers."""
    low = casefold_entity(entity)
    if not low:
        return None
    if low in service_entities:
        return LAYER_TARGET
    if _HOUSING_ENTITY_RE.search(low) and not _NOT_HOUSING_ENTITY_RE.search(low):
        return LAYER_HOUSING
    return None


def _layer_entities(plan: dict[str, Any]) -> dict[str, str]:
    layers = (plan.get("declared_requirements") or {}).get("layers") or []
    return {
        str(layer.get("role")): str(layer.get("entity") or "")
        for layer in layers
        if layer.get("role")
    }


def _distance(params: dict[str, Any]) -> float | None:
    value = params.get("distance_m")
    limit = params.get("limit") or {}
    if value is None and limit.get("kind") == "distance":
        value = limit.get("meters")
    try:
        distance = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return distance if distance and distance > 0 else None


def _counterpart(
    own: list[str], others: list[str], service_entities: frozenset[str]
) -> str | None:
    """The layer the service is measured against: the other side of the norm."""
    if any(casefold_entity(e) in service_entities for e in own):
        return others[0] if others else None
    if any(casefold_entity(e) in service_entities for e in others):
        return own[0] if own else None
    return None


def _number(value: float) -> str:
    return f"{value:g}"


def rule_from_restriction(
    hit: dict[str, Any], service_entities: frozenset[str]
) -> PlacementRule:
    """Invert one NormGraph restriction of the service into a placement rule.

    ``service_entities`` are the casefolded names of the service as NormGraph
    knows it; a layer named so is the layer of existing objects of this type.
    """
    provenance = hit.get("provenance") or {}
    plan = hit.get("check_plan") or {}
    params = plan.get("params") or {}
    text = " ".join(str(hit.get("extraction_text") or "").split())[:_MAX_TEXT]
    base: dict[str, Any] = {
        "restriction_id": str(hit.get("id") or ""),
        "text": text,
        "document": str(provenance.get("name") or ""),
        "clause": str(provenance.get("numbering") or ""),
        "reviewed": hit.get("check_plan_review_status") == "approved",
    }

    def unchecked(reason: str) -> PlacementRule:
        return PlacementRule(
            strictness=UNCHECKED,
            effect=EFFECT_NONE,
            requirement=text,
            reason=reason,
            **base,
        )

    template = plan.get("template")
    layers = _layer_entities(plan)
    if not template or template == "unsupported":
        return unchecked("норма не формализована для автоматической проверки")
    if template == "object_attribute_threshold":
        return PlacementRule(
            strictness=SOFT,
            effect=EFFECT_NONE,
            requirement=text,
            reason="требование к самому объекту, учитывается при проектировании",
            **base,
        )
    if template == "distance_from_source":
        source = [layers.get(str(params.get("source_layer") or ""), "")]
        targets = [layers.get(str(role), "") for role in params.get("targets") or []]
        other = _counterpart(source, targets, service_entities)
        if other is None:
            return unchecked("норма не относится к размещаемому объекту")
        kind = layer_kind(other, service_entities)
        if kind is None:
            return unchecked(f"нет данных о слое «{other}»")
        if params.get("geometry_mode") == "source_geometry":
            return PlacementRule(
                strictness=STRICT,
                effect=EFFECT_EXCLUDE_ON,
                requirement=f"не размещать на {_LAYER_PREPOSITIONAL[kind]}",
                layer=kind,
                **base,
            )
        distance = _distance(params)
        if distance is None:
            return unchecked("в норме не указано расстояние")
        return PlacementRule(
            strictness=STRICT,
            effect=EFFECT_EXCLUDE_NEAR,
            requirement=f"не ближе {_number(distance)} м от {_LAYER_GENITIVE[kind]}",
            layer=kind,
            distance_m=distance,
            **base,
        )
    if template in ("presence_within", "accessibility_within"):
        objects = [layers.get(str(params.get("objects_layer") or ""), "")]
        neighbors = [
            layers.get(str(role), "")
            for role in params.get("required_neighbor_layers") or []
        ]
        other = _counterpart(objects, neighbors, service_entities)
        if other is None:
            return unchecked("норма не относится к размещаемому объекту")
        if layer_kind(other, service_entities) != LAYER_HOUSING:
            return unchecked(f"нет данных о слое «{other}»")
        distance = _distance(params)
        if distance is None:
            return unchecked("доступность задана не расстоянием")
        return PlacementRule(
            strictness=SOFT,
            effect=EFFECT_PREFER_NEAR,
            requirement=(
                f"в пределах {_number(distance)} м от "
                f"{_LAYER_GENITIVE[LAYER_HOUSING]}"
            ),
            layer=LAYER_HOUSING,
            distance_m=distance,
            **base,
        )
    return unchecked(f"шаблон проверки «{template}» не поддерживается")
